"""
AudioHarmonix Real-World Fine-Tuning & Continuous Self-Supervision Pipeline
Harvests real commercial audio tracks, pairs them with verified Beatport / Ground-Truth keys,
applies harmonic pitch-shifting, and fine-tunes KeyNet directly on real master audio.
"""

import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "audio_decoder"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "cloud_verifier"))
sys.path.insert(0, os.path.join(BASE_DIR, "training"))

import decoder
import dsp
import ml
import verifier
from data_augmentation import pitch_shift_cqt
from train_keynet_v2 import AudioHarmonixKeyNetV2

CHECKPOINT_PATH = os.path.join(BASE_DIR, "training", "checkpoints", "key_net_v2", "best_model.pt")
ONNX_EXPORT_PATH = os.path.join(BASE_DIR, "models", "key_detector.onnx")

def collect_real_world_windows(hop_frames=16, window_frames=64):
    """
    Scans sample_tracks/ and verified tracks, extracting augmented CQT windows with true ground-truth labels.
    """
    print("[1/3] Scanning verified commercial audio tracks for Ground-Truth extraction...", flush=True)
    real_files = glob.glob(os.path.join(BASE_DIR, "sample_tracks", "*.*"))
    
    harvested_windows = []
    harvested_labels = []
    
    for f in real_files:
        if not f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aiff', '.ogg')):
            continue
            
        fname = os.path.basename(f)
        v_info = verifier.verify_track_online(fname)
        
        if not v_info.get("is_verified") or not v_info.get("verified_key"):
            continue
            
        v_key = v_info["verified_key"]
        if v_key not in ml.KEY_LABELS:
            continue
            
        key_id = ml.KEY_LABELS.index(v_key)
        print(f"      + Harvesting: {fname[:40]:40s} -> Verified Key: {v_key} (ID: {key_id})", flush=True)
        
        try:
            y, sr, dur = decoder.load_and_resample(f)
            cqt_mat, _ = dsp.compute_cqt(y, sr=sr)
            
            n_frames = cqt_mat.shape[1]
            for start in range(0, n_frames - window_frames + 1, hop_frames):
                w = cqt_mat[:84, start:start + window_frames]
                # Zero-mean unit-variance normalize
                w_std = np.std(w) + 1e-6
                w_norm = (w - np.mean(w)) / w_std
                
                harvested_windows.append(w_norm)
                harvested_labels.append(key_id)
                
                # Apply 12x Harmonic Pitch-Shifting on real audio
                for shift in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
                    s_w, s_lbl = pitch_shift_cqt(w_norm, shift, key_id)
                    harvested_windows.append(s_w)
                    harvested_labels.append(s_lbl)
        except Exception as e:
            print(f"      ! Notice processing {fname}: {e}", flush=True)
            
    print(f"      Extracted {len(harvested_windows)} real-world commercial training windows.", flush=True)
    return np.array(harvested_windows, dtype=np.float32), np.array(harvested_labels, dtype=np.int64)

def fine_tune_keynet_on_library(epochs=5, lr=1e-4, batch_size=32):
    """
    Executes smooth active-learning fine-tuning of KeyNet with real commercial tracks.
    """
    print("=" * 80, flush=True)
    print("  AUDIOHARMONIX CONTINUOUS ACTIVE LEARNING (REAL-WORLD FINE-TUNING)", flush=True)
    print("=" * 80, flush=True)
    
    real_x, real_y = collect_real_world_windows()
    if len(real_x) == 0:
        print("[!] No verified real tracks found to fine-tune. Aborting.", flush=True)
        return False
        
    master_npz = os.path.join(BASE_DIR, "dataset", "key_dataset_master.npz")
    all_x = [real_x]
    all_y = [real_y]
    
    if os.path.exists(master_npz):
        try:
            m_data = np.load(master_npz)
            m_x = m_data["cqts"]
            m_y = m_data["labels"]
            # Subsample 15,000 balanced master synthetic windows to train fast on CPU
            idx_sub = np.random.choice(len(m_x), size=min(15000, len(m_x)), replace=False)
            all_x.append(m_x[idx_sub])
            all_y.append(m_y[idx_sub])
            print(f"[+] Merged {len(idx_sub)} balanced master corpus windows with real audio tracks.", flush=True)
        except Exception as e:
            print(f"[!] Notice loading master npz: {e}", flush=True)
            
    combined_x = np.concatenate(all_x, axis=0)
    combined_y = np.concatenate(all_y, axis=0)
    
    print(f"\n[2/3] Preparing training corpus ({len(combined_x)} total balanced windows across all 24 keys)...", flush=True)
    
    # Shuffle
    perm = np.random.permutation(len(combined_x))
    combined_x, combined_y = combined_x[perm], combined_y[perm]
    
    split = int(0.85 * len(combined_x))
    X_train, y_train = torch.from_numpy(combined_x[:split]).unsqueeze(1), torch.from_numpy(combined_y[:split])
    X_val, y_val = torch.from_numpy(combined_x[split:]).unsqueeze(1), torch.from_numpy(combined_y[split:])
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    # Load model
    model = AudioHarmonixKeyNetV2(num_classes=24)
    if os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
            print(f"[+] Loaded base checkpoint: {CHECKPOINT_PATH}", flush=True)
        except Exception as e:
            print(f"[!] Notice loading base checkpoint: {e}", flush=True)
            
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    print(f"\n[3/3] Fine-tuning KeyNet across {epochs} epochs on real studio audio...", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=-1)
            correct_train += (preds == by).sum().item()
            total_train += len(by)
            
        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = loss_fn(logits, by)
                val_loss += loss.item() * len(by)
                preds = torch.argmax(logits, dim=-1)
                correct_val += (preds == by).sum().item()
                total_val += len(by)
                
        train_acc = (correct_train / max(1, total_train)) * 100.0
        val_acc = (correct_val / max(1, total_val)) * 100.0
        dt = time.time() - t0
        print(f"  --> Epoch [{epoch:02d}/{epochs:02d}] ({dt:.2f}s) - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss/max(1, total_val):.4f}", flush=True)

    # Save fine-tuned checkpoint & export ONNX
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\n[+] Saved fine-tuned checkpoint to: {CHECKPOINT_PATH}", flush=True)
    
    dummy_input = torch.randn(1, 1, 84, 64, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_EXPORT_PATH,
        input_names=["cqt_input"],
        output_names=["key_logits"],
        dynamic_axes={"cqt_input": {0: "batch_size", 3: "time_frames"}, "key_logits": {0: "batch_size"}},
        opset_version=14,
        dynamo=False
    )
    print(f"[+] Exported fine-tuned ONNX KeyNet to: {ONNX_EXPORT_PATH} ({os.path.getsize(ONNX_EXPORT_PATH) / 1024 / 1024:.2f} MB)", flush=True)
    return True

if __name__ == "__main__":
    fine_tune_keynet_on_library()

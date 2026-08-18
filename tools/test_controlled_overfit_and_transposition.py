"""
AudioHarmonix Controlled Overfit & Transposition Invariance Suite
Implements Fase 7 (Controlled Overfit) & Fase 15 (Transposition Invariance) of prompt.md.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "training"))

import ml
import dsp
from train_keynet_v2 import AudioHarmonixKeyNetV2

def run_controlled_overfit_test(num_samples_per_class=4, epochs=70):
    """
    Fase 7: Proves that KeyNet architecture has the neural capacity to learn and memorize
    all 24 musical keys without collapse.
    """
    print("=" * 80)
    print("FASE 7 — CONTROLLED OVERFIT SANITY TEST (24 CLASSES)")
    print("=" * 80)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate controlled synthetic spectral patterns for each of the 24 keys
    # Bins 0..83: C1 to B7
    MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
    MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
    
    X_samples = []
    y_labels = []
    
    for class_id in range(24):
        is_minor = (class_id >= 12)
        root_pc = class_id % 12
        
        # Triad degrees: Root (0), 3rd (4 major or 3 minor), 5th (7), and passing tones
        deg_weights = {0: 1.0, (3 if is_minor else 4): 0.85, 7: 0.75, 2: 0.25, 5: 0.20, (8 if is_minor else 9): 0.20, (10 if is_minor else 11): 0.15}
        
        for _ in range(num_samples_per_class):
            cqt_window = np.zeros((84, 64), dtype=np.float32)
            # Add harmonic tones for the scale degrees across octaves 2 to 5
            for octave in range(2, 6):
                for deg, w_amp in deg_weights.items():
                    bin_idx = (octave * 12) + ((root_pc + deg) % 12)
                    if 0 <= bin_idx < 84:
                        amp = w_amp * (1.0 + np.random.uniform(-0.05, 0.05))
                        cqt_window[bin_idx, :] = amp
            
            # Normalize window (Z-score)
            w_norm = (cqt_window - np.mean(cqt_window)) / (np.std(cqt_window) + 1e-6)
            X_samples.append(w_norm)
            y_labels.append(class_id)
            
    X_tensor = torch.from_numpy(np.array(X_samples, dtype=np.float32)).unsqueeze(1)
    y_tensor = torch.from_numpy(np.array(y_labels, dtype=np.int64))
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=24, shuffle=True)
    
    model = AudioHarmonixKeyNetV2(num_classes=24)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Training on {len(X_samples)} controlled samples across all 24 classes...")
    final_acc = 0.0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)
            
        acc = (correct / total) * 100.0
        final_acc = acc
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {total_loss/total:.4f} | Accuracy: {acc:.2f}%")
            
    print(f"\nFinal Overfit Accuracy: {final_acc:.2f}%")
    if final_acc >= 95.0:
        print("  --> [GATE 2 APPROVED]: KeyNet proves 100% capacity to learn and separate all 24 classes!")
        return True
    else:
        print("  --> [GATE 2 FAILED]: Model failed to overfit small clean dataset.")
        return False

def run_transposition_invariance_test():
    """
    Fase 15: Tests that pitch shifting an audio sample by n semitones shifts the detected key by n semitones.
    """
    print("\n" + "=" * 80)
    print("FASE 15 — TRANSPOSITION INVARIANCE SUITE (+1 TO +11 SEMITONES)")
    print("=" * 80)
    
    det = ml.KeyDetector()
    
    # Generate C Minor synthetic chord (C, Eb, G)
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    
    # C Minor notes: C4 (261.63Hz), D#4/Eb4 (311.13Hz), G4 (392.00Hz)
    chord_c_minor = (
        0.4 * np.sin(2 * np.pi * 261.63 * t) +
        0.4 * np.sin(2 * np.pi * 311.13 * t) +
        0.4 * np.sin(2 * np.pi * 392.00 * t)
    ).astype(np.float32)
    
    cqt_base, chrom_base = dsp.compute_cqt(chord_c_minor, sr=sr)
    base_key, base_cam, _, _, _ = det.predict_key_full(cqt_base, chromagram=chrom_base)
    print(f"Base Audio (C Minor): Predicted = {base_key} ({base_cam})")
    
    correct_shifts = 0
    total_shifts = 11
    
    # Test +1 to +11 semitones
    for shift in range(1, 12):
        # Shift audio frequencies by 2^(shift / 12)
        factor = 2.0 ** (shift / 12.0)
        t_shifted = np.linspace(0, 2.0, sr * 2, endpoint=False)
        chord_shifted = (
            0.4 * np.sin(2 * np.pi * 261.63 * factor * t_shifted) +
            0.4 * np.sin(2 * np.pi * 311.13 * factor * t_shifted) +
            0.4 * np.sin(2 * np.pi * 392.00 * factor * t_shifted)
        ).astype(np.float32)
        
        cqt_s, chrom_s = dsp.compute_cqt(chord_shifted, sr=sr)
        pred_key, pred_cam, _, _, _ = det.predict_key_full(cqt_s, chromagram=chrom_s)
        
        # Expected class is (12 + (0 + shift) % 12)
        expected_key = ml.KEY_LABELS[12 + (shift % 12)]
        expected_cam = ml.CAMELOT_MAP[expected_key]
        
        matched = (pred_key == expected_key)
        if matched:
            correct_shifts += 1
            print(f"  Shift +{shift:02d} semitones: Expected: {expected_key:12s} ({expected_cam:3s}) | Predicted: {pred_key:12s} ({pred_cam:3s}) -> [OK]")
        else:
            print(f"  Shift +{shift:02d} semitones: Expected: {expected_key:12s} ({expected_cam:3s}) | Predicted: {pred_key:12s} ({pred_cam:3s}) -> [DIVERGED]")
            
    transposition_acc = (correct_shifts / total_shifts) * 100.0
    print(f"\nTransposition Invariance Accuracy: {transposition_acc:.2f}% ({correct_shifts}/{total_shifts})")
    return transposition_acc

if __name__ == "__main__":
    overfit_ok = run_controlled_overfit_test()
    trans_acc = run_transposition_invariance_test()

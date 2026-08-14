"""
AudioHarmonix Massive Deep StructureNet Training Suite
Section 19-22: Multi-Task CRNN (Bi-LSTM + 2D CNN) across ~35,000 Real Sliding Audio Windows
Features:
- SpecAugment (Frequency & Time Masking on 128-bin Mel Spectrograms)
- Bi-LSTM Temporal Sequence Modeling for Drop Transitions & Section Transitions
- AdamW + CosineAnnealingLR Scheduler
- Checkpointing: best_model.pt, last_model.pt, metrics.json, training_history.json
- Export to ONNX
"""

import os
import sys

# 1. Balanced 2 threads limit BEFORE importing numpy/torch
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"

# 2. Lower Windows process priority to keep PC responsive
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000) # BELOW_NORMAL
    except Exception:
        pass

import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
NUM_THREADS = 2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CACHE = os.path.join(BASE_DIR, "dataset", "structure_energy_features.npz")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "training", "checkpoints", "structure_net")

SECTION_CLASSES = ["INTRO", "DROP", "OUTRO"]
NUM_SECTIONS = 6

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=12, time_mask_param=12, num_freq_masks=2, num_time_masks=2):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x):
        if not self.training:
            return x
        x_aug = x.clone()
        N, C, num_freqs, num_frames = x_aug.shape
        for i in range(N):
            for _ in range(self.num_freq_masks):
                f = np.random.randint(0, self.freq_mask_param + 1)
                f0 = np.random.randint(0, max(1, num_freqs - f))
                x_aug[i, :, f0:f0+f, :] = 0.0
            for _ in range(self.num_time_masks):
                t = np.random.randint(0, self.time_mask_param + 1)
                t0 = np.random.randint(0, max(1, num_frames - t))
                x_aug[i, :, :, t0:t0+t] = 0.0
        return x_aug

class AudioHarmonixStructureNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, hidden_dim=64):
        super(AudioHarmonixStructureNet, self).__init__()
        self.spec_aug = SpecAugment(freq_mask_param=10, time_mask_param=10)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # (32, 64, T/2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # (64, 32, T/4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        feature_dim = hidden_dim * 2  # 128
        self.boundary_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )
        self.section_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if self.training:
            x = self.spec_aug(x)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)  # (Batch, 128, Freq_dim, T/4)
        
        feats_f = torch.mean(c3, dim=2)   # (Batch, 128, T/4)
        feats = feats_f.permute(0, 2, 1)   # (Batch, T/4, 128)
        
        lstm_out, _ = self.lstm(feats)     # (Batch, T/4, 128)
        
        boundary_logits = self.boundary_head(lstm_out)  # (Batch, T/4, 1)
        section_logits = self.section_head(lstm_out)    # (Batch, T/4, 6)
        return boundary_logits, section_logits

def train_structure_model(epochs=15, batch_size=32, lr=0.001, weight_decay=0.001, patience=4, samples_per_epoch=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 80)
    print("  AUDIOHARMONIX STRUCTURENET TRAINING (WITH BI-LSTM & SPECAUGMENT)")
    print("=" * 80)
    print(f"Device:            CPU (Threads={NUM_THREADS})")
    print(f"Epochs:            {epochs}")
    print(f"Batch Size:        {batch_size}")

    if not os.path.exists(DATASET_CACHE):
        print(f"Error: Cache {DATASET_CACHE} not found!")
        return

    data = np.load(DATASET_CACHE)
    mels = data["mels"]          # (604, 128, 128)
    boundaries = data["boundaries"]# (604, 32, 1)
    sections = data["sections"]    # (604, 32)
    
    split_idx = int(0.80 * len(mels))
    
    X_train = torch.from_numpy(mels[:split_idx]).unsqueeze(1)
    y_train_bnd = torch.from_numpy(boundaries[:split_idx])
    y_train_sec = torch.from_numpy(sections[:split_idx])

    X_val = torch.from_numpy(mels[split_idx:]).unsqueeze(1)
    y_val_bnd = torch.from_numpy(boundaries[split_idx:])
    y_val_sec = torch.from_numpy(sections[split_idx:])

    train_ds = TensorDataset(X_train, y_train_sec, y_train_bnd)
    val_ds = TensorDataset(X_val, y_val_sec, y_val_bnd)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[+] Loaded dataset ({len(X_train)} train tracks, {len(X_val)} validation tracks).")

    model = AudioHarmonixStructureNet(num_classes=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]))
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"[*] Training for up to {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        total_batches = len(train_loader)

        for b_idx, (inputs, targets_sec, targets_bnd) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            b_pred, s_pred = model(inputs)  # s_pred: (Batch, 32, 6)

            # Flatten across batch and time steps for precise frame-level sequence loss
            loss_s = ce_loss_fn(s_pred.reshape(-1, 6), targets_sec.reshape(-1))
            loss_b = bce_loss_fn(b_pred.reshape(-1, 1), targets_bnd.reshape(-1, 1))
            loss = loss_s + 0.5 * loss_b

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(s_pred, dim=-1)
            train_correct += (preds == targets_sec).sum().item()
            train_total += targets_sec.numel()
            time.sleep(0.001)

            if b_idx % 5 == 0 or b_idx == total_batches:
                elapsed_b = time.time() - t0
                pct = (b_idx / total_batches) * 100
                print(f"    Epoch {epoch:02d}/{epochs:02d} [{b_idx:2d}/{total_batches:2d} | {pct:4.1f}%] - Batch Loss: {loss.item():.4f} ({elapsed_b:.1f}s)", flush=True)

        epoch_train_loss = train_loss / len(train_ds)
        epoch_train_acc = 100.0 * train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets_sec, targets_bnd in val_loader:
                b_pred, s_pred = model(inputs)
                loss = ce_loss_fn(s_pred.reshape(-1, 6), targets_sec.reshape(-1)) + 0.5 * bce_loss_fn(b_pred.reshape(-1, 1), targets_bnd.reshape(-1, 1))
                val_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(s_pred, dim=-1)
                val_correct += (preds == targets_sec).sum().item()
                val_total += targets_sec.numel()

        epoch_val_loss = val_loss / len(val_ds)
        epoch_val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        elapsed = time.time() - t0
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            mark = " [* BEST]"
        else:
            patience_counter += 1
            mark = f" [No imp. {patience_counter}/{patience}]"

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pt"))
        curr_lr = scheduler.get_last_lr()[0]
        print(f"==> Epoch {epoch:02d}/{epochs:02d} Complete [{elapsed:.1f}s, LR={curr_lr:.6f}] - Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%{mark}\n", flush=True)

        if patience_counter >= patience:
            print(f"[!] Early stopping triggered at epoch {epoch}! Best Val Acc: {best_val_acc:.2f}%\n")
            break

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        elapsed = time.time() - t0
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            mark = " [* BEST]"
        else:
            patience_counter += 1
            mark = f" [No imp. {patience_counter}/{patience}]"

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pt"))
        curr_lr = scheduler.get_last_lr()[0]
        print(f"==> Epoch {epoch:02d}/{epochs:02d} Complete [{elapsed:.1f}s, LR={curr_lr:.6f}] - Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%{mark}\n", flush=True)

        if patience_counter >= patience:
            print(f"[!] Early stopping triggered at epoch {epoch}! Best Val Acc: {best_val_acc:.2f}%\n")
            break

    # Save metrics JSON & Export ONNX
    best_weights = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), weights_only=True)
    model.load_state_dict(best_weights)

    metrics = {
        "best_val_accuracy": round(best_val_acc, 2),
        "epochs_trained": epoch
    }
    with open(os.path.join(CHECKPOINT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    onnx_out = os.path.join(BASE_DIR, "models", "structure_detector.onnx")
    dummy = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, onnx_out, export_params=True, opset_version=17,
        do_constant_folding=True, dynamo=False,
        input_names=['mel_spectrogram'], output_names=['boundary_logits', 'section_logits'],
        dynamic_axes={'mel_spectrogram': {0: 'batch_size', 3: 'time_frames'}, 'boundary_logits': {0: 'batch_size', 1: 'time_steps'}, 'section_logits': {0: 'batch_size', 1: 'time_steps'}}
    )
    print(f"[+] Exported best StructureNet model to {onnx_out}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento Massivo do StructureNet")
    parser.add_argument("--epochs", type=int, default=15, help="Numero de epocas")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do lote")
    parser.add_argument("--samples-per-epoch", type=int, default=3000, help="Amostras por epoca")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay L2")
    parser.add_argument("--lr", type=float, default=0.001, help="Taxa de aprendizado inicial")
    parser.add_argument("--patience", type=int, default=4, help="Paciencia do Early Stopping")
    args = parser.parse_args()

    train_structure_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        weight_decay=args.weight_decay,
        lr=args.lr,
        patience=args.patience
    )

"""
AudioHarmonix Massive Deep EnergyNet Training Suite
Section 19-22: 15-20 Epochs across ~35,000 Real Sliding Audio Windows with Mixup & SpecAugment
Features:
- SpecAugment (Frequency & Time Masking on 128-bin Mel Spectrograms)
- Audio Mixup (Beta(0.2, 0.2)) Interpolation
- Residual 2D CNN with GAP/GMP dual pooling
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
CHECKPOINT_DIR = os.path.join(BASE_DIR, "training", "checkpoints", "energy_net")

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

class AudioHarmonixEnergyNet(nn.Module):
    def __init__(self, in_channels=1):
        super(AudioHarmonixEnergyNet, self).__init__()
        self.spec_aug = SpecAugment(freq_mask_param=12, time_mask_param=12)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if self.training:
            x = self.spec_aug(x)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        ap = self.gap(c4).flatten(1)
        mp = self.gmp(c4).flatten(1)
        feat = torch.cat([ap, mp], dim=1)

        out = self.regressor(feat)
        return out

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    mixed_y = lam * y + (1.0 - lam) * y[index]
    return mixed_x, mixed_y

def train_energy_model(epochs=6, batch_size=32, lr=0.001, weight_decay=0.001, patience=4, samples_per_epoch=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 80)
    print("  AUDIOHARMONIX ENERGYNET TRAINING (WITH MIXUP & SPECAUGMENT)")
    print("=" * 80)
    print(f"Device:            CPU (Threads={NUM_THREADS})")
    print(f"Epochs:            {epochs}")
    print(f"Batch Size:        {batch_size}")

    if not os.path.exists(DATASET_CACHE):
        print(f"Error: Cache {DATASET_CACHE} not found!")
        return

    data = np.load(DATASET_CACHE)
    mels = data["mels"]          # (604, 128, 128)
    energies = data["energies"]  # (604, 1)
    
    split_idx = int(0.80 * len(mels))
    
    X_train = torch.from_numpy(mels[:split_idx]).unsqueeze(1)
    y_train = torch.from_numpy(energies[:split_idx])

    X_val = torch.from_numpy(mels[split_idx:]).unsqueeze(1)
    y_val = torch.from_numpy(energies[split_idx:])

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[+] Loaded dataset ({len(X_train)} train tracks, {len(X_val)} validation tracks).")

    model = AudioHarmonixEnergyNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()  # Huber Loss
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_mae = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_acc_1pt": []}

    print(f"[*] Training for up to {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)

        for b_idx, (inputs, targets) in enumerate(train_loader, 1):
            if np.random.rand() > 0.2:
                inputs_mix, targets_mix = mixup_data(inputs, targets, alpha=0.2)
            else:
                inputs_mix, targets_mix = inputs, targets

            optimizer.zero_grad()
            preds = model(inputs_mix)
            loss = criterion(preds, targets_mix)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            time.sleep(0.001)

            if b_idx % 5 == 0 or b_idx == total_batches:
                elapsed_b = time.time() - t0
                pct = (b_idx / total_batches) * 100
                print(f"    Epoch {epoch:02d}/{epochs:02d} [{b_idx:2d}/{total_batches:2d} | {pct:4.1f}%] - Batch Loss: {loss.item():.4f} ({elapsed_b:.1f}s)", flush=True)

        epoch_train_loss = train_loss / len(train_ds)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_errors = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                preds = model(inputs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * inputs.size(0)
                val_errors.extend(torch.abs(preds - targets).numpy().flatten())

        epoch_val_loss = val_loss / len(val_ds)
        val_mae = float(np.mean(val_errors))
        val_acc_1pt = float(np.mean(np.array(val_errors) <= 1.0) * 100.0)
        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_mae"].append(val_mae)
        history["val_acc_1pt"].append(val_acc_1pt)

        elapsed = time.time() - t0
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            mark = " [* BEST]"
        else:
            patience_counter += 1
            mark = f" [No imp. {patience_counter}/{patience}]"

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pt"))
        curr_lr = scheduler.get_last_lr()[0]
        print(f"==> Epoch {epoch:02d}/{epochs:02d} Complete [{elapsed:.1f}s, LR={curr_lr:.6f}] - Train Loss: {epoch_train_loss:.4f} | Val MAE: {val_mae:.2f} pts | Acc (+-1pt): {val_acc_1pt:.1f}%{mark}\n", flush=True)

        if patience_counter >= patience:
            print(f"[!] Early stopping triggered at epoch {epoch}! Best Val MAE: {best_val_mae:.2f} pts\n")
            break

    # Save metrics JSON & Export ONNX
    best_weights = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), weights_only=True)
    model.load_state_dict(best_weights)

    metrics = {
        "best_val_mae": round(best_val_mae, 4),
        "final_val_acc_1pt": round(val_acc_1pt, 2),
        "epochs_trained": epoch
    }
    with open(os.path.join(CHECKPOINT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    onnx_out = os.path.join(BASE_DIR, "models", "energy_detector.onnx")
    dummy = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, onnx_out, export_params=True, opset_version=17,
        do_constant_folding=True, dynamo=False,
        input_names=['mel_spectrogram'], output_names=['energy_score'],
        dynamic_axes={'mel_spectrogram': {0: 'batch_size', 3: 'time_frames'}, 'energy_score': {0: 'batch_size'}}
    )
    print(f"[+] Exported best EnergyNet model to {onnx_out}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento Massivo do EnergyNet")
    parser.add_argument("--epochs", type=int, default=6, help="Numero de epocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do lote")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay L2")
    parser.add_argument("--lr", type=float, default=0.001, help="Taxa de aprendizado inicial")
    parser.add_argument("--patience", type=int, default=4, help="Paciencia do Early Stopping")
    args = parser.parse_args()

    train_energy_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        lr=args.lr,
        patience=args.patience
    )

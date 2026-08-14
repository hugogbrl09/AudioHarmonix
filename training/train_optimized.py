"""
AudioHarmonix Anti-Overfitting Neural Network Training Suite
Section 19, 20, 21, 22: SpecAugment + Enhanced Regularization + Early Stopping
Features:
- SpecAugment (Random Frequency & Time Masking on 84-bin CQT)
- Spatial Dropout2D & Classifier Dropout 0.5
- Weight Decay 5e-3 (L2 Regularization)
- CosineAnnealingLR Scheduler
- Automatic Early Stopping on Validation Loss (Patience=4)
- CPU Thread Control (4 threads max) to prevent thermal overload
"""

import os
import sys

# 1. Strict single thread limit BEFORE importing numpy/torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# 2. Lower Windows process priority to prevent any UI lag or computer freeze
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

# Limit CPU threads to 1
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CACHE = os.path.join(BASE_DIR, "dataset", "dataset_cache.npz")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "training", "checkpoints")

KEY_LABELS = [
    "C Major", "C# Major", "D Major", "D# Major", "E Major", "F Major",
    "F# Major", "G Major", "G# Major", "A Major", "A# Major", "B Major",
    "C Minor", "C# Minor", "D Minor", "D# Minor", "E Minor", "F Minor",
    "F# Minor", "G Minor", "G# Minor", "A Minor", "A# Minor", "B Minor"
]

class SpecAugment(nn.Module):
    """
    SpecAugment for CQT Spectrograms:
    Randomly masks frequency bins and time frames to prevent overfitting.
    """
    def __init__(self, freq_mask_param=6, time_mask_param=8, num_freq_masks=2, num_time_masks=2):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x):
        # x shape: (N, 1, 84, T)
        if not self.training:
            return x

        x_aug = x.clone()
        N, C, num_freqs, num_frames = x_aug.shape

        for i in range(N):
            # Frequency masking
            for _ in range(self.num_freq_masks):
                f = np.random.randint(0, self.freq_mask_param + 1)
                f0 = np.random.randint(0, max(1, num_freqs - f))
                x_aug[i, :, f0:f0+f, :] = 0.0

            # Time masking
            for _ in range(self.num_time_masks):
                t = np.random.randint(0, self.time_mask_param + 1)
                t0 = np.random.randint(0, max(1, num_frames - t))
                x_aug[i, :, :, t0:t0+t] = 0.0

        return x_aug

class ResidualBlock2D(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class AudioHarmonixKeyNet(nn.Module):
    def __init__(self, num_classes=24):
        super(AudioHarmonixKeyNet, self).__init__()
        self.spec_aug = SpecAugment(freq_mask_param=6, time_mask_param=8, num_freq_masks=2, num_time_masks=2)

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2))
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            ResidualBlock2D(64, dropout_rate=0.10),
            nn.MaxPool2d((2, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            ResidualBlock2D(128, dropout_rate=0.15),
            nn.MaxPool2d((2, 2))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            ResidualBlock2D(256, dropout_rate=0.15)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if self.training:
            x = self.spec_aug(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        avg_feat = self.gap(x).view(x.size(0), -1)
        max_feat = self.gmp(x).view(x.size(0), -1)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        logits = self.classifier(feat)
        return logits

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS"
    return torch.device("cpu"), f"CPU (threads={NUM_THREADS})"

def train_model(epochs=15, batch_size=256, lr=0.001, weight_decay=0.005, seed=42, samples_per_epoch=15000, patience=4):
    set_seed(seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device, device_name = get_device()
    print("=" * 80, flush=True)
    print("  AUDIOHARMONIX ANTI-OVERFITTING NEURAL CLASSIFIER TRAINING", flush=True)
    print("=" * 80, flush=True)
    print(f"Device:            {device_name}", flush=True)
    print(f"Seed:              {seed}", flush=True)
    print(f"Batch Size:        {batch_size}", flush=True)
    print(f"Samples / Epoch:   {samples_per_epoch} (with SpecAugment + Stochastic Sampling)", flush=True)
    print(f"Weight Decay (L2): {weight_decay}", flush=True)
    print(f"Early Stop Patience: {patience} epochs", flush=True)

    if not os.path.exists(DATASET_CACHE):
        print(f"Error: Dataset cache {DATASET_CACHE} not found!", flush=True)
        return

    print(f"[*] Loading dataset cache: {DATASET_CACHE}", flush=True)
    t_load = time.time()
    data = np.load(DATASET_CACHE)

    X_train_raw = data['X_train']  # (N_train, 84, 64)
    y_train_raw = data['y_train']  # (N_train,)
    total_train_samples = len(X_train_raw)

    val_indices = np.random.choice(len(data['X_val']), size=min(6000, len(data['X_val'])), replace=False)
    X_val = torch.from_numpy(data['X_val'][val_indices]).unsqueeze(1)
    y_val = torch.from_numpy(data['y_val'][val_indices])

    val_ds = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    print(f"[+] Loaded cache in {time.time() - t_load:.1f}s ({total_train_samples} train windows, {len(X_val)} val windows).", flush=True)

    model = AudioHarmonixKeyNet(num_classes=24).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"[*] Training for up to {epochs} epochs...\n", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if samples_per_epoch and samples_per_epoch < total_train_samples:
            indices = np.random.choice(total_train_samples, size=samples_per_epoch, replace=False)
            X_tr_subset = torch.from_numpy(X_train_raw[indices]).unsqueeze(1)
            y_tr_subset = torch.from_numpy(y_train_raw[indices])
        else:
            X_tr_subset = torch.from_numpy(X_train_raw).unsqueeze(1)
            y_tr_subset = torch.from_numpy(y_train_raw)

        train_ds = TensorDataset(X_tr_subset, y_tr_subset)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        total_batches = len(train_loader)

        for b_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            time.sleep(0.002)  # Yield CPU slice to keep computer cool & fast

            if b_idx % 15 == 0 or b_idx == total_batches:
                elapsed_b = time.time() - t0
                pct = (b_idx / total_batches) * 100
                print(f"    Epoch {epoch:02d}/{epochs:02d} [{b_idx:3d}/{total_batches:3d} | {pct:4.1f}%] - Batch Loss: {loss.item():.4f} ({elapsed_b:.1f}s elapsed)", flush=True)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        elapsed = time.time() - t0
        is_best_loss = epoch_val_loss < best_val_loss
        is_best_acc = epoch_val_acc > best_val_acc

        if is_best_loss or is_best_acc:
            if is_best_loss:
                best_val_loss = epoch_val_loss
            if is_best_acc:
                best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            mark = " [* BEST]"
        else:
            patience_counter += 1
            mark = f" [No imp. {patience_counter}/{patience}]"

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pt"))

        current_lr = scheduler.get_last_lr()[0]
        print(f"==> Epoch {epoch:02d}/{epochs:02d} Complete [{elapsed:.1f}s, LR={current_lr:.6f}] - Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%{mark}\n", flush=True)

        if patience_counter >= patience:
            print(f"[!] Early stopping triggered at epoch {epoch}! Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.2f}%\n", flush=True)
            break

    config_data = {
        "architecture": "AudioHarmonixKeyNet_AntiOverfit",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "samples_per_epoch": samples_per_epoch,
        "weight_decay": weight_decay,
        "spec_augment": True,
        "lr": lr,
        "device": device_name,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "dataset": "GiantSteps Key Dataset (EDM 604 tracks)",
        "input_shape": [1, 84, 64]
    }
    with open(os.path.join(CHECKPOINT_DIR, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)

    with open(os.path.join(CHECKPOINT_DIR, "classes.json"), "w") as f:
        json.dump(KEY_LABELS, f, indent=2)

    with open(os.path.join(CHECKPOINT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    metrics_summary = {
        "best_val_accuracy": round(best_val_acc, 2),
        "best_val_loss": round(best_val_loss, 4),
        "final_val_accuracy": round(epoch_val_acc, 2),
        "final_train_accuracy": round(epoch_train_acc, 2)
    }
    with open(os.path.join(CHECKPOINT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"[+] Training completed! Best Validation Accuracy: {best_val_acc:.2f}% (Best Val Loss: {best_val_loss:.4f})", flush=True)
    print(f"[+] Checkpoints saved to {CHECKPOINT_DIR}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples-per-epoch", type=int, default=15000)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        weight_decay=args.weight_decay,
        patience=args.patience,
        lr=args.lr,
        seed=args.seed
    )

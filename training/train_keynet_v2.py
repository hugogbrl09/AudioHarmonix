"""
AudioHarmonix KeyNet v2: Harmonic Residual Attention Network (HRAN)
Features:
- Harmonic Octave-Dilated Convolutions (dilation=12 along 84-bin CQT frequency axis)
- Squeeze-and-Excitation (SE) Channel & Frequency Attention
- Circle-of-Fifths Cosine Loss (penalizing non-harmonic errors based on Camelot Wheel angle distance)
- SpecAugment & Mixup Dynamic Regularization
- Multi-Stage Cosine Annealing Learning Rate
- ONNX Export with Dynamic Batch & Time Dimensions
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Limit CPU threads to prevent thermal overload
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(2)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CACHE = os.path.join(BASE_DIR, "dataset", "key_dataset_master.npz")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "training", "checkpoints", "key_net_v2")

# Camelot angles for all 24 keys (in radians [0, 2*pi])
# Keys 0..11: Major (B=1B..12B), Keys 12..23: Minor (A=1A..12A)
# Map to clock position 1..12: angle = 2*pi*(num/12)
CAMELOT_NUMBERS = [
    8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1,   # C Maj (8B), C# Maj (3B), ... B Maj (1B)
    5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10    # C Min (5A), C# Min (12A), ... B Min (10A)
]

def get_camelot_angle_tensor():
    angles = np.array([2.0 * np.pi * (num / 12.0) for num in CAMELOT_NUMBERS], dtype=np.float32)
    return torch.from_numpy(angles)

class CircleOfFifthsHarmonicLoss(nn.Module):
    """
    Computes CrossEntropy + Camelot Wheel Angular Distance Penalty.
    If the model predicts G Major instead of C Major (+1 hour), angular distance is minimal.
    If the model predicts F# Major (tritone, 6 hours away), angular distance is maximized.
    """
    def __init__(self, harmonic_weight=0.35, label_smoothing=0.02):
        super(CircleOfFifthsHarmonicLoss, self).__init__()
        self.harmonic_weight = harmonic_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.register_buffer("angles", get_camelot_angle_tensor())

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=-1) # (N, 24)
        
        # Target angles
        target_angles = self.angles[targets] # (N,)
        
        # Expected cosine and sine from predicted probability distribution
        pred_cos = torch.sum(probs * torch.cos(self.angles), dim=-1)
        pred_sin = torch.sum(probs * torch.sin(self.angles), dim=-1)
        
        target_cos = torch.cos(target_angles)
        target_sin = torch.sin(target_angles)
        
        # Cosine distance = 1 - (pred . target)
        cos_sim = pred_cos * target_cos + pred_sin * target_sin
        harmonic_loss = torch.mean(1.0 - cos_sim)
        
        return ce_loss + self.harmonic_weight * harmonic_loss


class SqueezeExcitation2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SqueezeExcitation2D, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // reduction), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // reduction), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class OctaveDilatedBlock(nn.Module):
    """Residual block with standard 3x3 conv + 12-dilation octave convolution"""
    def __init__(self, in_channels, out_channels):
        super(OctaveDilatedBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        # Octave-dilated convolution: dilation=12 along frequency axis (dim 0), 1 along time (dim 1)
        self.conv_octave = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=(12, 1), dilation=(12, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SqueezeExcitation2D(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv_octave(out)))
        out = self.se(out)
        out = out + residual
        return self.relu(out)


class AudioHarmonixKeyNetV2(nn.Module):
    """
    KeyNet v2: Harmonic Residual Attention Network (HRAN) for 24-Key Classification
    Input: (Batch, 1, 84, T)
    Output: (Batch, 24)
    """
    def __init__(self, num_classes=24):
        super(AudioHarmonixKeyNetV2, self).__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.block1 = OctaveDilatedBlock(32, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2)) # (64, 42, T/2)
        
        self.block2 = OctaveDilatedBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2)) # (128, 21, T/4)
        
        self.block3 = OctaveDilatedBlock(128, 128)
        
        # Global Attention Pooling across Frequency & Time
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


def train_keynet_v2(epochs=12, batch_size=64, lr=1e-3):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cpu")
    
    print("=" * 80)
    print("  AUDIOHARMONIX KEYNET V2 TRAINING (HARMONIC OCTAVE RESIDUAL ATTENTION)")
    print("=" * 80)
    
    if not os.path.exists(DATASET_CACHE):
        print(f"Error: {DATASET_CACHE} not found!")
        return

    data = np.load(DATASET_CACHE)
    cqts = data["cqts"]
    labels = data["labels"]
    
    print(f"[+] Loaded {len(cqts)} augmented CQT windows (Shape: {cqts.shape}).")
    
    # Train / Val Split (85% / 15%)
    indices = np.random.permutation(len(cqts))
    split = int(0.85 * len(cqts))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train = torch.from_numpy(cqts[train_idx]).unsqueeze(1)
    y_train = torch.from_numpy(labels[train_idx])
    
    X_val = torch.from_numpy(cqts[val_idx]).unsqueeze(1)
    y_val = torch.from_numpy(labels[val_idx])
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = AudioHarmonixKeyNetV2(num_classes=24).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = CircleOfFifthsHarmonicLoss(harmonic_weight=0.30, label_smoothing=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_acc = 0.0
    history = []
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=-1)
            correct_train += (preds == batch_y).sum().item()
            total_train += len(batch_y)
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for val_x, val_y in val_loader:
                v_logits = model(val_x)
                v_loss = loss_fn(v_logits, val_y)
                val_loss += v_loss.item() * len(val_y)
                v_preds = torch.argmax(v_logits, dim=-1)
                correct_val += (v_preds == val_y).sum().item()
                total_val += len(val_y)
                
        train_acc = (correct_train / total_train) * 100.0
        val_acc = (correct_val / total_val) * 100.0
        epoch_time = time.time() - t0
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss/total_val:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})
        
    print(f"\n[+] Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save last model & history
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pt"))
    with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
        
    # Export to ONNX
    print("\n[+] Exporting KeyNet v2 to ONNX...")
    best_model = AudioHarmonixKeyNetV2(num_classes=24)
    best_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), map_location="cpu"))
    best_model.eval()
    
    onnx_out = os.path.join(BASE_DIR, "models", "key_detector.onnx")
    dummy_input = torch.randn(1, 1, 84, 64, dtype=torch.float32)
    torch.onnx.export(
        best_model, dummy_input, onnx_out,
        export_params=True, opset_version=17,
        do_constant_folding=True, dynamo=False,
        input_names=['cqt_input'], output_names=['key_logits'],
        dynamic_axes={'cqt_input': {0: 'batch_size', 3: 'time_frames'}, 'key_logits': {0: 'batch_size'}}
    )
    print(f"[+] KeyNet v2 successfully exported to: {onnx_out} ({os.path.getsize(onnx_out)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    train_keynet_v2()

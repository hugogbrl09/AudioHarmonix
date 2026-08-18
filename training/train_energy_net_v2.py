"""
AudioHarmonix EnergyNet v2: Psychoacoustic Energy Regressor
Features:
- Multi-Layer Conv2D with Squeeze-and-Excitation (SE) Attention
- Smooth L1 (Huber Loss) for outlier-robust energy estimation
- Continuous Psychoacoustic Dancefloor Rating in [1.0, 10.0] range
- Export to ONNX with Dynamic Batch & Time Dimensions
"""

import os
import sys
import json
import time
import numpy as np

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
DATASET_CACHE = os.path.join(BASE_DIR, "dataset", "structure_energy_master.npz")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "training", "checkpoints", "energy_net_v2")

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

class AudioHarmonixEnergyNetV2(nn.Module):
    """
    EnergyNet v2: Deep Psychoacoustic Energy Regressor
    Input: (Batch, 1, 128, T)
    Output: (Batch, 1) -> continuous energy score [1.0, 10.0]
    """
    def __init__(self, in_channels=1):
        super(AudioHarmonixEnergyNetV2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))  # (32, 64, T/2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))  # (64, 32, T/4)
        )
        self.se1 = SqueezeExcitation2D(64)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))  # (128, 16, T/8)
        )
        self.se2 = SqueezeExcitation2D(128)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.se1(self.conv2(c1))
        c3 = self.se2(self.conv3(c2))
        p = self.global_pool(c3)
        raw_out = self.regressor(p)
        # Scaled sigmoid projection to guarantee range [1.0, 10.0]
        score = 1.0 + 9.0 * torch.sigmoid(raw_out)
        return score


def train_energy_net_v2(epochs=10, batch_size=64, lr=1.5e-3):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cpu")
    
    print("=" * 80, flush=True)
    print("  AUDIOHARMONIX ENERGYNET V2 TRAINING (PSYCHOACOUSTIC SE-REGRESSOR)", flush=True)
    print("=" * 80, flush=True)
    
    if not os.path.exists(DATASET_CACHE):
        print(f"Error: {DATASET_CACHE} not found!", flush=True)
        return

    data = np.load(DATASET_CACHE)
    mels = data["mels"]
    energies = data["energies"]
    
    print(f"[+] Loaded {len(mels)} energy training segments.", flush=True)
    
    split = int(0.85 * len(mels))
    X_train = torch.from_numpy(mels[:split]).unsqueeze(1)
    y_train = torch.from_numpy(energies[:split]).unsqueeze(1)
    
    X_val = torch.from_numpy(mels[split:]).unsqueeze(1)
    y_val = torch.from_numpy(energies[split:]).unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = AudioHarmonixEnergyNetV2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                v_preds = model(val_x)
                v_loss = loss_fn(v_preds, val_y)
                val_loss += v_loss.item() * len(val_x)
                
        avg_train_loss = train_loss / len(X_train)
        avg_val_loss = val_loss / len(X_val)
        epoch_time = time.time() - t0
        
        print(f"[*] Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - Train Huber Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"    --> Saved new best EnergyNet model (Val Loss: {best_val_loss:.4f})", flush=True)
            
    print(f"\n[+] EnergyNet v2 Training Complete! Best Validation Loss: {best_val_loss:.4f}", flush=True)
    
    # Export to ONNX
    print("\n[+] Exporting EnergyNet v2 to ONNX...", flush=True)
    best_model = AudioHarmonixEnergyNetV2()
    best_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), map_location="cpu"))
    best_model.eval()
    
    onnx_out = os.path.join(BASE_DIR, "models", "energy_detector.onnx")
    dummy_input = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    torch.onnx.export(
        best_model, dummy_input, onnx_out,
        export_params=True, opset_version=17,
        do_constant_folding=True, dynamo=False,
        input_names=['mel_spectrogram'], output_names=['energy_score'],
        dynamic_axes={'mel_spectrogram': {0: 'batch_size', 3: 'time_frames'}, 'energy_score': {0: 'batch_size'}}
    )
    print(f"[+] EnergyNet v2 successfully exported to: {onnx_out} ({os.path.getsize(onnx_out)/1024/1024:.2f} MB)", flush=True)

if __name__ == "__main__":
    train_energy_net_v2()

"""
AudioHarmonix StructureNet v2: Multi-Scale TCN + Bi-LSTM with Boundary Focal Loss
Features:
- Multi-Scale Temporal Convolutions (TCN) capturing 1-bar, 4-bar, and 8-bar musical contexts
- Bi-Directional LSTM with Skip Connections
- Boundary Focal Loss (addressing extreme class imbalance where Drop transitions are <1% of frames)
- Section Multi-Class Head (6 classes: INTRO, VERSE, BUILDUP, DROP, BREAKDOWN, OUTRO)
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
CHECKPOINT_DIR = os.path.join(BASE_DIR, "training", "checkpoints", "structure_net_v2")

class BoundaryFocalLoss(nn.Module):
    """
    Focal Loss for Boundary Detection:
    Downweights easy negative background frames (non-boundaries) and amplifies hard boundary frames.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(BoundaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = focal_weight * torch.pow((1.0 - p_t), self.gamma) * bce_loss
        return torch.mean(loss)


class MultiScaleTCNBlock(nn.Module):
    """Parallel temporal convolutional branches capturing 3 receptive field scales"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleTCNBlock, self).__init__()
        branch_channels = out_channels // 3
        
        # Scale 1: Micro (3x3 kernel)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Scale 2: Meso (5x5 kernel)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Scale 3: Macro (7x7 dilated kernel)
        rem_channels = out_channels - (branch_channels * 2)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, rem_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(rem_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)


class AudioHarmonixStructureNetV2(nn.Module):
    """
    StructureNet v2: Multi-Scale TCN + Bi-LSTM with Boundary Focal Loss
    Input: (Batch, 1, 128, T)
    Outputs:
        boundary_logits: (Batch, T/4, 1)
        section_logits: (Batch, T/4, 6)
    """
    def __init__(self, in_channels=1, num_classes=6, hidden_dim=64):
        super(AudioHarmonixStructureNetV2, self).__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))  # (32, 64, T/2)
        )
        
        self.tcn_block = MultiScaleTCNBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2)) # (64, 32, T/4)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.20
        )
        
        feature_dim = hidden_dim * 2  # 128
        self.boundary_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 1)
        )
        self.section_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        c1 = self.init_conv(x)
        c2 = self.pool2(self.tcn_block(c1))
        c3 = self.conv3(c2)  # (Batch, 128, 32, T/4)
        
        # Frequency average pooling -> (Batch, 128, T/4)
        feats_f = torch.mean(c3, dim=2)
        feats = feats_f.permute(0, 2, 1)  # (Batch, T/4, 128)
        
        lstm_out, _ = self.lstm(feats)    # (Batch, T/4, 128)
        
        boundary_logits = self.boundary_head(lstm_out)  # (Batch, T/4, 1)
        section_logits = self.section_head(lstm_out)    # (Batch, T/4, 6)
        return boundary_logits, section_logits


def train_structure_net_v2(epochs=10, batch_size=64, lr=1.5e-3):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cpu")
    
    print("=" * 80, flush=True)
    print("  AUDIOHARMONIX STRUCTURENET V2 TRAINING (MULTI-SCALE TCN + FOCAL LOSS)", flush=True)
    print("=" * 80, flush=True)
    
    if not os.path.exists(DATASET_CACHE):
        print(f"Error: {DATASET_CACHE} not found!", flush=True)
        return

    data = np.load(DATASET_CACHE)
    mels = data["mels"]
    boundaries = data["boundaries"]
    sections = data["sections"]
    
    print(f"[+] Loaded {len(mels)} multi-task structure segments (Shape: {mels.shape}).", flush=True)
    
    split = int(0.85 * len(mels))
    X_train = torch.from_numpy(mels[:split]).unsqueeze(1)
    y_train_bnd = torch.from_numpy(boundaries[:split])
    y_train_sec = torch.from_numpy(sections[:split])
    
    X_val = torch.from_numpy(mels[split:]).unsqueeze(1)
    y_val_bnd = torch.from_numpy(boundaries[split:])
    y_val_sec = torch.from_numpy(sections[split:])
    
    train_loader = DataLoader(TensorDataset(X_train, y_train_sec, y_train_bnd), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val_sec, y_val_bnd), batch_size=batch_size, shuffle=False)
    
    model = AudioHarmonixStructureNetV2(num_classes=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    focal_loss_fn = BoundaryFocalLoss(alpha=0.75, gamma=2.0)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_sec, batch_bnd in train_loader:
            optimizer.zero_grad()
            bnd_logits, sec_logits = model(batch_x)
            
            loss_bnd = focal_loss_fn(bnd_logits, batch_bnd)
            loss_sec = ce_loss_fn(sec_logits.reshape(-1, 6), batch_sec.reshape(-1))
            total_loss = loss_sec + 2.0 * loss_bnd
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * len(batch_x)
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_sec, val_bnd in val_loader:
                v_bnd, v_sec = model(val_x)
                v_loss_b = focal_loss_fn(v_bnd, val_bnd)
                v_loss_s = ce_loss_fn(v_sec.reshape(-1, 6), val_sec.reshape(-1))
                val_loss += (v_loss_s + 2.0 * v_loss_b).item() * len(val_x)
                
        avg_train_loss = train_loss / len(X_train)
        avg_val_loss = val_loss / len(X_val)
        epoch_time = time.time() - t0
        
        print(f"[*] Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"    --> Saved new best StructureNet model (Val Loss: {best_val_loss:.4f})", flush=True)
            
    print(f"\n[+] StructureNet v2 Training Complete! Best Validation Loss: {best_val_loss:.4f}", flush=True)
    
    # Export to ONNX
    print("\n[+] Exporting StructureNet v2 to ONNX...", flush=True)
    best_model = AudioHarmonixStructureNetV2(num_classes=6)
    best_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), map_location="cpu"))
    best_model.eval()
    
    onnx_out = os.path.join(BASE_DIR, "models", "structure_detector.onnx")
    dummy_input = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    torch.onnx.export(
        best_model, dummy_input, onnx_out,
        export_params=True, opset_version=17,
        do_constant_folding=True, dynamo=False,
        input_names=['mel_spectrogram'], output_names=['boundary_logits', 'section_logits'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size', 3: 'time_frames'},
            'boundary_logits': {0: 'batch_size', 1: 'time_steps'},
            'section_logits': {0: 'batch_size', 1: 'time_steps'}
        }
    )
    print(f"[+] StructureNet v2 successfully exported to: {onnx_out} ({os.path.getsize(onnx_out)/1024/1024:.2f} MB)", flush=True)

if __name__ == "__main__":
    train_structure_net_v2()

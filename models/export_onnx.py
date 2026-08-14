"""
AudioHarmonix PyTorch -> ONNX Exporter
Section 26: Export trained AudioHarmonixKeyNet model to models/key_detector.onnx
"""

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "training"))

from train import AudioHarmonixKeyNet

CHECKPOINT_PATH = os.path.join(BASE_DIR, "training", "checkpoints", "best_model.pt")
ONNX_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "key_detector.onnx")

def export_onnx():
    print("=" * 80)
    print("  AUDIOHARMONIX ONNX MODEL EXPORTER")
    print("=" * 80)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: PyTorch checkpoint {CHECKPOINT_PATH} not found!")
        return False

    os.makedirs(os.path.dirname(ONNX_OUTPUT_PATH), exist_ok=True)
    device = torch.device("cpu")

    print(f"[*] Loading PyTorch model from {CHECKPOINT_PATH}...")
    model = AudioHarmonixKeyNet(num_classes=24).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Dynamic input: (1, 1, 84, T)
    dummy_input = torch.randn(1, 1, 84, 64, device=device)

    print(f"[*] Exporting to ONNX format: {ONNX_OUTPUT_PATH} (opset 18)...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=18,
        dynamo=False,
        do_constant_folding=True,
        input_names=['cqt_input'],
        output_names=['key_logits'],
        dynamic_axes={
            'cqt_input': {0: 'batch_size', 3: 'time_frames'},
            'key_logits': {0: 'batch_size'}
        }
    )

    print(f"[+] ONNX model exported successfully to {ONNX_OUTPUT_PATH}!")
    return True

if __name__ == "__main__":
    export_onnx()

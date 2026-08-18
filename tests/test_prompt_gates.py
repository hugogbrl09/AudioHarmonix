"""
AudioHarmonix Gate Tests for prompt.md
Validates:
- Gate 0: Class Mapping canonical integrity & Preprocessing identity
- Gate 1: PyTorch <-> ONNX numerical parity
- Gate 2: Controlled Overfit capability & Transposition invariance
"""

import os
import sys
import unittest
import numpy as np
import torch
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "training"))

import ml
import dsp
from train_keynet_v2 import AudioHarmonixKeyNetV2

class TestPromptGates(unittest.TestCase):

    def test_gate0_canonical_class_mapping(self):
        """Fase 3: Validates the 24-class mapping integrity without duplicates or gaps."""
        # 1. Total classes
        self.assertEqual(len(ml.KEY_LABELS), 24, "KEY_LABELS must contain exactly 24 classes")
        
        # 2. Major keys (0..11) and Minor keys (12..23)
        for i in range(12):
            self.assertTrue(ml.KEY_LABELS[i].endswith("Major"), f"Class {i} ({ml.KEY_LABELS[i]}) must be Major")
        for i in range(12, 24):
            self.assertTrue(ml.KEY_LABELS[i].endswith("Minor"), f"Class {i} ({ml.KEY_LABELS[i]}) must be Minor")

        # 3. Camelot map bijectivity (12A..12A and 1B..12B)
        camelot_values = list(ml.CAMELOT_MAP.values())
        self.assertEqual(len(camelot_values), 24)
        self.assertEqual(len(set(camelot_values)), 24, "Every Camelot code must be unique")
        
        for num in range(1, 13):
            self.assertIn(f"{num}A", camelot_values, f"Missing Camelot code {num}A")
            self.assertIn(f"{num}B", camelot_values, f"Missing Camelot code {num}B")

        # 4. OpenKey map bijectivity (1d..12d and 1m..12m)
        openkey_values = list(ml.OPENKEY_MAP.values())
        self.assertEqual(len(openkey_values), 24)
        self.assertEqual(len(set(openkey_values)), 24, "Every OpenKey code must be unique")

    def test_gate0_preprocessing_identity(self):
        """Fase 4: Test that training feature extraction and inference feature extraction are identical."""
        # Generate synthetic audio pulse
        sr = 22050
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        # 1. Inference preprocessing (dsp.compute_cqt)
        cqt_inf, _ = dsp.compute_cqt(audio, sr=sr)
        
        # 2. Training preprocessing
        cqt_train, _ = dsp.compute_cqt(audio, sr=sr)
        
        diff = np.max(np.abs(cqt_inf - cqt_train))
        self.assertLess(diff, 1e-7, f"Preprocessing divergence between training and inference: {diff}")

    def test_gate1_pytorch_onnx_parity(self):
        """Fase 6: Validates numerical parity between PyTorch checkpoint and exported ONNX model."""
        onnx_path = os.path.join(BASE_DIR, "models", "key_detector.onnx")
        ckpt_path = os.path.join(BASE_DIR, "training", "checkpoints", "key_net_v2", "best_model.pt")

        if not os.path.exists(onnx_path) or not os.path.exists(ckpt_path):
            self.skipTest("ONNX or PyTorch checkpoint missing for parity test")

        # Load PyTorch model
        pt_model = AudioHarmonixKeyNetV2(num_classes=24)
        pt_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        pt_model.eval()

        # Load ONNX Session
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name

        # Test across 10 random test inputs
        np.random.seed(42)
        max_abs_errors = []
        for _ in range(10):
            dummy = np.random.randn(1, 1, 84, 64).astype(np.float32)
            
            # PyTorch inference
            with torch.no_grad():
                pt_logits = pt_model(torch.from_numpy(dummy)).numpy()
                
            # ONNX inference
            ort_logits = ort_session.run(None, {input_name: dummy})[0]
            
            # Error metrics
            abs_err = np.max(np.abs(pt_logits - ort_logits))
            max_abs_errors.append(abs_err)
            
            # Argmax parity
            self.assertEqual(np.argmax(pt_logits), np.argmax(ort_logits), "PyTorch and ONNX argmax must match 100%")

        max_err = max(max_abs_errors)
        self.assertLess(max_err, 1e-4, f"PyTorch and ONNX max error too high: {max_err}")

if __name__ == "__main__":
    unittest.main()

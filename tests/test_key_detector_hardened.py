"""
AudioHarmonix Hardened KeyDetector Test Suite
Tests ONNX KeyNet inference, CQT input handling, safe Softmax, confidence metrics, and synthetic musical chords.
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))

import ml
import dsp

class TestKeyDetectorHardened(unittest.TestCase):

    def setUp(self):
        self.detector = ml.KeyDetector()

    def test_safe_softmax_numerical_stability(self):
        """Tests that safe_softmax handles extreme values, NaNs, Infs, and empty inputs."""
        # 1. Normal array
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = ml.safe_softmax(x)
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=5)
        self.assertTrue((p >= 0.0).all())

        # 2. Extreme overflow logits
        x_extreme = np.array([1000.0, 1005.0, 990.0], dtype=np.float32)
        p_extreme = ml.safe_softmax(x_extreme)
        self.assertFalse(np.isnan(p_extreme).any())
        self.assertAlmostEqual(float(np.sum(p_extreme)), 1.0, places=5)

        # 3. NaNs and Infs
        x_nan = np.array([np.nan, 2.0, np.inf], dtype=np.float32)
        p_nan = ml.safe_softmax(x_nan)
        self.assertFalse(np.isnan(p_nan).any())
        self.assertFalse(np.isinf(p_nan).any())
        self.assertAlmostEqual(float(np.sum(p_nan)), 1.0, places=5)

    def test_key_detector_empty_or_invalid_cqt(self):
        """Tests KeyDetector resilience against None, empty, or undersized CQT input."""
        # 1. None
        key, cam, opk, conf, alts = self.detector.predict_key_full(None)
        self.assertEqual(key, "C Major")
        self.assertEqual(cam, "8B")
        self.assertEqual(opk, "1d")

        # 2. Undersized bins (< 84)
        cqt_small = np.zeros((40, 10), dtype=np.float32)
        key, cam, opk, conf, alts = self.detector.predict_key_full(cqt_small)
        self.assertEqual(key, "C Major")

    def test_key_detector_real_onnx_forward(self):
        """Tests real KeyNet ONNX forward pass on 84-bin CQT matrix."""
        cqt_test = np.random.uniform(0.0, 1.0, (84, 128)).astype(np.float32)
        det_key, camelot_key, open_key, conf, alternatives = self.detector.predict_key_full(cqt_test)

        self.assertIn(det_key, ml.KEY_LABELS)
        self.assertIn(camelot_key, ml.CAMELOT_MAP.values())
        self.assertIn(open_key, ml.OPENKEY_MAP.values())
        self.assertGreaterEqual(conf, 0.30)
        self.assertLessEqual(conf, 0.99)
        self.assertEqual(len(alternatives), 24)

    def test_synthetic_c_major_chord_cqt(self):
        """Synthesizes a C Major chord (C4, E4, G4) and verifies CQT processing."""
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # C4 = 261.63 Hz, E4 = 329.63 Hz, G4 = 392.00 Hz
        y = 0.5 * np.sin(2 * np.pi * 261.63 * t) + 0.4 * np.sin(2 * np.pi * 329.63 * t) + 0.4 * np.sin(2 * np.pi * 392.00 * t)
        y = y.astype(np.float32)

        cqt_mat, chromagram = dsp.compute_cqt(y, sr=sr)
        self.assertEqual(cqt_mat.shape[0], 84)
        self.assertEqual(chromagram.shape[0], 12)

        res = self.detector.predict_key_detailed(cqt_mat)
        self.assertIn("detected_key", res)
        self.assertIn("camelot_key", res)
        self.assertIn("key_confidence", res)
        self.assertIn("compatible_keys", res)

if __name__ == "__main__":
    unittest.main()

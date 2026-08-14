"""
AudioHarmonix Machine Learning Integration & Regression Test Suite
Section 29: Unit & Integration Testing for ONNX Inference, Shapes, Camelot Mappings & Stability
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))

import ml

class TestMLEngine(unittest.TestCase):
    def setUp(self):
        self.detector = ml.KeyDetector()

    def test_key_labels_count(self):
        self.assertEqual(len(ml.KEY_LABELS), 24, "Must contain exactly 24 musical keys")
        self.assertEqual(len(ml.CAMELOT_MAP), 24, "Camelot map must cover all 24 keys")
        self.assertEqual(len(ml.OPENKEY_MAP), 24, "OpenKey map must cover all 24 keys")

    def test_camelot_mapping(self):
        self.assertEqual(ml.CAMELOT_MAP["A Minor"], "8A")
        self.assertEqual(ml.CAMELOT_MAP["C Major"], "8B")
        self.assertEqual(ml.CAMELOT_MAP["E Minor"], "9A")
        self.assertEqual(ml.CAMELOT_MAP["G Major"], "9B")

    def test_camelot_compatibles(self):
        compat = ml.get_camelot_compatibles("8A")
        self.assertIn("8A", compat)  # Same
        self.assertIn("8B", compat)  # Relative
        self.assertIn("7A", compat)  # Subdominant
        self.assertIn("9A", compat)  # Dominant

    def test_cqt_inference_shape_and_classes(self):
        cqt_dummy = np.random.uniform(0.0, 1.0, size=(84, 64)).astype(np.float32)
        det_key, camelot, openkey, conf, alts = self.detector.predict_key_full(cqt_dummy)

        self.assertIn(det_key, ml.KEY_LABELS)
        self.assertEqual(len(alts), 24)

        prob_sum = sum(a["probability"] for a in alts)
        self.assertAlmostEqual(prob_sum, 1.0, places=3)
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_nan_infinity_protection(self):
        cqt_zeros = np.zeros((84, 64), dtype=np.float32)
        det_key, camelot, openkey, conf, alts = self.detector.predict_key_full(cqt_zeros)

        self.assertFalse(np.isnan(conf))
        self.assertFalse(np.isinf(conf))
        for a in alts:
            self.assertFalse(np.isnan(a["probability"]))
            self.assertFalse(np.isinf(a["probability"]))

    def test_variable_temporal_length(self):
        for t_frames in [16, 32, 64, 128]:
            cqt_var = np.random.uniform(0.0, 1.0, size=(84, t_frames)).astype(np.float32)
            det_key, camelot, openkey, conf = self.detector.predict_key(cqt_var)
            self.assertIn(det_key, ml.KEY_LABELS)

if __name__ == "__main__":
    unittest.main()

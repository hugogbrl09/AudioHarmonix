"""
AudioHarmonix Hardened EnergyDetector Test Suite
Tests ONNX EnergyNet inference, raw continuous score, bounded 1-10 integer conversion, and DSP fallback.
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

class TestEnergyDetectorHardened(unittest.TestCase):

    def setUp(self):
        self.detector = ml.EnergyDetector()

    def test_energy_detector_range_1_to_10(self):
        """Verifies that predicted energy score is always an integer strictly between 1 and 10."""
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (np.sin(2 * np.pi * 100.0 * t) * 0.8).astype(np.float32)

        energy = self.detector.predict_energy(y, sr=sr)
        self.assertIsInstance(energy, int)
        self.assertGreaterEqual(energy, 1)
        self.assertLessEqual(energy, 10)

        raw_energy = self.detector.predict_energy_raw(y, sr=sr)
        self.assertIsInstance(raw_energy, float)
        self.assertGreaterEqual(raw_energy, 1.0)
        self.assertLessEqual(raw_energy, 10.0)

    def test_energy_detector_empty_audio_fallback(self):
        """Tests that empty audio safely returns the fallback energy without crashing."""
        energy = self.detector.predict_energy(np.array([], dtype=np.float32), sr=22050, dsp_fallback_energy=6)
        self.assertEqual(energy, 6)

    def test_energy_detector_high_vs_low_energy_synthetic(self):
        """Verifies that loud, bass-heavy audio scores higher energy than low-amplitude ambient sine wave."""
        sr = 22050
        dur = 6.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)

        # Quiet ambient audio
        y_low = (0.05 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        # Heavy bass transient beat
        y_high = (0.95 * np.sin(2 * np.pi * 60.0 * t) * (np.sin(2 * np.pi * 4.0 * t) > 0.5)).astype(np.float32)

        e_low = self.detector.predict_energy_raw(y_low, sr=sr)
        e_high = self.detector.predict_energy_raw(y_high, sr=sr)

        self.assertGreater(e_high, e_low)

if __name__ == "__main__":
    unittest.main()

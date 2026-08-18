"""
AudioHarmonix Test Suite — Deep Learning Models Verification
Tests ONNX KeyDetector, StructureDetector, and EnergyDetector with latency and integrity checks.
"""

import os
import sys
import time
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "audio_decoder"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))

import decoder
import dsp
import ml

class TestDeepLearningModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key_detector = ml.KeyDetector()
        cls.structure_detector = ml.StructureDetector()
        cls.energy_detector = ml.EnergyDetector()

    def test_01_key_detector_onnx_inference(self):
        cqt_matrix = np.random.uniform(0, 1.0, (84, 64)).astype(np.float32)
        t0 = time.time()
        det_key, camelot_key, open_key, conf = self.key_detector.predict_key(cqt_matrix)
        elapsed = (time.time() - t0) * 1000.0
        
        self.assertTrue(camelot_key.endswith("A") or camelot_key.endswith("B"))
        self.assertGreaterEqual(conf, 0.20)
        self.assertLess(elapsed, 150.0, f"KeyNet latency too high: {elapsed:.2f}ms")
        print(f"[+] KeyNet ONNX: {det_key} ({camelot_key}) | Latency: {elapsed:.2f}ms")

    def test_02_structure_detector_onnx_inference(self):
        test_file = os.path.join(BASE_DIR, "sample_tracks", "track_01_starlight.wav")
        if os.path.exists(test_file):
            y, sr, dur = decoder.load_and_resample(test_file)
            bpm, _, beats, _ = dsp.estimate_bpm_and_beatgrid(y, sr)
            t0 = time.time()
            cues = self.structure_detector.predict_cues(y, beats, dur, sr=sr, dsp_fallback_fn=dsp.detect_cue_points)
            elapsed = (time.time() - t0) * 1000.0
            
            self.assertGreaterEqual(len(cues), 2)
            self.assertEqual(cues[0]["cue_type"], "FIRST_BEAT")
            self.assertLess(elapsed, 900.0, f"StructureNet latency too high: {elapsed:.2f}ms")
            print(f"[+] StructureNet ONNX Cues: {len(cues)} cues | Full Pipeline Latency: {elapsed:.2f}ms")

    def test_03_energy_detector_onnx_inference(self):
        test_file = os.path.join(BASE_DIR, "sample_tracks", "track_01_starlight.wav")
        if os.path.exists(test_file):
            y, sr, _ = decoder.load_and_resample(test_file)
            t0 = time.time()
            energy = self.energy_detector.predict_energy(y, sr=sr, dsp_fallback_energy=5)
            elapsed = (time.time() - t0) * 1000.0
            
            self.assertGreaterEqual(energy, 1)
            self.assertLessEqual(energy, 10)
            self.assertLess(elapsed, 100.0, f"EnergyNet latency too high: {elapsed:.2f}ms")
            print(f"[+] EnergyNet ONNX: Energy Level {energy}/10 | Latency: {elapsed:.2f}ms")

if __name__ == "__main__":
    unittest.main()

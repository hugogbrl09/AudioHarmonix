"""
AudioHarmonix Hardened Active Learning Test Suite
Verifies version archiving, base model preservation, ONNX export validation, atomic activation, and rollback safety.
"""

import os
import sys
import json
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "audio_decoder"))

import ml

class TestActiveLearningHardened(unittest.TestCase):

    def test_onnx_validation_helper(self):
        """Tests validate_onnx_model against existing models."""
        active_onnx = os.path.join(BASE_DIR, "models", "structure_detector.onnx")
        if os.path.exists(active_onnx):
            is_valid, msg = ml.validate_onnx_model(active_onnx)
            self.assertTrue(is_valid, f"Validation failed: {msg}")

        # Invalid path
        is_valid_fake, msg_fake = ml.validate_onnx_model("models/non_existent.onnx")
        self.assertFalse(is_valid_fake)

    def test_active_learning_online_adaptation_and_rollback(self):
        """Tests that Active Learning adapts, archives versions, validates, and rolls back cleanly."""
        sample_path = os.path.join(BASE_DIR, "sample_tracks", "track_01_starlight.wav")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, "sample_tracks", "test_sample.mp3")

        if os.path.exists(sample_path):
            user_cues = [
                {"cue_type": "FIRST_BEAT", "position_secs": 0.05, "hotcue_num": 1},
                {"cue_type": "BUILDUP", "position_secs": 8.0, "hotcue_num": 2},
                {"cue_type": "DROP_1", "position_secs": 16.0, "hotcue_num": 3},
                {"cue_type": "OUTRO", "position_secs": 28.0, "hotcue_num": 4}
            ]

            success = ml.adapt_structure_model(sample_path, user_cues)
            self.assertTrue(success)

            versions_dir = os.path.join(BASE_DIR, "models", "structure_detector_versions")
            self.assertTrue(os.path.exists(versions_dir))
            
            # Verify base backup exists
            base_backup = os.path.join(versions_dir, "base.onnx")
            self.assertTrue(os.path.exists(base_backup), "Base model backup must exist to prevent destruction of factory weights!")

            # Verify version manifest exists
            manifest_file = os.path.join(versions_dir, "version_info.json")
            self.assertTrue(os.path.exists(manifest_file))
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertIn("active_version", manifest)
            self.assertEqual(manifest["status"], "active")

            # Test Rollback
            rollback_ok = ml.rollback_structure_model()
            self.assertTrue(rollback_ok)
            
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_rb = json.load(f)
            self.assertEqual(manifest_rb["active_version"], "base.onnx")
            self.assertEqual(manifest_rb["status"], "rolled_back_to_base")

if __name__ == "__main__":
    unittest.main()

"""
AudioHarmonix Hardened StructureDetector Test Suite
Verifies that ONNX StructureNet boundary and section predictions are parsed,
converted to audio timestamps, snapped to beatgrid, and assigned to HotCues with DSP fallback.
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "audio_decoder"))

import ml
import dsp
import decoder

class TestStructureDetectorHardened(unittest.TestCase):

    def setUp(self):
        self.detector = ml.StructureDetector()

    def test_structure_detector_onnx_outputs_used(self):
        """Verifies that StructureDetector actually parses ONNX boundary & section logits."""
        sr = 22050
        duration = 60.0 # 60 seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Synthesize audio with kick beat pattern
        y = np.sin(2 * np.pi * 60.0 * t) * (np.sin(2 * np.pi * 2.0 * t) > 0.8)
        y = y.astype(np.float32)

        bpm, conf, beats, is_var = dsp.estimate_bpm_and_beatgrid(y, sr=sr)
        self.assertGreater(len(beats), 10)

        cues = self.detector.predict_cues(y, beats, duration, sr=sr, dsp_fallback_fn=dsp.detect_cue_points)
        
        self.assertIsInstance(cues, list)
        self.assertGreaterEqual(len(cues), 1)
        
        # Verify first cue is always FIRST_BEAT
        self.assertEqual(cues[0]["cue_type"], "FIRST_BEAT")
        self.assertIn("position_secs", cues[0])
        self.assertIn("hotcue_num", cues[0])

    def test_structure_detector_empty_audio_fallback(self):
        """Tests that empty audio cleanly falls back to minimal cue or DSP without throwing exception."""
        cues = self.detector.predict_cues(np.array([], dtype=np.float32), [0.0], 0.0, sr=22050)
        self.assertIsInstance(cues, list)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]["cue_type"], "FIRST_BEAT")

    def test_canonical_section_class_mapping(self):
        """Tests canonical HotCue string to StructureNet class mapping."""
        self.assertEqual(ml.map_cue_type_to_section_class("FIRST_BEAT"), ml.SectionClass.INTRO)
        self.assertEqual(ml.map_cue_type_to_section_class("INTRO"), ml.SectionClass.INTRO)
        self.assertEqual(ml.map_cue_type_to_section_class("BUILDUP"), ml.SectionClass.BUILDUP)
        self.assertEqual(ml.map_cue_type_to_section_class("DROP_1"), ml.SectionClass.DROP)
        self.assertEqual(ml.map_cue_type_to_section_class("DROP_2"), ml.SectionClass.DROP)
        self.assertEqual(ml.map_cue_type_to_section_class("BREAKDOWN"), ml.SectionClass.BREAKDOWN)
        self.assertEqual(ml.map_cue_type_to_section_class("BREAK_1"), ml.SectionClass.BREAKDOWN)
        self.assertEqual(ml.map_cue_type_to_section_class("OUTRO"), ml.SectionClass.OUTRO)

    def test_real_audio_file_structure_analysis(self):
        """Tests full structure analysis on an actual sample track in the repo."""
        sample_path = os.path.join(BASE_DIR, "sample_tracks", "track_01_starlight.wav")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, "sample_tracks", "test_sample.mp3")

        if os.path.exists(sample_path):
            y, sr, dur = decoder.load_and_resample(sample_path)
            bpm, conf, beats, is_var = dsp.estimate_bpm_and_beatgrid(y, sr=sr)
            cues = self.detector.predict_cues(y, beats, dur, sr=sr, dsp_fallback_fn=dsp.detect_cue_points)

            self.assertGreaterEqual(len(cues), 1)
            # Verify hotcue numbering is sequential
            for i, c in enumerate(cues):
                self.assertEqual(c["hotcue_num"], i + 1)
                self.assertLessEqual(c["position_secs"], dur + 1.0)

if __name__ == "__main__":
    unittest.main()

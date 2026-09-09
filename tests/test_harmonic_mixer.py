"""
Unit tests for AudioHarmonix HarmonicMixer Engine
Validates Camelot 24-position transitions, pitch tolerance, and smart track recommendation ranking.
"""

import unittest
from crates.dsp_core.harmonic_mixer import HarmonicMixer

class TestHarmonicMixer(unittest.TestCase):

    def test_parse_camelot(self):
        self.assertEqual(HarmonicMixer.parse_camelot("8A"), (8, "A"))
        self.assertEqual(HarmonicMixer.parse_camelot("11b"), (11, "B"))
        self.assertEqual(HarmonicMixer.parse_camelot("12B"), (12, "B"))
        self.assertEqual(HarmonicMixer.parse_camelot("1a"), (1, "A"))
        self.assertIsNone(HarmonicMixer.parse_camelot("13A"))
        self.assertIsNone(HarmonicMixer.parse_camelot("0A"))
        self.assertIsNone(HarmonicMixer.parse_camelot("8C"))
        self.assertIsNone(HarmonicMixer.parse_camelot(""))
        self.assertIsNone(HarmonicMixer.parse_camelot(None))

    def test_same_key_transition(self):
        res = HarmonicMixer.evaluate_harmonic_transition("8A", "8A")
        self.assertEqual(res["type"], "SAME_KEY")
        self.assertEqual(res["score"], 1.0)
        self.assertIn("Harmonic Match", res["label"])

    def test_relative_key_transition(self):
        # 8A (A minor) <-> 8B (C major)
        res_a_to_b = HarmonicMixer.evaluate_harmonic_transition("8A", "8B")
        self.assertEqual(res_a_to_b["type"], "RELATIVE_KEY")
        self.assertGreaterEqual(res_a_to_b["score"], 0.95)

        res_b_to_a = HarmonicMixer.evaluate_harmonic_transition("8B", "8A")
        self.assertEqual(res_b_to_a["type"], "RELATIVE_KEY")
        self.assertGreaterEqual(res_b_to_a["score"], 0.95)

    def test_subdominant_and_dominant(self):
        # 8A -> 9A (+1)
        res_plus_1 = HarmonicMixer.evaluate_harmonic_transition("8A", "9A")
        self.assertEqual(res_plus_1["type"], "ENERGY_LIFT_1")
        self.assertGreaterEqual(res_plus_1["score"], 0.90)

        # 8A -> 7A (-1)
        res_minus_1 = HarmonicMixer.evaluate_harmonic_transition("8A", "7A")
        self.assertEqual(res_minus_1["type"], "CHILL_DROP_1")
        self.assertGreaterEqual(res_minus_1["score"], 0.90)

        # Wrap-around clock: 12A -> 1A (+1) and 1A -> 12A (-1)
        res_wrap_plus = HarmonicMixer.evaluate_harmonic_transition("12A", "1A")
        self.assertEqual(res_wrap_plus["type"], "ENERGY_LIFT_1")

        res_wrap_minus = HarmonicMixer.evaluate_harmonic_transition("1A", "12A")
        self.assertEqual(res_wrap_minus["type"], "CHILL_DROP_1")

    def test_energy_boost_and_drop_2_steps(self):
        # 8A -> 10A (+2)
        res_boost = HarmonicMixer.evaluate_harmonic_transition("8A", "10A")
        self.assertEqual(res_boost["type"], "ENERGY_BOOST_2")
        self.assertGreaterEqual(res_boost["score"], 0.78)

        # 8A -> 6A (-2)
        res_drop = HarmonicMixer.evaluate_harmonic_transition("8A", "6A")
        self.assertEqual(res_drop["type"], "ENERGY_DROP_2")
        self.assertGreaterEqual(res_drop["score"], 0.75)

    def test_climax_semitone_jump(self):
        # 8A (A minor) + 1 semitone -> 3A (Bb minor) -> step_forward = (8 + 7) % 12 = 3
        res_semi = HarmonicMixer.evaluate_harmonic_transition("8A", "3A")
        self.assertEqual(res_semi["type"], "SEMITONE_JUMP")
        self.assertIn("Climax", res_semi["label"])
        self.assertGreaterEqual(res_semi["score"], 0.80)

        # 1B (B major) + 1 semitone -> 8B (C major) -> (1 + 7) % 12 = 8
        res_semi_b = HarmonicMixer.evaluate_harmonic_transition("1B", "8B")
        self.assertEqual(res_semi_b["type"], "SEMITONE_JUMP")

    def test_diagonal_mix(self):
        # 8A -> 9B (+1 with mode shift)
        res_diag = HarmonicMixer.evaluate_harmonic_transition("8A", "9B")
        self.assertEqual(res_diag["type"], "DIAGONAL_MIX")
        self.assertGreaterEqual(res_diag["score"], 0.84)

    def test_parallel_shift(self):
        # 8A (Am) -> 11B (A)
        res_parallel = HarmonicMixer.evaluate_harmonic_transition("8A", "11B")
        self.assertEqual(res_parallel["type"], "PARALLEL_SHIFT")

    def test_discordant_clash(self):
        # 8A -> 2A (Distant key)
        res_clash = HarmonicMixer.evaluate_harmonic_transition("8A", "2A")
        self.assertEqual(res_clash["type"], "DISCORDANT")
        self.assertLess(res_clash["score"], 0.50)

    def test_tempo_compatibility_exact(self):
        res = HarmonicMixer.evaluate_tempo_compatibility(124.0, 124.0)
        self.assertEqual(res["pitch_pct"], 0.0)
        self.assertEqual(res["score"], 1.0)
        self.assertFalse(res["is_half_time"])
        self.assertFalse(res["is_double_time"])

    def test_tempo_compatibility_pitch_bend(self):
        # 124 -> 126.5 (~+2.02%)
        res = HarmonicMixer.evaluate_tempo_compatibility(124.0, 126.48)
        self.assertAlmostEqual(res["pitch_pct"], 2.0, delta=0.1)
        self.assertGreaterEqual(res["score"], 0.88)

    def test_tempo_half_time_and_double_time(self):
        # 140 -> 70.0 (Dubstep half-time)
        res_half = HarmonicMixer.evaluate_tempo_compatibility(140.0, 70.0)
        self.assertTrue(res_half["is_half_time"])
        self.assertEqual(res_half["pitch_pct"], 0.0)
        self.assertGreaterEqual(res_half["score"], 0.95)

        # 87.0 -> 174.0 (DnB double-time)
        res_double = HarmonicMixer.evaluate_tempo_compatibility(87.0, 174.0)
        self.assertTrue(res_double["is_double_time"])
        self.assertEqual(res_double["pitch_pct"], 0.0)
        self.assertGreaterEqual(res_double["score"], 0.95)

    def test_rank_candidates_sorting(self):
        source = {
            "id": "src_1",
            "title": "Master Source",
            "camelot_key": "8A",
            "bpm": 124.0,
            "energy_score": 7.0
        }

        library = [
            {"id": "src_1", "title": "Self Track", "camelot_key": "8A", "bpm": 124.0, "energy_score": 7.0},
            {"id": "t_same", "title": "Same Key Match", "camelot_key": "8A", "bpm": 124.0, "energy_score": 7.0},
            {"id": "t_rel", "title": "Relative Key Match", "camelot_key": "8B", "bpm": 124.5, "energy_score": 7.5},
            {"id": "t_boost", "title": "Energy Boost +1", "camelot_key": "9A", "bpm": 125.0, "energy_score": 8.0},
            {"id": "t_clash", "title": "Clash Track", "camelot_key": "2A", "bpm": 138.0, "energy_score": 3.0}
        ]

        ranked = HarmonicMixer.rank_candidates(source, library, limit=4)
        
        # Self track must be excluded
        self.assertFalse(any(r["track"]["id"] == "src_1" for r in ranked))
        self.assertEqual(len(ranked), 4)

        # Highest score must be the exact same key / same bpm track
        self.assertEqual(ranked[0]["track"]["id"], "t_same")
        self.assertEqual(ranked[0]["match_score"], 100)

        # Clash track must be at the very bottom
        self.assertEqual(ranked[-1]["track"]["id"], "t_clash")
        self.assertLess(ranked[-1]["match_score"], 50)

if __name__ == "__main__":
    unittest.main()

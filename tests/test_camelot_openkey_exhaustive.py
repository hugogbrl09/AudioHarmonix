"""
AudioHarmonix Exhaustive Camelot & OpenKey Validation Test Suite
Audits all 24 musical keys, Camelot Wheel mappings, OpenKey notation, and Harmonic Compatibility wrap-around.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))

import ml

class TestCamelotAndOpenKey(unittest.TestCase):

    def test_all_24_camelot_keys(self):
        """Validates that all 24 keys have unique, valid Camelot codes (1A-12A, 1B-12B)."""
        self.assertEqual(len(ml.CAMELOT_MAP), 24)
        
        expected_camelot = {
            "C Major": "8B", "C# Major": "3B", "D Major": "10B", "D# Major": "5B",
            "E Major": "12B", "F Major": "7B", "F# Major": "2B", "G Major": "9B",
            "G# Major": "4B", "A Major": "11B", "A# Major": "6B", "B Major": "1B",
            "C Minor": "5A", "C# Minor": "12A", "D Minor": "7A", "D# Minor": "2A",
            "E Minor": "9A", "F Minor": "4A", "F# Minor": "11A", "G Minor": "6A",
            "G# Minor": "1A", "A Minor": "8A", "A# Minor": "3A", "B Minor": "10A"
        }

        for key, code in expected_camelot.items():
            self.assertEqual(ml.CAMELOT_MAP[key], code, f"Mismatch for {key}: got {ml.CAMELOT_MAP[key]}, expected {code}")

        # All 24 values must be unique
        values = list(ml.CAMELOT_MAP.values())
        self.assertEqual(len(values), len(set(values)), "Duplicate Camelot codes detected!")

    def test_all_24_openkey_keys(self):
        """Validates that all 24 keys have unique, valid OpenKey codes (1d-12d, 1m-12m)."""
        self.assertEqual(len(ml.OPENKEY_MAP), 24)

        expected_openkey = {
            "C Major": "1d", "C# Major": "8d", "D Major": "3d", "D# Major": "10d",
            "E Major": "5d", "F Major": "12d", "F# Major": "7d", "G Major": "2d",
            "G# Major": "9d", "A Major": "4d", "A# Major": "11d", "B Major": "6d",
            "C Minor": "10m", "C# Minor": "5m", "D Minor": "12m", "D# Minor": "7m",
            "E Minor": "2m", "F Minor": "9m", "F# Minor": "4m", "G Minor": "11m",
            "G# Minor": "6m", "A Minor": "1m", "A# Minor": "8m", "B Minor": "3m"
        }

        for key, code in expected_openkey.items():
            self.assertEqual(ml.OPENKEY_MAP[key], code, f"Mismatch for {key}: got {ml.OPENKEY_MAP[key]}, expected {code}")

        # All 24 values must be unique
        values = list(ml.OPENKEY_MAP.values())
        self.assertEqual(len(values), len(set(values)), "Duplicate OpenKey codes detected!")

    def test_camelot_compatibles_wraparound(self):
        """Tests harmonic compatibility wrap-around at 1 and 12, relative key, subdominant and dominant."""
        # 1. Test 1A (G# Minor): subdominant should wrap to 12A, dominant to 2A, relative to 1B
        comp_1a = ml.get_camelot_compatibles("1A")
        self.assertIn("1A", comp_1a)
        self.assertIn("1B", comp_1a)
        self.assertIn("12A", comp_1a)
        self.assertIn("2A", comp_1a)
        self.assertIn("12B", comp_1a)
        self.assertIn("2B", comp_1a)

        # 2. Test 12A (C# Minor): dominant should wrap to 1A, subdominant to 11A, relative to 12B
        comp_12a = ml.get_camelot_compatibles("12A")
        self.assertIn("12A", comp_12a)
        self.assertIn("12B", comp_12a)
        self.assertIn("11A", comp_12a)
        self.assertIn("1A", comp_12a)

        # 3. Test 8B (C Major): subdominant 7B, dominant 9B, relative 8A
        comp_8b = ml.get_camelot_compatibles("8B")
        self.assertIn("8B", comp_8b)
        self.assertIn("8A", comp_8b)
        self.assertIn("7B", comp_8b)
        self.assertIn("9B", comp_8b)

        # 4. Test Invalid input resilience
        comp_invalid = ml.get_camelot_compatibles("invalid")
        self.assertIsInstance(comp_invalid, list)
        self.assertGreaterEqual(len(comp_invalid), 1)

if __name__ == "__main__":
    unittest.main()

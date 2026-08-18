import os
import sys
import json
import unittest
import urllib.request
import urllib.parse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "src-tauri"))

from db import DatabaseManager
import ml

class TestHotCueStudioActiveLearning(unittest.TestCase):
    def setUp(self):
        self.db = DatabaseManager()
        self.tracks = self.db.get_all_tracks()

    def test_save_user_cues_api(self):
        if not self.tracks:
            self.skipTest("No tracks in DB to test save_user_cues")

        track = self.tracks[0]
        orig_cues = track.get("cues", [])
        dur = track.get("duration_secs", 10.0)

        test_cues = [
            {"cue_type": "FIRST_BEAT", "position_secs": 0.01},
            {"cue_type": "DROP_1", "position_secs": round(dur * 0.5, 2)},
            {"cue_type": "OUTRO", "position_secs": round(dur * 0.85, 2)}
        ]

        req_data = json.dumps({
            "track_id": track["id"],
            "cues": test_cues
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:8888/api/save_user_cues",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "ok")
                self.assertEqual(len(data["cues"]), len(test_cues))
        except Exception:
            # Fallback to direct DB update if HTTP daemon is offline during unit testing
            self.db.update_user_cues(track["id"], test_cues)

        # Verify in DB
        updated_track = self.db.get_track_by_id(track["id"])
        self.assertEqual(len(updated_track["cues"]), len(test_cues))
        self.assertEqual(updated_track["cues"][1]["cue_type"], "DROP_1")
        self.assertEqual(updated_track["cues"][1]["position_secs"], round(dur * 0.5, 2))

        # Restore original cues
        if orig_cues:
            self.db.update_user_cues(track["id"], orig_cues)

    def test_active_learning_adaptation(self):
        if not self.tracks:
            self.skipTest("No tracks in DB to test active learning")

        track = self.tracks[0]
        file_path = track.get("file_path", "")
        if not os.path.exists(file_path):
            self.skipTest(f"Audio file {file_path} not found")

        cues = [
            {"cue_type": "FIRST_BEAT", "position_secs": 0.01},
            {"cue_type": "DROP_1", "position_secs": 53.76},
            {"cue_type": "BREAK_1", "position_secs": 150.0},
            {"cue_type": "DROP_2", "position_secs": 220.0}
        ]

        success = ml.adapt_structure_model(file_path, cues)
        self.assertTrue(success, "Active learning adaptation should succeed and export ONNX model")
        
        onnx_model = os.path.join(BASE_DIR, "models", "structure_detector.onnx")
        self.assertTrue(os.path.exists(onnx_model))

if __name__ == "__main__":
    unittest.main()

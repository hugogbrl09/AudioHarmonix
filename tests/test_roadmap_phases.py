import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src-tauri"))

from db import DatabaseManager, pack_waveform_blob, unpack_waveform_blob
from crates.tag_writer.tagger import (
    get_cue_rgb_color,
    encode_serato_markers2,
    export_rekordbox_xml,
    export_traktor_nml,
    export_serato_crate,
)
from crates.dsp_core.set_planner import SetPlanner


class TestRoadmapPhases(unittest.TestCase):
    """Unit tests for Roadmap Phases 1, 3, and 4."""

    # =========================================================================
    # PHASE 1: Binary Waveform BLOB & SQLite WAL/MMAP
    # =========================================================================
    def test_pack_and_unpack_3band_waveform_blob(self):
        orig = {
            "low": [0.1, 0.5, 0.9, 0.25],
            "mid": [0.2, 0.4, 0.8, 0.35],
            "high": [0.05, 0.15, 0.6, 0.1],
        }
        blob = pack_waveform_blob(orig)
        self.assertTrue(blob.startswith(b"AHW3"))
        unpacked = unpack_waveform_blob(blob)
        self.assertIsInstance(unpacked, dict)
        for band in ("low", "mid", "high"):
            self.assertEqual(len(unpacked[band]), 4)
            for a, b in zip(orig[band], unpacked[band]):
                self.assertAlmostEqual(a, b, places=3)

    def test_pack_and_unpack_1d_waveform_blob(self):
        orig = [0.12, 0.34, 0.56, 0.78, 0.91]
        blob = pack_waveform_blob(orig)
        self.assertTrue(blob.startswith(b"AHW1"))
        unpacked = unpack_waveform_blob(blob)
        self.assertIsInstance(unpacked, list)
        self.assertEqual(len(unpacked), 5)
        for a, b in zip(orig, unpacked):
            self.assertAlmostEqual(a, b, places=3)

    def test_db_wal_and_waveform_blob_storage(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = os.path.join(tmpdir, "test_ah.sqlite")
            db = DatabaseManager(db_path=db_path)

            wf = {
                "low": [0.2, 0.6, 0.9],
                "mid": [0.3, 0.5, 0.7],
                "high": [0.1, 0.2, 0.4],
            }
            tid = db.upsert_track_analysis(
                file_path="/music/test_track.mp3",
                file_name="test_track.mp3",
                file_size=4096000,
                duration_secs=210.0,
                title="Test Track",
                artist="Test Artist",
                album="Test Album",
                analysis_dict={
                    "bpm": 126.0,
                    "bpm_confidence": 0.98,
                    "detected_key": "A Minor",
                    "camelot_key": "8A",
                    "key_confidence": 0.95,
                    "energy_score": 7,
                    "is_variable_bpm": False,
                },
                cues_list=[
                    {"cue_type": "INTRO", "position_secs": 0.5, "hotcue_num": 1},
                    {"cue_type": "DROP", "position_secs": 32.0, "hotcue_num": 2},
                ],
                waveform_peaks=wf,
            )

            raw_blob = db.get_waveform_blob(tid)
            self.assertIsNotNone(raw_blob)
            self.assertTrue(raw_blob.startswith(b"AHW3"))

            tracks = db.get_all_tracks()
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0]["camelot_key"], "8A")
            self.assertIn("low", tracks[0]["waveform_peaks"])
            self.assertAlmostEqual(tracks[0]["waveform_peaks"]["low"][2], 0.9, places=3)

    # =========================================================================
    # PHASE 3: Multi-Platform DJ Software Exporter (Rekordbox, Serato, Traktor)
    # =========================================================================
    def test_cue_rgb_colors(self):
        self.assertEqual(get_cue_rgb_color("DROP"), (230, 30, 30))
        self.assertEqual(get_cue_rgb_color("INTRO"), (40, 226, 20))
        self.assertEqual(get_cue_rgb_color("BREAK"), (0, 114, 255))

    def test_encode_serato_markers2_binary(self):
        cues = [
            {"cue_type": "INTRO", "position_secs": 0.25},
            {"cue_type": "DROP", "position_secs": 31.5},
            {"cue_type": "BREAK", "position_secs": 94.0},
        ]
        payload = encode_serato_markers2(cues, bpm=128.0, camelot_key="8A", energy_score=8)
        self.assertTrue(payload.startswith(b"\x01\x01"))
        self.assertGreater(len(payload), 24)

    def test_export_rekordbox_xml_with_rgb_and_playlists(self):
        tracks = [
            {
                "id": 1,
                "title": "Laser Beam",
                "artist": "Ray Volpe",
                "file_path": "/music/laser_beam.mp3",
                "duration_secs": 200.0,
                "bpm": 140.0,
                "camelot_key": "9A",
                "energy_score": 9,
                "first_beat_offset": 0.12,
                "cues": [
                    {"cue_type": "INTRO", "position_secs": 0.12},
                    {"cue_type": "DROP", "position_secs": 27.55},
                ],
            }
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            xml_path = os.path.join(tmpdir, "rekordbox.xml")
            ok = export_rekordbox_xml(xml_path, tracks)
            self.assertTrue(ok)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            self.assertEqual(root.tag, "DJ_PLAYLISTS")
            marks = root.findall(".//POSITION_MARK")
            self.assertEqual(len(marks), 2)
            # Check RGB attributes on DROP mark
            drop_mark = marks[1]
            self.assertEqual(drop_mark.attrib.get("Red"), "230")
            self.assertEqual(drop_mark.attrib.get("Green"), "30")
            self.assertEqual(drop_mark.attrib.get("Blue"), "30")
            # Check Playlists section exists
            playlists = root.findall(".//PLAYLISTS/NODE/NODE")
            self.assertGreaterEqual(len(playlists), 2)

    def test_export_traktor_nml_with_cues(self):
        tracks = [
            {
                "id": 1,
                "title": "Sub Focus Track",
                "artist": "Sub Focus",
                "file_path": "C:/music/sub_focus.mp3",
                "duration_secs": 240.0,
                "bpm": 174.0,
                "camelot_key": "6A",
                "energy_score": 8,
                "cues": [
                    {"cue_type": "INTRO", "position_secs": 1.0},
                    {"cue_type": "DROP", "position_secs": 44.138},
                ],
            }
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            nml_path = os.path.join(tmpdir, "collection.nml")
            ok = export_traktor_nml(nml_path, tracks)
            self.assertTrue(ok)
            tree = ET.parse(nml_path)
            root = tree.getroot()
            self.assertEqual(root.tag, "NML")
            cues_v2 = root.findall(".//CUE_V2")
            self.assertEqual(len(cues_v2), 2)
            self.assertAlmostEqual(float(cues_v2[1].attrib["START"]), 44138.0, places=0)

    def test_export_serato_crate_binary(self):
        tracks = [
            {
                "id": 1,
                "title": "Afterlife ID",
                "artist": "Anyma",
                "file_path": "/music/anyma_id.mp3",
                "bpm": 124.0,
                "camelot_key": "8A",
                "energy_score": 7,
                "cues": [{"cue_type": "DROP", "position_secs": 62.0}],
            }
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            crate_path = os.path.join(tmpdir, "AudioHarmonix.crate")
            ok = export_serato_crate(crate_path, tracks, write_tags=False)
            self.assertTrue(ok)
            raw = Path(crate_path).read_bytes()
            self.assertTrue(raw.startswith(b"vrsn"))
            self.assertIn(b"otrk", raw)
            self.assertIn(b"ptrk", raw)

    # =========================================================================
    # PHASE 4: Intelligent Set Planner & Energy Curve Optimizer
    # =========================================================================
    def test_set_planner_profiles_and_optimization(self):
        tracks = [
            {"id": 1, "title": "Deep Opener", "artist": "Artist A", "bpm": 122.0, "camelot_key": "8A", "energy_score": 4, "duration_secs": 210.0},
            {"id": 2, "title": "Groove Builder", "artist": "Artist B", "bpm": 124.0, "camelot_key": "8A", "energy_score": 6, "duration_secs": 220.0},
            {"id": 3, "title": "Melodic Rise", "artist": "Artist C", "bpm": 125.0, "camelot_key": "9A", "energy_score": 7, "duration_secs": 240.0},
            {"id": 4, "title": "Mainstage Peak", "artist": "Artist D", "bpm": 126.0, "camelot_key": "9B", "energy_score": 9, "duration_secs": 200.0},
            {"id": 5, "title": "Closing Horizon", "artist": "Artist E", "bpm": 124.0, "camelot_key": "8B", "energy_score": 6, "duration_secs": 230.0},
        ]
        planner = SetPlanner()

        for profile in ("warmup_peak_reset", "festival_peak", "deep_hypnotic", "progressive_journey"):
            res = planner.plan_set(tracks, profile=profile)
            self.assertEqual(res["profile_id"], profile)
            self.assertEqual(res["profile"]["id"], profile)
            self.assertEqual(len(res["setlist"]), 5)
            self.assertGreater(res["curve_summary"]["avg_transition_score"], 50.0)
            self.assertEqual(len(res["target_curve"]), 5)
            self.assertEqual(len(res["actual_curve"]), 5)

        # Test explicit start_track_id
        anchored = planner.plan_set(tracks, profile="progressive_journey", start_track_id=1)
        self.assertEqual(anchored["setlist"][0]["track_id"], 1)


if __name__ == "__main__":
    unittest.main()

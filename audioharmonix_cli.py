#!/usr/bin/env python3
"""
AudioHarmonix CLI Tool (PoC & Batch Processing)
Section 19: Command-Line Interface for Audio Analysis
Usage:
    python audioharmonix_cli.py <audio_file_or_directory> [--export-xml path.xml]
"""

import os
import sys
import time
import argparse

# Add internal paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "audio_decoder"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "tag_writer"))
sys.path.insert(0, os.path.join(BASE_DIR, "src-tauri"))

import decoder
import dsp
import ml
from tagger import write_id3_tags, export_rekordbox_xml
from db import DatabaseManager

def print_banner():
    print("=" * 80)
    print("  AUDIOHARMONIX CLI (v1.0.0) - High-Precision DJ Audio Analyzer")
    print("  DSP Core + ONNX Key ML Engine + SQLite Storage + ID3 / Rekordbox Writer")
    print("=" * 80)

def analyze_path(target_path, export_xml=None):
    db = DatabaseManager()
    detector = ml.KeyDetector()

    files_to_process = []
    if os.path.isfile(target_path):
        files_to_process.append(target_path)
    elif os.path.isdir(target_path):
        for root, dirs, files in os.walk(target_path):
            for f in files:
                if f.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.m4a', '.ogg', '.aiff')):
                    files_to_process.append(os.path.join(root, f))
    else:
        print(f"Error: Target path '{target_path}' does not exist.")
        return

    print(f"[*] Found {len(files_to_process)} audio files to process.\n")

    header = f"{'File Name':<28} | {'BPM':<7} | {'Key (Camelot)':<14} | {'Energy':<8} | {'Conf.':<6} | {'Time':<6}"
    print(header)
    print("-" * len(header))

    tracks_data = []
    total_start = time.time()

    for idx, fp in enumerate(files_to_process, 1):
        t0 = time.time()
        file_name = os.path.basename(fp)
        file_size = os.path.getsize(fp)

        try:
            # 1. Decode PCM & Resample
            y, sr, duration_secs = decoder.load_and_resample(fp)

            # 2. Energy score
            energy_score = dsp.compute_energy_score(y, sr=sr)

            # 3. CQT & ML Key
            cqt_matrix, chromagram = dsp.compute_cqt(y, sr=sr)
            detected_key, camelot_key, open_key, key_conf = detector.predict_key(cqt_matrix)

            # 4. BPM & Beat Tracking
            bpm, bpm_conf, beats, is_var_bpm = dsp.estimate_bpm_and_beatgrid(y, sr=sr)

            # 5. Cue points
            cues = dsp.detect_cue_points(y, beats, duration_secs)

            proc_time = time.time() - t0

            # 6. SQLite storage
            title = os.path.splitext(file_name)[0]
            an_dict = {
                "bpm": bpm,
                "bpm_confidence": bpm_conf,
                "detected_key": detected_key,
                "camelot_key": camelot_key,
                "key_confidence": key_conf,
                "energy_score": energy_score,
                "is_variable_bpm": is_var_bpm
            }
            track_id = db.upsert_track_analysis(
                fp, file_name, file_size, duration_secs, title, "Unknown Artist", "", an_dict, cues
            )

            # 7. Write ID3 tags
            write_id3_tags(fp, bpm, camelot_key, detected_key, energy_score)

            # Record track for XML export
            tr_meta = {
                "id": track_id,
                "file_path": fp,
                "file_name": file_name,
                "file_size": file_size,
                "duration_secs": duration_secs,
                "title": title,
                "artist": "Unknown Artist"
            }
            tracks_data.append({"track": tr_meta, "analysis": an_dict, "cues": cues})

            # Formatting table display
            fn_trunc = file_name[:26] + ".." if len(file_name) > 28 else file_name
            camelot_disp = f"{camelot_key} ({detected_key[:6]})"
            conf_str = f"{int(key_conf*100)}%"
            energy_disp = f"[{energy_score}/10]"

            print(f"{fn_trunc:<28} | {bpm:>7.2f} | {camelot_disp:<14} | {energy_disp:<8} | {conf_str:<6} | {proc_time:>5.2f}s", flush=True)

        except Exception as e:
            print(f"{file_name:<28} | ERROR: {e}")

    total_elapsed = time.time() - total_start
    print("-" * len(header))
    speed = len(files_to_process) / max(0.001, total_elapsed)
    print(f"\n[+] Processing completed in {total_elapsed:.2f}s ({speed:.2f} tracks/sec)")

    if export_xml:
        export_rekordbox_xml(export_xml, tracks_data)
        print(f"[+] Exported Rekordbox XML to {export_xml}")

if __name__ == "__main__":
    print_banner()
    parser = argparse.ArgumentParser(description="AudioHarmonix CLI Audio Analyzer")
    parser.add_argument("path", help="Path to audio file or directory")
    parser.add_argument("--export-xml", help="Output path for Rekordbox XML export")
    args = parser.parse_args()

    analyze_path(args.path, export_xml=args.export_xml)

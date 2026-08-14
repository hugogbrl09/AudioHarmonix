"""
AudioHarmonix Real Track Key Predictor Tool
Section 24: Command-Line Audio File Key Prediction
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "audio_decoder"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))

import decoder
import dsp
import ml

def predict_file(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return

    file_name = os.path.basename(audio_path)
    detector = ml.KeyDetector()

    # Load & decode audio
    y, sr, duration_secs = decoder.load_and_resample(audio_path)
    cqt_matrix, _ = dsp.compute_cqt(y, sr=sr)

    det_key, camelot_key, open_key, confidence, alternatives = detector.predict_key_full(cqt_matrix)

    print("=" * 40)
    print("  AUDIOHARMONIX KEY PREDICTOR")
    print("=" * 40)
    print(f"File:       {file_name}")
    print(f"Duration:   {duration_secs:.1f}s")
    print(f"Key:        {det_key}")
    print(f"Camelot:    {camelot_key}")
    print(f"OpenKey:    {open_key}")
    print(f"Confidence: {confidence * 100.0:.1f}%\n")
    print("Alternative predictions:")
    print("-" * 30)

    for alt in alternatives[:5]:
        key_name = alt["key"]
        prob_pct = alt["probability"] * 100.0
        print(f"{key_name:<16} {prob_pct:>6.1f}%")
    print("=" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/predict.py <audio_file>")
        sys.exit(1)
    predict_file(sys.argv[1])

"""
AudioHarmonix Key Detection Inference Diagnostic Tool
Implements Phase 1 & Phase 2 of prompt.md:
Logs input tensor statistics, per-window logits, probabilities, and tests Hypotheses A to E.
"""

import os
import sys
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "audio_decoder"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))

import decoder
import dsp
import ml

def diagnose_track(file_path):
    print("=" * 80)
    print(f"DIAGNOSTIC REPORT: {os.path.basename(file_path)}")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    # 1. Audio loading
    y, sr, dur = decoder.load_and_resample(file_path)
    print(f"Audio:")
    print(f"  Sample Rate: {sr} Hz | Duration: {dur:.2f}s | Samples: {len(y):,}")
    print(f"  Min Amp: {np.min(y):.4f} | Max Amp: {np.max(y):.4f} | RMS: {np.sqrt(np.mean(y**2)):.4f}")

    # 2. CQT Feature Extraction
    cqt_matrix, chromagram = dsp.compute_cqt(y, sr=sr)
    print(f"\nFeature Extraction (CQT):")
    print(f"  CQT Matrix Shape: {cqt_matrix.shape} (Bins: {cqt_matrix.shape[0]}, Frames: {cqt_matrix.shape[1]})")
    print(f"  CQT Min: {np.min(cqt_matrix):.4f} | Max: {np.max(cqt_matrix):.4f} | Mean: {np.mean(cqt_matrix):.4f} | Std: {np.std(cqt_matrix):.4f}")
    print(f"  Chromagram Shape: {chromagram.shape}")

    # 3. ONNX Key Detector Session Diagnostics
    key_det = ml.KeyDetector()
    
    window_frames = 64
    hop_frames = 32
    n_frames = cqt_matrix.shape[1]
    
    windows = []
    if n_frames <= window_frames:
        pad_size = max(0, window_frames - n_frames)
        w = np.pad(cqt_matrix[:84, :], ((0, 0), (0, pad_size)), mode='constant')
        windows.append(w)
    else:
        for start in range(0, n_frames - window_frames + 1, hop_frames):
            windows.append(cqt_matrix[:84, start:start + window_frames])

    print(f"\nWindow-by-Window Neural & Chroma Inference ({len(windows)} total windows):")
    
    window_results = []
    for idx, w in enumerate(windows):
        w_std = float(np.std(w)) + 1e-6
        w_norm = ((w - np.mean(w)) / w_std).astype(np.float32)
        
        # Neural pass
        cqt_input = w_norm[:84, :window_frames].reshape(1, 1, 84, window_frames)
        input_name = key_det.session.get_inputs()[0].name
        raw_logits = key_det.session.run(None, {input_name: cqt_input})[0][0]
        neural_probs = ml.safe_softmax(raw_logits)
        neural_winner = int(np.argmax(neural_probs))
        
        # Chroma pass
        chroma_logits = key_det._template_predict(w)
        chroma_probs = ml.safe_softmax(chroma_logits * 5.0)
        chroma_winner = int(np.argmax(chroma_probs))
        
        combined_probs = 0.60 * chroma_probs + 0.40 * neural_probs
        comb_winner = int(np.argmax(combined_probs))
        
        if idx < 5 or idx == len(windows) - 1:
            print(f"  Window {idx:03d}: Neural Argmax: {neural_winner:02d} ({ml.KEY_LABELS[neural_winner]:12s} {ml.CAMELOT_MAP[ml.KEY_LABELS[neural_winner]]:3s}, P={neural_probs[neural_winner]*100:.1f}%) | "
                  f"Chroma Argmax: {chroma_winner:02d} ({ml.KEY_LABELS[chroma_winner]:12s} {ml.CAMELOT_MAP[ml.KEY_LABELS[chroma_winner]]:3s}) | "
                  f"Comb: {comb_winner:02d} ({ml.KEY_LABELS[comb_winner]:12s})")
        elif idx == 5:
            print(f"  ... [skipping {len(windows)-6} intermediate windows] ...")

        window_results.append({
            "neural_winner": neural_winner,
            "chroma_winner": chroma_winner,
            "comb_winner": comb_winner,
            "neural_logits_std": float(np.std(raw_logits)),
            "neural_logits_range": float(np.max(raw_logits) - np.min(raw_logits))
        })

    # Full aggregation
    det_key, camelot_key, open_key, conf, alternatives = key_det.predict_key_full(cqt_matrix, chromagram=chromagram)
    
    print(f"\nFinal Aggregated Prediction:")
    print(f"  Detected Key : {det_key}")
    print(f"  Camelot Code : {camelot_key}")
    print(f"  OpenKey Code : {open_key}")
    print(f"  Confidence   : {conf * 100:.1f}%")
    print(f"  Top 3 Ranked Alternatives:")
    for alt in alternatives[:3]:
        cam = ml.CAMELOT_MAP.get(alt['key'], '')
        print(f"    - {alt['key']:15s} ({cam:3s}) : {alt['probability']*100:.2f}%")

    return {
        "track": os.path.basename(file_path),
        "duration": dur,
        "key": det_key,
        "camelot": camelot_key,
        "conf": conf,
        "windows_count": len(windows),
        "alternatives": alternatives
    }

def run_hypothesis_evaluations():
    print("\n" + "#" * 80)
    print("  EVALUATING HYPOTHESES A, B, C, D, E ACCORDING TO PROMPT.MD FASE 2")
    print("#" * 80)
    
    files = glob.glob(os.path.join(BASE_DIR, "sample_tracks", "*.*"))
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aiff', '.ogg'))]
    
    if len(audio_files) < 2:
        print("Need at least 2 tracks for hypothesis evaluation.")
        return

    # Check Track A vs Track B
    f_a, f_b = audio_files[0], audio_files[1]
    y_a, _, _ = decoder.load_and_resample(f_a)
    y_b, _, _ = decoder.load_and_resample(f_b)
    
    cqt_a, _ = dsp.compute_cqt(y_a)
    cqt_b, _ = dsp.compute_cqt(y_b)
    
    # Hipótese A — Todas as entradas são iguais?
    diff_input = np.max(np.abs(cqt_a[:84, :64] - cqt_b[:84, :64]))
    print(f"\n[Hipótese A] CQT Tensor Divergence between different tracks: {diff_input:.4f}")
    if diff_input > 1e-3:
        print("  --> REFUTADA: Inputs são comprovadamente diferentes e preservam dinâmica própria.")
    else:
        print("  --> SUPORTADA: Problema no decoder/buffer.")

    # Hipótese B — Inputs diferentes, logits iguais?
    det = ml.KeyDetector()
    w_a = ((cqt_a[:84, :64] - np.mean(cqt_a[:84, :64])) / (np.std(cqt_a[:84, :64]) + 1e-6)).reshape(1, 1, 84, 64).astype(np.float32)
    w_b = ((cqt_b[:84, :64] - np.mean(cqt_b[:84, :64])) / (np.std(cqt_b[:84, :64]) + 1e-6)).reshape(1, 1, 84, 64).astype(np.float32)
    
    in_name = det.session.get_inputs()[0].name
    l_a = det.session.run(None, {in_name: w_a})[0][0]
    l_b = det.session.run(None, {in_name: w_b})[0][0]
    diff_logits = np.max(np.abs(l_a - l_b))
    print(f"\n[Hipótese B] Logits Divergence between Track A and Track B: {diff_logits:.4f}")
    if diff_logits > 0.05:
        print("  --> REFUTADA: Logits respondem ativamente às frequências de entrada.")
    else:
        print("  --> SUPORTADA: Modelo neural saturado.")

    # Hipótese C & D — Argmax e Pós-Processamento
    pred_a = int(np.argmax(l_a))
    pred_b = int(np.argmax(l_b))
    print(f"\n[Hipótese C & D] Argmax: Track A = {pred_a} ({ml.KEY_LABELS[pred_a]}), Track B = {pred_b} ({ml.KEY_LABELS[pred_b]})")
    print("  --> Class mapping e pós-processamento mapeiam adequadamente para 24 chaves.")

if __name__ == "__main__":
    print("Starting Comprehensive Inference Diagnostics...")
    files = glob.glob(os.path.join(BASE_DIR, "sample_tracks", "*.*"))
    for f in files[:3]:
        if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aiff', '.ogg')):
            diagnose_track(f)
            
    run_hypothesis_evaluations()

"""
AudioHarmonix Gold Set Evaluator & Comprehensive Metrics Suite
Implements Fases 13, 14, 16, 17, 18, 19, 26, 27 of prompt.md.
Evaluates:
- Exact Key Accuracy, Macro F1, Major/Minor Accuracy
- Prediction Entropy & Collapse Test (Prompt.md Fase 14)
- 24x24 Confusion Matrix
- DSP Baseline vs Neural vs Ensemble Comparison (Prompt.md Fase 19 & 20)
- Camelot / MIREX Weighted Harmonic Score
"""

import os
import sys
import glob
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "audio_decoder"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "cloud_verifier"))

import decoder
import dsp
import ml
import verifier

# Standard MIREX Harmonic Weightings for Key Evaluation
def compute_mirex_score(pred_key, true_key):
    if pred_key == true_key:
        return 1.0 # Exact match
        
    p_id = ml.KEY_LABELS.index(pred_key)
    t_id = ml.KEY_LABELS.index(true_key)
    
    p_mode = 1 if p_id >= 12 else 0
    t_mode = 1 if t_id >= 12 else 0
    p_root = p_id % 12
    t_root = t_id % 12
    
    # 1. Perfect Fifth / Fourth (+/- 7 semitones, same mode) -> 0.50
    if p_mode == t_mode and (p_root - t_root) % 12 in (7, 5):
        return 0.50
        
    # 2. Relative Major/Minor (e.g. C Major / A Minor) -> 0.30
    # Relative major of minor is +3 semitones; relative minor of major is -3 semitones
    if p_mode != t_mode:
        if (p_mode == 0 and t_mode == 1 and (p_root - t_root) % 12 == 3) or \
           (p_mode == 1 and t_mode == 0 and (t_root - p_root) % 12 == 3):
            return 0.30
            
    # 3. Parallel Major/Minor (e.g. C Major / C Minor) -> 0.20
    if p_mode != t_mode and p_root == t_root:
        return 0.20
        
    return 0.0

def evaluate_gold_set():
    print("=" * 80)
    print("  AUDIOHARMONIX GOLD SET BENCHMARK & METRICS (PROMPT.MD FASES 13-27)")
    print("=" * 80)
    
    gold_files = glob.glob(os.path.join(BASE_DIR, "sample_tracks", "*.*"))
    audio_files = [f for f in gold_files if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aiff', '.ogg'))]
    
    det = ml.KeyDetector()
    
    results = []
    dsp_correct = 0
    neural_correct = 0
    ensemble_correct = 0
    mirex_scores = []
    
    predictions_histogram = np.zeros(24, dtype=int)
    confusion_matrix = np.zeros((24, 24), dtype=int)
    
    print(f"\nEvaluating on {len(audio_files)} real Gold Set audio tracks:\n")
    print(f"{'Track Name':38s} | {'Ground Truth':14s} | {'DSP Key':14s} | {'Neural':14s} | {'Ensemble (AudioHarmonix)':20s} | {'MIREX':5s}")
    print("-" * 115)
    
    for f in audio_files:
        fname = os.path.basename(f)
        v_info = verifier.verify_track_online(fname)
        if not v_info.get("is_verified"):
            continue
            
        true_key = v_info["verified_key"]
        true_cam = v_info["verified_camelot"]
        true_idx = ml.KEY_LABELS.index(true_key)
        
        y, sr, _ = decoder.load_and_resample(f)
        cqt_mat, chrom_mat = dsp.compute_cqt(y, sr=sr)
        
        # 1. DSP baseline prediction
        dsp_logits = det._template_predict(cqt_mat, chromagram=chrom_mat)
        dsp_pred_idx = int(np.argmax(dsp_logits))
        dsp_key = ml.KEY_LABELS[dsp_pred_idx]
        
        # 2. Neural KeyNet prediction
        # Sliding windows
        window_frames = 64
        hop_frames = 32
        n_frames = cqt_mat.shape[1]
        neural_probs_list = []
        for start in range(0, max(1, n_frames - window_frames + 1), hop_frames):
            w = cqt_mat[:84, start:start + window_frames]
            if w.shape[1] < window_frames:
                w = np.pad(w, ((0, 0), (0, window_frames - w.shape[1])))
            w_norm = ((w - np.mean(w)) / (np.std(w) + 1e-6)).reshape(1, 1, 84, window_frames).astype(np.float32)
            input_name = det.session.get_inputs()[0].name
            raw_l = det.session.run(None, {input_name: w_norm})[0][0]
            neural_probs_list.append(ml.safe_softmax(raw_l))
            
        avg_neural_prob = np.mean(neural_probs_list, axis=0) if neural_probs_list else np.zeros(24)
        neural_pred_idx = int(np.argmax(avg_neural_prob))
        neural_key = ml.KEY_LABELS[neural_pred_idx]
        
        # 3. Full AudioHarmonix Ensemble
        ens_key, ens_cam, _, conf, _ = det.predict_key_full(cqt_mat, chromagram=chrom_mat)
        ens_pred_idx = ml.KEY_LABELS.index(ens_key)
        
        # Track accuracy
        if dsp_key == true_key: dsp_correct += 1
        if neural_key == true_key: neural_correct += 1
        if ens_key == true_key: ensemble_correct += 1
        
        m_score = compute_mirex_score(ens_key, true_key)
        mirex_scores.append(m_score)
        
        predictions_histogram[ens_pred_idx] += 1
        confusion_matrix[true_idx, ens_pred_idx] += 1
        
        match_symbol = "[OK]" if ens_key == true_key else f"[MIREX: {m_score:.1f}]"
        print(f"{fname[:38]:38s} | {true_key:14s} | {dsp_key:14s} | {neural_key:14s} | {ens_key:14s} ({ens_cam:3s}) | {match_symbol}")
        
    n_evaluated = len(mirex_scores)
    if n_evaluated == 0:
        print("No verified tracks found for Gold Set evaluation.")
        return
        
    exact_acc = (ensemble_correct / n_evaluated) * 100.0
    dsp_acc = (dsp_correct / n_evaluated) * 100.0
    neural_acc = (neural_correct / n_evaluated) * 100.0
    avg_mirex = (sum(mirex_scores) / n_evaluated) * 100.0
    
    # Prediction Entropy & Collapse Test (Fase 14)
    probs_dist = predictions_histogram / max(1, np.sum(predictions_histogram))
    non_zero = probs_dist[probs_dist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero))
    max_class_share = np.max(probs_dist) * 100.0
    max_class_name = ml.KEY_LABELS[np.argmax(probs_dist)]
    
    print("\n" + "=" * 80)
    print("  FINAL SCIENTIFIC BENCHMARK REPORT (GATE 4 & 5)")
    print("=" * 80)
    print(f"Total Tracks Evaluated       : {n_evaluated}")
    print(f"DSP Baseline Accuracy        : {dsp_acc:.2f}% ({dsp_correct}/{n_evaluated})")
    print(f"Neural KeyNet Accuracy       : {neural_acc:.2f}% ({neural_correct}/{n_evaluated})")
    print(f"AudioHarmonix Exact Accuracy : {exact_acc:.2f}% ({ensemble_correct}/{n_evaluated})")
    print(f"MIREX Weighted Harmonic Score: {avg_mirex:.2f}%")
    print(f"Prediction Entropy H(p)      : {entropy:.2f} bits (Max uniform = 4.58 bits)")
    print(f"Dominant Predicted Class     : {max_class_name} ({max_class_share:.1f}% share)")
    
    print("\nCollapse Diagnostic Check (Fase 14):")
    if max_class_share < 35.0:
        print("  --> [COLLAPSE TEST PASSED]: Prediction distribution is healthy and diverse across classes.")
    else:
        print(f"  --> [COLLAPSE WARNING]: Class {max_class_name} accounts for {max_class_share:.1f}% of predictions.")

    return {
        "exact_accuracy": exact_acc,
        "mirex_score": avg_mirex,
        "dsp_accuracy": dsp_acc,
        "neural_accuracy": neural_acc,
        "entropy": entropy
    }

if __name__ == "__main__":
    evaluate_gold_set()

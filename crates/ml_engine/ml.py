"""
AudioHarmonix Machine Learning & DSP Inference Engine
Comprehensive Hardened ML Pipeline:
- Key Classification via Deep CQT Neural Network (KeyNet ONNX) + Template Fallback
- Numerically Safe Softmax & Calibrated Confidence Estimation
- Exact 24-Key Camelot Wheel & OpenKey Mappings with Harmonic Compatibility
- Real-Time Structural Analysis via Multi-Task Bi-LSTM StructureNet (Boundary & Section Heads)
- Long Track Sliding Window Chunking with Overlap Crossfade & Beatgrid Alignment
- Continuous Psychoacoustic Energy Scoring (EnergyNet ONNX) (1-10 Scale)
- Hardened Active Learning Engine: Versioning, Validation, Atomic Writes, Thread Safety & Rollback
"""

import os
import sys
import time
import json
import shutil
import logging
import threading
import numpy as np

# Configure Structured Logging
logger = logging.getLogger("AudioHarmonix.ML")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    import librosa
except Exception:
    librosa = None

# Global Lock for Active Learning & Model Exports
_ACTIVE_LEARNING_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# 1. MUSICAL KEY DEFINITIONS & MAPPINGS
# ---------------------------------------------------------------------------

KEY_LABELS = [
    "C Major", "C# Major", "D Major", "D# Major", "E Major", "F Major",
    "F# Major", "G Major", "G# Major", "A Major", "A# Major", "B Major",
    "C Minor", "C# Minor", "D Minor", "D# Minor", "E Minor", "F Minor",
    "F# Minor", "G Minor", "G# Minor", "A Minor", "A# Minor", "B Minor"
]

# Exact 24-Key Camelot Wheel Mapping (Standard Electronic DJ Standard)
CAMELOT_MAP = {
    "C Major": "8B", "C# Major": "3B", "D Major": "10B", "D# Major": "5B",
    "E Major": "12B", "F Major": "7B", "F# Major": "2B", "G Major": "9B",
    "G# Major": "4B", "A Major": "11B", "A# Major": "6B", "B Major": "1B",
    "C Minor": "5A", "C# Minor": "12A", "D Minor": "7A", "D# Minor": "2A",
    "E Minor": "9A", "F Minor": "4A", "F# Minor": "11A", "G Minor": "6A",
    "G# Minor": "1A", "A Minor": "8A", "A# Minor": "3A", "B Minor": "10A"
}

# Exact 24-Key OpenKey Mapping (1d-12d for Major, 1m-12m for Minor)
OPENKEY_MAP = {
    "C Major": "1d", "C# Major": "8d", "D Major": "3d", "D# Major": "10d",
    "E Major": "5d", "F Major": "12d", "F# Major": "7d", "G Major": "2d",
    "G# Major": "9d", "A Major": "4d", "A# Major": "11d", "B Major": "6d",
    "C Minor": "10m", "C# Minor": "5m", "D Minor": "12m", "D# Minor": "7m",
    "E Minor": "2m", "F Minor": "9m", "F# Minor": "4m", "G Minor": "11m",
    "G# Minor": "6m", "A Minor": "1m", "A# Minor": "8m", "B Minor": "3m"
}

# Reverse Mappings for Quick Lookup
CAMELOT_TO_KEY = {v: k for k, v in CAMELOT_MAP.items()}
OPENKEY_TO_KEY = {v: k for k, v in OPENKEY_MAP.items()}


def get_camelot_compatibles(camelot_key):
    """
    Computes all harmonically compatible Camelot keys for a given key:
    - Same Key (exact match)
    - Relative Major/Minor (same number, opposite letter: e.g. 8A <-> 8B)
    - Subdominant (-1 hour / 4th degree: e.g. 8A -> 7A)
    - Dominant (+1 hour / 5th degree: e.g. 8A -> 9A)
    - Energy Boost / Cross-mode Dominant (+1 hour opposite letter: e.g. 8A -> 9B)
    """
    if not camelot_key or len(camelot_key) < 2:
        return ["8A", "8B", "7A", "9A"]

    try:
        letter = camelot_key[-1].upper()
        if letter not in ("A", "B"):
            return ["8A", "8B", "7A", "9A"]
            
        num = int(camelot_key[:-1])
        if num < 1 or num > 12:
            return ["8A", "8B", "7A", "9A"]
    except Exception:
        return ["8A", "8B", "7A", "9A"]

    other_letter = "B" if letter == "A" else "A"

    same = f"{num}{letter}"
    relative = f"{num}{other_letter}"
    subdom = f"{12 if num == 1 else num - 1}{letter}"
    dom = f"{1 if num == 12 else num + 1}{letter}"
    boost_sub = f"{12 if num == 1 else num - 1}{other_letter}"
    boost_dom = f"{1 if num == 12 else num + 1}{other_letter}"

    compatibles = []
    for k in [same, relative, subdom, dom, boost_sub, boost_dom]:
        if k not in compatibles:
            compatibles.append(k)

    return compatibles


def safe_softmax(x, axis=-1):
    """
    Numerically safe Softmax implementation handling NaN, Inf, and overflow/underflow.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x

    # Replace NaNs or Infs with zero
    x = np.nan_to_num(x, nan=0.0, posinf=30.0, neginf=-30.0)

    # Shift logits by max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x - x_max, -50.0, 50.0))
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)

    # Prevent division by zero
    sum_exp = np.where(sum_exp == 0.0, 1e-9, sum_exp)
    probs = exp_x / sum_exp
    return probs.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. KEY DETECTOR (KeyNet ONNX + DSP Template Fallback)
# ---------------------------------------------------------------------------

class KeyDetector:
    """
    Inference Engine for KeyNet ONNX (2D CNN over Constant-Q Transform).
    Expected input shape: ['batch_size', 1, 84, 'time_frames']
    Expected output shape: ['batch_size', 24]
    """
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "key_detector.onnx")

        self.model_path = model_path
        self.session = None

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
                logger.info(f"Loaded ONNX Key Detector model: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX KeyNet model ({e}). Using DSP fallback.")
        else:
            logger.info(f"ONNX model {model_path} not found. Using DSP fallback.")

    def predict_key_full(self, cqt_matrix, window_frames=64, hop_frames=32):
        """
        Predicts musical key across sliding temporal windows.
        Returns:
            detected_key (str): e.g. "C Minor"
            camelot_key (str): e.g. "5A"
            open_key (str): e.g. "10m"
            confidence (float): [0.30, 0.99]
            alternatives (list): Top predicted keys with probabilities
        """
        if cqt_matrix is None or cqt_matrix.ndim < 2 or cqt_matrix.shape[0] < 84 or cqt_matrix.shape[1] == 0:
            alts = [{"key": "C Major", "probability": 0.50}]
            return "C Major", "8B", "1d", 0.50, alts

        n_bins, n_frames = cqt_matrix.shape

        # Extract sliding temporal windows
        windows = []
        if n_frames <= window_frames:
            pad_size = max(0, window_frames - n_frames)
            w = np.pad(cqt_matrix[:84, :], ((0, 0), (0, pad_size)), mode='constant')
            windows.append(w)
        else:
            for start in range(0, n_frames - window_frames + 1, hop_frames):
                windows.append(cqt_matrix[:84, start:start + window_frames])

        if not windows:
            windows.append(np.zeros((84, window_frames), dtype=np.float32))

        all_window_probs = []

        for w in windows:
            logits = None
            if self.session is not None:
                try:
                    cqt_input = w[:84, :window_frames].reshape(1, 1, 84, window_frames).astype(np.float32)
                    input_name = self.session.get_inputs()[0].name
                    outputs = self.session.run(None, {input_name: cqt_input})
                    logits = outputs[0][0]
                except Exception as e:
                    logger.debug(f"ONNX Key inference step notice: {e}")
                    logits = self._template_predict(w)
            else:
                logits = self._template_predict(w)

            probs = safe_softmax(logits)
            all_window_probs.append(probs)

        # Average probabilities across temporal windows
        avg_probs = np.mean(all_window_probs, axis=0)

        # Inter-window prediction consistency metric
        top_per_window = [int(np.argmax(p)) for p in all_window_probs]
        best_idx = int(np.argmax(avg_probs))
        consistency = float(np.mean([1 if t == best_idx else 0 for t in top_per_window]))

        top_prob = float(avg_probs[best_idx])
        runner_up = float(np.partition(avg_probs, -2)[-2]) if len(avg_probs) >= 2 else 0.0
        margin = top_prob - runner_up

        # Calibrated heuristic confidence metric
        confidence = float(np.clip(0.40 + margin * 1.2 + consistency * 0.2, 0.30, 0.99))

        detected_key = KEY_LABELS[best_idx]
        camelot_key = CAMELOT_MAP.get(detected_key, "8A")
        open_key = OPENKEY_MAP.get(detected_key, "1m")

        sorted_indices = np.argsort(avg_probs)[::-1]
        alternatives = [
            {"key": KEY_LABELS[i], "probability": float(round(avg_probs[i], 4))}
            for i in sorted_indices
        ]

        return detected_key, camelot_key, open_key, confidence, alternatives

    def predict_key(self, cqt_matrix):
        det_key, camelot_key, open_key, confidence, _ = self.predict_key_full(cqt_matrix)
        return det_key, camelot_key, open_key, confidence

    def predict_key_detailed(self, cqt_matrix):
        det_key, camelot_key, open_key, confidence, alternatives = self.predict_key_full(cqt_matrix)
        compatibles = get_camelot_compatibles(camelot_key)

        return {
            "detected_key": det_key,
            "camelot_key": camelot_key,
            "open_key": open_key,
            "key_confidence": confidence,
            "compatible_keys": compatibles,
            "alternatives": alternatives[:5]
        }

    def _template_predict(self, cqt_matrix):
        """Krumhansl-Schmuckler Key Profile Correlation Fallback over CQT Chroma"""
        chroma = np.zeros(12, dtype=np.float32)
        for b in range(84):
            chroma[b % 12] += np.mean(cqt_matrix[b, :])

        norm = np.linalg.norm(chroma)
        if norm > 1e-6:
            chroma = chroma / norm

        # Krumhansl-Kessler Key Profiles
        major_prof = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
        minor_prof = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

        logits = np.zeros(24, dtype=np.float32)
        for i in range(12):
            p_maj = np.roll(major_prof, i)
            p_maj = p_maj / np.linalg.norm(p_maj)
            logits[i] = float(np.dot(chroma, p_maj))

            p_min = np.roll(minor_prof, i)
            p_min = p_min / np.linalg.norm(p_min)
            logits[i + 12] = float(np.dot(chroma, p_min))

        return logits


# ---------------------------------------------------------------------------
# 3. STRUCTURE DETECTOR & ACTIVE LEARNING MAPPINGS
# ---------------------------------------------------------------------------

class SectionClass:
    INTRO = 0       # Intro / First Beat Downbeat
    VERSE = 1       # Verse / Melodic Theme / Groove
    BUILDUP = 2     # Buildup / Riser
    DROP = 3        # Main Drop / Chorus / Peak Energy
    BREAKDOWN = 4   # Breakdown / Bridge / Strip-down
    OUTRO = 5       # Outro / Mix-out


SECTION_LABEL_NAMES = {
    0: "INTRO",
    1: "VERSE",
    2: "BUILDUP",
    3: "DROP",
    4: "BREAKDOWN",
    5: "OUTRO"
}


def map_cue_type_to_section_class(cue_type_str):
    """Canonical mapping from HotCue type names to StructureNet section class IDs (0-5)"""
    c_type = str(cue_type_str).upper()
    if "FIRST_BEAT" in c_type or "INTRO" in c_type:
        return SectionClass.INTRO
    elif "BUILDUP" in c_type or "RISER" in c_type:
        return SectionClass.BUILDUP
    elif "DROP" in c_type:
        return SectionClass.DROP
    elif "BREAK" in c_type:
        return SectionClass.BREAKDOWN
    elif "OUTRO" in c_type:
        return SectionClass.OUTRO
    elif "VERSE" in c_type or "GROOVE" in c_type:
        return SectionClass.VERSE
    return SectionClass.VERSE


class StructureDetector:
    """
    Inference Engine for AudioHarmonix StructureNet ONNX (Multi-Task Bi-LSTM).
    Expected inputs: ['batch_size', 1, 128, 'time_frames']
    Expected outputs:
      - boundary_logits: ['batch_size', 'time_steps', 1]
      - section_logits: ['batch_size', 'time_steps', 6]
    """
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "structure_detector.onnx")

        self.model_path = model_path
        self.session = None

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
                logger.info(f"Loaded ONNX Structure Detector: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX StructureNet model ({e}). Using DSP fallback.")
        else:
            logger.info(f"ONNX model {model_path} not found. Using DSP state machine fallback.")

    def predict_cues(self, y, beat_timestamps, duration_secs, sr=22050, dsp_fallback_fn=None,
                     boundary_threshold=0.35, min_cue_distance_sec=4.0):
        """
        Parses ONNX StructureNet boundary & section outputs into high-precision HotCues snapped to beatgrid.
        Gracefully falls back to DSP State Machine when ONNX is unavailable or produces insufficient points.
        """
        dsp_cues = []
        if dsp_fallback_fn is not None:
            try:
                dsp_cues = dsp_fallback_fn(y, beat_timestamps, duration_secs, sr)
            except Exception as e:
                logger.debug(f"DSP fallback error: {e}")

        if not beat_timestamps or len(beat_timestamps) == 0:
            beat_timestamps = [0.0]

        if self.session is None or len(y) == 0 or librosa is None:
            return dsp_cues if dsp_cues else [{"cue_type": "FIRST_BEAT", "position_secs": beat_timestamps[0], "hotcue_num": 1}]

        try:
            # 1. Compute 128-bin Mel Spectrogram (hop=512, sr=22050)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            norm_mel = ((log_mel - np.min(log_mel)) / (np.max(log_mel) - np.min(log_mel) + 1e-6)).astype(np.float32)

            total_mel_frames = norm_mel.shape[1]
            if total_mel_frames < 32:
                return dsp_cues if dsp_cues else [{"cue_type": "FIRST_BEAT", "position_secs": beat_timestamps[0], "hotcue_num": 1}]

            # 2. Run StructureNet ONNX
            # Process in temporal chunks if audio exceeds 4096 mel frames (~95s) or direct dynamic inference
            chunk_size = 2048
            hop_chunk = 1024

            if total_mel_frames <= chunk_size:
                inp = norm_mel.reshape(1, 1, 128, total_mel_frames)
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: inp})
                boundary_logits = outputs[0][0, :, 0]  # (time_steps,)
                section_logits = outputs[1][0, :, :]   # (time_steps, 6)
            else:
                # Sliding window chunking with linear overlap blending
                all_b_logits = []
                all_s_logits = []
                weights = []
                total_steps = total_mel_frames // 4
                boundary_accum = np.zeros(total_steps, dtype=np.float32)
                section_accum = np.zeros((total_steps, 6), dtype=np.float32)
                weight_accum = np.zeros(total_steps, dtype=np.float32)

                for start_f in range(0, total_mel_frames - 64, hop_chunk):
                    end_f = min(total_mel_frames, start_f + chunk_size)
                    chunk_mel = norm_mel[:, start_f:end_f]
                    # Ensure frames divisible by 4
                    pad_needed = (4 - (chunk_mel.shape[1] % 4)) % 4
                    if pad_needed > 0:
                        chunk_mel = np.pad(chunk_mel, ((0, 0), (0, pad_needed)), mode='constant')

                    inp = chunk_mel.reshape(1, 1, 128, chunk_mel.shape[1])
                    input_name = self.session.get_inputs()[0].name
                    outputs = self.session.run(None, {input_name: inp})
                    b_out = outputs[0][0, :, 0]
                    s_out = outputs[1][0, :, :]

                    start_step = start_f // 4
                    out_len = min(b_out.shape[0], total_steps - start_step)
                    if out_len <= 0:
                        continue

                    # Triangular window weighting for seamless chunk transitions
                    w = np.hanning(out_len * 2)[:out_len] + 1e-3
                    boundary_accum[start_step:start_step + out_len] += b_out[:out_len] * w
                    section_accum[start_step:start_step + out_len, :] += s_out[:out_len, :] * w[:, np.newaxis]
                    weight_accum[start_step:start_step + out_len] += w

                weight_accum = np.where(weight_accum == 0.0, 1.0, weight_accum)
                boundary_logits = boundary_accum / weight_accum
                section_logits = section_accum / weight_accum[:, np.newaxis]

            # 3. Convert logits to probabilities
            # Boundary sigmoid
            b_clipped = np.clip(boundary_logits, -15.0, 15.0)
            boundary_probs = 1.0 / (1.0 + np.exp(-b_clipped))
            section_probs = safe_softmax(section_logits, axis=-1)
            pred_sections = np.argmax(section_probs, axis=-1)

            # 4. Map model time steps to exact audio seconds
            # Each time_step in StructureNet represents 4 mel frames (4 * 512 samples = 2048 samples = ~92.88ms)
            time_per_step = (4 * 512) / float(sr)
            total_steps = len(boundary_probs)

            # 5. Extract Boundary Peaks & Section Transition Candidates
            candidates = []
            
            # First Beat is always at the true audio start
            first_beat = beat_timestamps[0] if beat_timestamps else 0.0
            candidates.append({"time_sec": first_beat, "cue_type": "FIRST_BEAT", "score": 1.0})

            # Peak picking across boundary probabilities
            for t in range(1, total_steps - 1):
                prob = boundary_probs[t]
                if prob >= boundary_threshold:
                    # Check if local maximum in a 3-step window (~278ms)
                    if prob >= boundary_probs[t - 1] and prob >= boundary_probs[t + 1]:
                        sec_t = t * time_per_step
                        if sec_t > first_beat + 2.0 and sec_t < duration_secs - 3.0:
                            s_cls = pred_sections[t]
                            candidates.append({
                                "time_sec": sec_t,
                                "section_cls": s_cls,
                                "score": float(prob)
                            })

            # 6. Snap Candidates to Nearest Beatgrid Downbeat & Assign Semantic Types
            snapped_cues = []
            drop_count = 0
            break_count = 0
            last_cue_time = -999.0

            # Always add FIRST_BEAT
            snapped_cues.append({
                "cue_type": "FIRST_BEAT",
                "position_secs": round(float(first_beat), 3)
            })
            last_cue_time = first_beat

            for cand in candidates[1:]:
                raw_time = cand["time_sec"]
                if raw_time - last_cue_time < min_cue_distance_sec:
                    continue

                # Find closest beat in beat_timestamps
                idx = int(np.argmin(np.abs(np.array(beat_timestamps) - raw_time)))
                snapped_time = beat_timestamps[idx]

                if snapped_time - last_cue_time < min_cue_distance_sec:
                    continue

                s_cls = cand.get("section_cls", SectionClass.VERSE)
                pos_ratio = snapped_time / max(1.0, duration_secs)

                if s_cls == SectionClass.DROP:
                    drop_count += 1
                    cue_type = f"DROP_{drop_count}" if drop_count > 1 else "DROP_1"
                elif s_cls == SectionClass.BREAKDOWN or s_cls == SectionClass.VERSE:
                    if pos_ratio > 0.40 and pos_ratio < 0.85:
                        break_count += 1
                        cue_type = f"BREAK_{break_count}" if break_count > 1 else "BREAKDOWN"
                    else:
                        cue_type = "BUILDUP" if pos_ratio < 0.40 else "BREAKDOWN"
                elif s_cls == SectionClass.BUILDUP:
                    cue_type = "BUILDUP"
                elif s_cls == SectionClass.OUTRO or pos_ratio >= 0.85:
                    cue_type = "OUTRO"
                else:
                    cue_type = "VERSE"

                snapped_cues.append({
                    "cue_type": cue_type,
                    "position_secs": round(float(snapped_time), 3)
                })
                last_cue_time = snapped_time

            # Ensure Outro cue exists if duration > 30s
            if duration_secs > 30.0 and not any("OUTRO" in c["cue_type"] for c in snapped_cues):
                outro_time = duration_secs - max(8.0, duration_secs * 0.15)
                # Snap to closest beat
                idx = int(np.argmin(np.abs(np.array(beat_timestamps) - outro_time)))
                snapped_outro = beat_timestamps[idx]
                if snapped_outro - last_cue_time >= min_cue_distance_sec:
                    snapped_cues.append({
                        "cue_type": "OUTRO",
                        "position_secs": round(float(snapped_outro), 3)
                    })

            # If ONNX produced insufficient cues, merge with DSP state machine
            if len(snapped_cues) <= 2 and dsp_cues and len(dsp_cues) > len(snapped_cues):
                logger.info("ONNX StructureNet yielded few points. Merging with DSP State Machine.")
                return dsp_cues

            # Assign sequential hotcue_num (1..8)
            final_cues = []
            for i, c in enumerate(snapped_cues[:8]):
                c["hotcue_num"] = i + 1
                final_cues.append(c)

            return final_cues

        except Exception as e:
            logger.warning(f"ONNX StructureNet inference notice ({e}). Falling back to DSP.")
            return dsp_cues if dsp_cues else [{"cue_type": "FIRST_BEAT", "position_secs": beat_timestamps[0], "hotcue_num": 1}]


# ---------------------------------------------------------------------------
# 4. ENERGY DETECTOR (EnergyNet ONNX + DSP Fallback)
# ---------------------------------------------------------------------------

class EnergyDetector:
    """
    Inference Engine for AudioHarmonix EnergyNet ONNX.
    Expected inputs: ['batch_size', 1, 128, 'time_frames']
    Expected output: energy_score ['batch_size', 1]
    """
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "energy_detector.onnx")

        self.model_path = model_path
        self.session = None

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
                logger.info(f"Loaded ONNX Energy Detector: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX EnergyNet model ({e}). Using DSP fallback.")
        else:
            logger.info(f"ONNX model {model_path} not found. Using DSP fallback.")

    def predict_energy_raw(self, y, sr=22050, dsp_fallback_energy=5.0):
        """
        Returns continuous psychoacoustic energy rating [1.0, 10.0].
        """
        if self.session is None or len(y) == 0 or librosa is None:
            return float(dsp_fallback_energy)

        try:
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            norm_mel = ((log_mel - np.min(log_mel)) / (np.max(log_mel) - np.min(log_mel) + 1e-6)).astype(np.float32)

            if norm_mel.shape[1] >= 128:
                start_f = (norm_mel.shape[1] - 128) // 2
                clip_mel = norm_mel[:, start_f:start_f + 128]
            else:
                pad = 128 - norm_mel.shape[1]
                clip_mel = np.pad(norm_mel, ((0, 0), (0, pad)), mode='constant')

            inp = clip_mel.reshape(1, 1, 128, 128)
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: inp})
            raw_val = float(np.squeeze(output[0]))
            return float(np.clip(raw_val, 1.0, 10.0))
        except Exception as e:
            logger.debug(f"ONNX EnergyNet inference notice ({e}). Using DSP fallback.")
            return float(dsp_fallback_energy)

    def predict_energy(self, y, sr=22050, dsp_fallback_energy=5):
        """
        Returns integer energy rating [1, 10] for DJ display and ID3 tagging.
        """
        raw_score = self.predict_energy_raw(y, sr=sr, dsp_fallback_energy=dsp_fallback_energy)
        return int(np.clip(round(raw_score), 1, 10))


# ---------------------------------------------------------------------------
# 5. HARDENED ACTIVE LEARNING & MODEL VERSIONING PIPELINE
# ---------------------------------------------------------------------------

def validate_onnx_model(onnx_path, expected_input_name="mel_spectrogram"):
    """
    Validates that an ONNX model file loads cleanly and runs inference without NaN or Inf.
    """
    if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) < 1024:
        return False, "File does not exist or is empty"

    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
        
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        if len(inputs) == 0 or len(outputs) == 0:
            return False, "Model has no inputs or outputs"

        # Run test inference with dummy input
        dummy = np.random.randn(1, 1, 128, 128).astype(np.float32)
        out = session.run(None, {inputs[0].name: dummy})
        
        for o in out:
            if np.isnan(o).any() or np.isinf(o).any():
                return False, "Model outputs contain NaN or Inf"

        return True, "Model valid"
    except Exception as e:
        return False, str(e)


def adapt_structure_model(audio_path, user_cues):
    """
    Hardened Few-Shot Online Active Learning Adaptation:
    - Thread-safe execution using _ACTIVE_LEARNING_LOCK
    - Preserves base model immutability with version archive
    - Atomic file operations with validation before activation
    - Rollback safety on failure
    """
    if not os.path.exists(audio_path) or not user_cues:
        return False

    with _ACTIVE_LEARNING_LOCK:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            checkpoint_dir = os.path.join(base_dir, "training", "checkpoints", "structure_net")
            models_dir = os.path.join(base_dir, "models")
            versions_dir = os.path.join(models_dir, "structure_detector_versions")
            active_onnx = os.path.join(models_dir, "structure_detector.onnx")

            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(versions_dir, exist_ok=True)

            # Preserve base model copy if not already backed up
            base_backup_onnx = os.path.join(versions_dir, "base.onnx")
            if os.path.exists(active_onnx) and not os.path.exists(base_backup_onnx):
                shutil.copy2(active_onnx, base_backup_onnx)
                logger.info(f"Created permanent base model archive: {base_backup_onnx}")

            sys.path.insert(0, os.path.join(base_dir, "crates", "audio_decoder"))
            import decoder

            y, sr, dur = decoder.load_and_resample(audio_path)
            if len(y) == 0:
                return False

            if librosa is None:
                return False

            # Extract Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            norm_mel = ((log_mel - np.min(log_mel)) / (np.max(log_mel) - np.min(log_mel) + 1e-6)).astype(np.float32)

            if norm_mel.shape[1] >= 128:
                start_f = (norm_mel.shape[1] - 128) // 2
                clip_mel = norm_mel[:, start_f:start_f + 128]
            else:
                pad = 128 - norm_mel.shape[1]
                clip_mel = np.pad(norm_mel, ((0, 0), (0, pad)), mode='constant')

            inp_tensor = torch.from_numpy(clip_mel).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)

            # Construct Ground-Truth Target Tensors from User Corrections
            target_sec = torch.zeros(1, 32, dtype=torch.long)
            target_bnd = torch.zeros(1, 32, 1, dtype=torch.float32)

            for c in user_cues:
                pos = float(c.get("position_secs", 0))
                c_type = str(c.get("cue_type", "")).upper()
                frame_idx = int(np.clip((pos / max(1.0, dur)) * 32, 0, 31))
                
                # Mark boundary peak with local smoothing
                target_bnd[0, max(0, frame_idx - 1):min(32, frame_idx + 2), 0] = 1.0
                
                # Map to canonical section class (0..5)
                sec_val = map_cue_type_to_section_class(c_type)
                target_sec[0, frame_idx:min(32, frame_idx + 8)] = sec_val

            sys.path.insert(0, os.path.join(base_dir, "training"))
            from train_structure_net_massive import AudioHarmonixStructureNet

            model = AudioHarmonixStructureNet(num_classes=6)
            best_ckpt = os.path.join(checkpoint_dir, "best_model.pt")
            if os.path.exists(best_ckpt):
                try:
                    model.load_state_dict(torch.load(best_ckpt, map_location="cpu", weights_only=True))
                except Exception as e:
                    logger.debug(f"Checkpoint load notice: {e}")

            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            ce_loss = nn.CrossEntropyLoss(label_smoothing=0.02)
            bce_loss = nn.BCEWithLogitsLoss()

            # 3 Steps of careful fine-tuning
            model.train()
            for _ in range(3):
                optimizer.zero_grad()
                b_pred, s_pred = model(inp_tensor)
                loss = ce_loss(s_pred.reshape(-1, 6), target_sec.reshape(-1)) + 0.5 * bce_loss(b_pred.reshape(-1, 1), target_bnd.reshape(-1, 1))
                loss.backward()
                optimizer.step()

            # Save Adapted PyTorch Checkpoint
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "user_adapted_model.pt"))

            # Export to Atomic Temporary ONNX File
            timestamp = int(time.time())
            temp_onnx = os.path.join(versions_dir, f"temp_adapted_{timestamp}.onnx")
            versioned_onnx = os.path.join(versions_dir, f"adapted_{timestamp}.onnx")

            dummy = torch.randn(1, 1, 128, 128, dtype=torch.float32)
            torch.onnx.export(
                model, dummy, temp_onnx, export_params=True, opset_version=17,
                do_constant_folding=True, dynamo=False,
                input_names=['mel_spectrogram'], output_names=['boundary_logits', 'section_logits'],
                dynamic_axes={
                    'mel_spectrogram': {0: 'batch_size', 3: 'time_frames'},
                    'boundary_logits': {0: 'batch_size', 1: 'time_steps'},
                    'section_logits': {0: 'batch_size', 1: 'time_steps'}
                }
            )

            # Validate Exported ONNX Model before Activation
            is_valid, val_msg = validate_onnx_model(temp_onnx)
            if not is_valid:
                logger.error(f"Active Learning ONNX validation failed ({val_msg}). Retaining previous active model.")
                if os.path.exists(temp_onnx):
                    os.remove(temp_onnx)
                return False

            # Archive valid adapted version
            shutil.copy2(temp_onnx, versioned_onnx)

            # Atomic Replace of Active Model
            if os.path.exists(active_onnx):
                try:
                    os.replace(temp_onnx, active_onnx)
                except OSError:
                    shutil.copy2(versioned_onnx, active_onnx)
                    if os.path.exists(temp_onnx):
                        os.remove(temp_onnx)
            else:
                shutil.copy2(versioned_onnx, active_onnx)
                if os.path.exists(temp_onnx):
                    os.remove(temp_onnx)

            # Write Version Info Manifest
            version_manifest = {
                "active_version": f"adapted_{timestamp}.onnx",
                "timestamp": timestamp,
                "track_source": os.path.basename(audio_path),
                "num_cues": len(user_cues),
                "status": "active"
            }
            with open(os.path.join(versions_dir, "version_info.json"), "w", encoding="utf-8") as f:
                json.dump(version_manifest, f, indent=2)

            logger.info(f"Active Learning: Successfully adapted & activated StructureNet version 'adapted_{timestamp}.onnx' ({len(user_cues)} cues).")
            return True

        except Exception as e:
            logger.error(f"Active Learning adaptation error: {e}")
            return False


def rollback_structure_model():
    """
    Restores the base factory StructureNet model from the version archive.
    """
    with _ACTIVE_LEARNING_LOCK:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(base_dir, "models")
        versions_dir = os.path.join(models_dir, "structure_detector_versions")
        base_backup = os.path.join(versions_dir, "base.onnx")
        active_onnx = os.path.join(models_dir, "structure_detector.onnx")

        if not os.path.exists(base_backup):
            logger.warning("Base model backup not found. Cannot perform rollback.")
            return False

        try:
            shutil.copy2(base_backup, active_onnx)
            version_manifest = {
                "active_version": "base.onnx",
                "timestamp": int(time.time()),
                "status": "rolled_back_to_base"
            }
            with open(os.path.join(versions_dir, "version_info.json"), "w", encoding="utf-8") as f:
                json.dump(version_manifest, f, indent=2)

            logger.info("Successfully rolled back StructureNet to base model.")
            return True
        except Exception as e:
            logger.error(f"Rollback error: {e}")
            return False

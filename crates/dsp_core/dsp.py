"""
AudioHarmonix DSP Core Engine
Section 4: Signal Processing, Spectral Analysis, Energy Scoring, Beat Tracking & Waveform Generation
"""

import os
from dataclasses import dataclass, field
import numpy as np
from scipy import signal
from scipy.fft import rfft, irfft

SAMPLE_RATE = 22050
FFT_WINDOW = 2048
HOP_SIZE = 512

def compute_rms_energy(y):
    """Calculates root-mean-square energy of 1D signal y"""
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y))))

def compute_spectral_centroid(y, sr=SAMPLE_RATE, n_fft=FFT_WINDOW, hop_length=HOP_SIZE):
    """Calculates average spectral centroid (audio brightness)"""
    _, _, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    stft = np.abs(Zxx)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    
    # Avoid zero division
    sum_mag = np.sum(stft, axis=0) + 1e-9
    centroid = np.sum(stft * freqs[:, np.newaxis], axis=0) / sum_mag
    return float(np.mean(centroid))

def compute_high_frequency_content(y, sr=SAMPLE_RATE, n_fft=FFT_WINDOW, hop_length=HOP_SIZE):
    """High Frequency Content (HFC) for cymbals/percussion density"""
    _, _, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    stft = np.abs(Zxx)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    high_mask = freqs >= 4000.0
    if not np.any(high_mask):
        return 0.0
    high_energy = np.mean(stft[high_mask, :])
    return float(high_energy)

def compute_transient_density(y, sr=SAMPLE_RATE, hop_length=HOP_SIZE):
    """Calculates density of transient peaks per second"""
    diff = np.diff(np.abs(y))
    threshold = np.std(diff) * 2.0
    peaks = np.where(diff > threshold)[0]
    duration_secs = max(1.0, len(y) / float(sr))
    return float(len(peaks) / duration_secs)

def compute_energy_score(y, sr=SAMPLE_RATE):
    """
    Section 2.5 & 4: Energy Score (1 to 10 scale)
    Combined metric of RMS energy, Spectral Centroid, High-Frequency Content, and Transient density.
    """
    rms = compute_rms_energy(y)
    centroid = compute_spectral_centroid(y, sr=sr)
    hfc = compute_high_frequency_content(y, sr=sr)
    transients = compute_transient_density(y, sr=sr)

    # Normalize metrics to 0-1 scale
    norm_rms = np.clip(rms / 0.25, 0.0, 1.0)
    norm_centroid = np.clip((centroid - 500.0) / 4500.0, 0.0, 1.0)
    norm_hfc = np.clip(hfc / 0.05, 0.0, 1.0)
    norm_transients = np.clip(transients / 50.0, 0.0, 1.0)

    # Ponderated formula
    raw_score = 0.45 * norm_rms + 0.25 * norm_centroid + 0.15 * norm_hfc + 0.15 * norm_transients
    
    # Scale from 1 to 10
    energy_1_to_10 = int(np.clip(round(raw_score * 9.0 + 1.0), 1, 10))
    return energy_1_to_10

def compute_cqt(y, sr=SAMPLE_RATE, n_bins=84, bins_per_octave=12, fmin=32.7, hop_length=HOP_SIZE):
    """
    Section 4.2: High-Precision Constant-Q Transform (CQT) & Octave-Weighted Harmonic Chromagram
    Uses harmonic band filtering (C2-C7) to eliminate sub-bass kick bleed and accurately separate Major/Minor tonality.
    """
    try:
        import librosa
        cqt_complex = librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin, hop_length=hop_length)
        cqt_matrix = np.abs(cqt_complex).astype(np.float32)
        
        # Log-scale compression
        cqt_matrix = np.log1p(cqt_matrix)
        
        # Harmonic Octave-Filtered Chroma CQT (fmin=C2 ~ 65.4Hz, ignoring sub-bass rumble)
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, n_octaves=6, fmin=librosa.note_to_hz('C2'), hop_length=hop_length).astype(np.float32)
        col_sums = np.sum(chromagram, axis=0) + 1e-9
        chromagram = chromagram / col_sums
        return cqt_matrix, chromagram
    except Exception:
        pass

    freqs = fmin * (2.0 ** (np.arange(n_bins) / float(bins_per_octave)))
    f_stft, t_stft, Zxx = signal.stft(y, fs=sr, nperseg=FFT_WINDOW, noverlap=FFT_WINDOW-hop_length)
    mag_stft = np.abs(Zxx)
    
    cqt_matrix = np.zeros((n_bins, mag_stft.shape[1]), dtype=np.float32)
    stft_bin_indices = np.digitize(f_stft, freqs) - 1
    
    for b in range(n_bins):
        mask = (stft_bin_indices == b)
        if np.any(mask):
            cqt_matrix[b, :] = np.mean(mag_stft[mask, :], axis=0)

    # Harmonic band (bins 12 to 72: C2 to C6)
    chromagram = np.zeros((12, mag_stft.shape[1]), dtype=np.float32)
    for b in range(12, min(72, n_bins)):
        chromagram[b % 12, :] += cqt_matrix[b, :]

    col_sums = np.sum(chromagram, axis=0) + 1e-9
    chromagram = chromagram / col_sums
    
    return cqt_matrix, chromagram

def compute_onset_detection_function(y, sr=SAMPLE_RATE, hop_length=HOP_SIZE):
    """
    Section 4.3: Onset Detection Function (ODF)
    Combines spectral flux with bass band transient energy.
    """
    f_stft, t_stft, Zxx = signal.stft(y, fs=sr, nperseg=1024, noverlap=1024-hop_length)
    mag = np.abs(Zxx)
    
    # Spectral Flux (half-wave rectified positive difference)
    diff = np.diff(mag, axis=1)
    diff = np.maximum(0, diff)
    odf = np.sum(diff, axis=0)
    
    # Bass band boost (20Hz - 150Hz)
    bass_idx = np.where((f_stft >= 20.0) & (f_stft <= 150.0))[0]
    if len(bass_idx) > 0:
        bass_diff = np.sum(diff[bass_idx, :], axis=0)
        odf = odf + 2.0 * bass_diff
        
    # Smooth ODF
    if len(odf) > 5:
        odf = signal.medfilt(odf, kernel_size=5)
        
    return odf

def estimate_bpm_and_beatgrid(y, sr=SAMPLE_RATE, hop_length=HOP_SIZE):
    """
    Section 4.3 & 4.5: BPM Detection & Beat Tracking
    Evaluates autocorrelation of ODF in range [60, 180 BPM].
    Returns:
        bpm (float): Estimated BPM
        bpm_confidence (float): Autocorrelation peak ratio confidence [0.0, 1.0]
        beat_timestamps (list): List of beat timestamps in seconds
        is_variable_bpm (bool): True if tempo variance exceeds threshold (>3%)
    """
    odf = compute_onset_detection_function(y, sr=sr, hop_length=hop_length)
    fps = sr / float(hop_length)
    
    if len(odf) < 100:
        return 120.0, 0.5, [0.0], False

    # Autocorrelation
    odf_norm = odf - np.mean(odf)
    autocorr = signal.correlate(odf_norm, odf_norm, mode='full', method='fft')
    autocorr = autocorr[len(odf)-1:]
    
    # Lag limits for 60 BPM to 180 BPM
    min_lag = int(round(fps * 60.0 / 180.0))  # ~7.3 frames for 180 BPM
    max_lag = int(round(fps * 60.0 / 60.0))   # ~22.0 frames for 60 BPM
    
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1

    search_window = autocorr[min_lag:max_lag+1]
    if len(search_window) == 0 or np.max(search_window) <= 0:
        return 120.0, 0.5, [0.0], False

    best_lag_rel = np.argmax(search_window)
    best_lag = min_lag + best_lag_rel
    
    # BPM calculation
    raw_bpm = (fps * 60.0) / float(best_lag)
    
    # Refine BPM to standard DJ range [80, 175] if doubled/halved
    bpm = raw_bpm
    while bpm < 80.0:
        bpm *= 2.0
    while bpm > 175.0:
        bpm /= 2.0

    bpm = round(bpm, 2)

    # Calculate BPM confidence (ratio of main peak vs median of search window)
    peak_val = search_window[best_lag_rel]
    avg_val = np.mean(search_window) + 1e-6
    ratio = peak_val / avg_val
    confidence = float(np.clip((ratio - 1.2) / 3.0, 0.2, 0.99))

    # Beat tracking timestamps
    beat_interval_sec = 60.0 / bpm
    duration_sec = len(y) / float(sr)
    
def find_first_audio_onset(y, sr=SAMPLE_RATE, threshold_db=-40.0):
    """
    Scans from t=0.0s forward to find the exact first frame where audio crosses threshold_db.
    Ensures FIRST_BEAT is ALWAYS anchored in the true beginning of the track (0.0s to 2.0s).
    """
    if len(y) == 0:
        return 0.0

    hop = 256
    frame_len = 512
    rms_env = [np.sqrt(np.mean(np.square(y[i:i+frame_len]))) for i in range(0, len(y)-frame_len, hop)]
    
    max_rms = max(rms_env) if rms_env else 1.0
    if max_rms <= 0:
        return 0.0

    threshold_linear = max_rms * (10.0 ** (threshold_db / 20.0))

    for idx, val in enumerate(rms_env):
        if val >= threshold_linear:
            t_sec = (idx * hop) / float(sr)
            return min(2.0, round(t_sec, 3))

    return 0.0

def estimate_bpm_and_beatgrid(y, sr=SAMPLE_RATE, hop_length=HOP_SIZE):
    """
    Section 4.3 & 4.5: BPM Detection & Beat Tracking
    Evaluates autocorrelation of ODF in range [60, 180 BPM].
    Returns:
        bpm (float): Estimated BPM
        bpm_confidence (float): Autocorrelation peak ratio confidence [0.0, 1.0]
        beat_timestamps (list): List of beat timestamps in seconds
        is_variable_bpm (bool): True if tempo variance exceeds threshold (>3%)
    """
    odf = compute_onset_detection_function(y, sr=sr, hop_length=hop_length)
    fps = sr / float(hop_length)
    
    if len(odf) < 100:
        return 120.0, 0.5, [0.0], False

    # Autocorrelation
    odf_norm = odf - np.mean(odf)
    autocorr = signal.correlate(odf_norm, odf_norm, mode='full', method='fft')
    autocorr = autocorr[len(odf)-1:]
    
    # Lag limits for 60 BPM to 180 BPM
    min_lag = int(round(fps * 60.0 / 180.0))  # ~7.3 frames for 180 BPM
    max_lag = int(round(fps * 60.0 / 60.0))   # ~22.0 frames for 60 BPM
    
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1

    search_window = autocorr[min_lag:max_lag+1]
    if len(search_window) == 0 or np.max(search_window) <= 0:
        return 120.0, 0.5, [0.0], False

    best_lag_rel = np.argmax(search_window)
    best_lag = min_lag + best_lag_rel
    
    # BPM calculation
    raw_bpm = (fps * 60.0) / float(best_lag)
    
    # Refine BPM to standard DJ range [80, 175] if doubled/halved
    bpm = raw_bpm
    while bpm < 80.0:
        bpm *= 2.0
    while bpm > 175.0:
        bpm /= 2.0

    # Tempo Octave Correction for Electronic / House / Dance music (80 - 95 BPM -> 120 - 142.5 BPM)
    if 80.0 <= bpm <= 95.0:
        bpm = bpm * 1.5

    bpm = round(bpm, 2)

    # Calculate BPM confidence (ratio of main peak vs median of search window)
    peak_val = search_window[best_lag_rel]
    avg_val = np.mean(search_window) + 1e-6
    ratio = peak_val / avg_val
    confidence = float(np.clip((ratio - 1.2) / 3.0, 0.2, 0.99))

    # Beat tracking timestamps
    beat_interval_sec = 60.0 / bpm
    duration_sec = len(y) / float(sr)
    
    # Find initial downbeat phase by detecting first audible audio onset
    start_sec = find_first_audio_onset(y, sr=sr)
    
    beat_timestamps = []
    t = start_sec
    while t < duration_sec:
        beat_timestamps.append(round(t, 3))
        t += beat_interval_sec

    # Check for Variable BPM (inter-beat interval variance)
    ibis = np.diff(beat_timestamps)
    is_variable_bpm = False
    if len(ibis) > 5:
        std_ibi = np.std(ibis)
        mean_ibi = np.mean(ibis)
        if (std_ibi / mean_ibi) > 0.03:
            is_variable_bpm = True

    return bpm, confidence, beat_timestamps, is_variable_bpm


# ===========================================================================
# EXPERIMENTAL DSP CONFIGURATION & 3-LAYER ARCHITECTURE
# ===========================================================================

@dataclass
class ExperimentalDSPConfig:
    """
    Centralized configuration for experimental DSP cue point detection.
    All thresholds are relative/dynamic parameters subject to empirical calibration.
    """
    min_phrase_beats: int = 16
    rel_transient_threshold: float = 0.15      # Break: TransientEnergy < 15% of track peak
    rel_beat_pulse_threshold: float = 0.25     # Break: BeatPulse < 25% of track peak
    min_break_duration_sec: float = 8.0        # Minimum duration to confirm Break
    buildup_slope_threshold: float = 0.55      # Buildup: Mid-High energy slope > +55%
    buildup_flux_accel_threshold: float = 1.70 # Buildup: Spectral flux acceleration > 1.7x
    buildup_lookback_beats: int = 32           # Maximum beats to inspect before a Drop
    pre_drop_gap_threshold: float = 0.85       # Pre-drop: Last beat RMS < 85% of mean

DEFAULT_EXPERIMENTAL_CONFIG = ExperimentalDSPConfig()


# ---------------------------------------------------------------------------
# LAYER 1: EVIDENCE EXTRACTION
# ---------------------------------------------------------------------------

def extract_dsp_evidence(y, sr, beat_timestamps, duration_secs, config=DEFAULT_EXPERIMENTAL_CONFIG):
    """
    Layer 1: Multi-band acoustic evidence extraction across phrase intervals.
    Produces continuous diagnostic features: TransientEnergy, BeatPulse, e_bass,
    MidHighEnergy, SubBassEnergy, SpectralFlux, and continuous BreakScore.
    """
    if not beat_timestamps or len(beat_timestamps) == 0:
        beat_timestamps = [0.0]

    phrase_step = config.min_phrase_beats
    phrase_beats = beat_timestamps[::phrase_step] if len(beat_timestamps) >= phrase_step else beat_timestamps
    if not phrase_beats:
        phrase_beats = [beat_timestamps[0]]

    # Filters for distinct frequency bands
    try:
        sos_sub = signal.butter(4, [30.0, 100.0], btype='bandpass', fs=sr, output='sos')
        sos_bass = signal.butter(4, [20.0, 250.0], btype='bandpass', fs=sr, output='sos')
        sos_mh = signal.butter(4, [1000.0, 8000.0], btype='bandpass', fs=sr, output='sos')
        sos_sub_only = signal.butter(4, 150.0, btype='lowpass', fs=sr, output='sos')

        y_sub = signal.sosfilt(sos_sub, y)
        y_bass = signal.sosfilt(sos_bass, y)
        y_mh = signal.sosfilt(sos_mh, y)
        y_sub_only = signal.sosfilt(sos_sub_only, y)
    except Exception:
        y_sub = y
        y_bass = y
        y_mh = y
        y_sub_only = y

    # Sub-bass envelope for transient onset tracking
    try:
        env_sub = signal.medfilt(np.abs(signal.hilbert(y_sub)), 101)
    except Exception:
        env_sub = np.abs(y_sub)

    diff_env_sub = np.maximum(0, np.diff(env_sub))

    phrase_records = []
    for i, p_sec in enumerate(phrase_beats):
        p_end = phrase_beats[i+1] if i+1 < len(phrase_beats) else duration_secs
        s_idx, e_idx = int(p_sec * sr), int(p_end * sr)

        seg_full = y[s_idx:e_idx]
        seg_bass = y_bass[s_idx:e_idx]
        seg_mh = y_mh[s_idx:e_idx]
        seg_sub_o = y_sub_only[s_idx:e_idx]
        seg_diff = diff_env_sub[s_idx:min(len(diff_env_sub), e_idx)]

        e_rms = float(np.sqrt(np.mean(np.square(seg_full)))) if len(seg_full) > 0 else 0.0
        e_bass = float(np.sqrt(np.mean(np.square(seg_bass)))) if len(seg_bass) > 0 else 0.0
        e_mh = float(np.sqrt(np.mean(np.square(seg_mh)))) if len(seg_mh) > 0 else 0.0
        e_sub_o = float(np.sqrt(np.mean(np.square(seg_sub_o)))) if len(seg_sub_o) > 0 else 0.0

        trans_e = float(np.mean(np.square(seg_diff)) * 1e6) if len(seg_diff) > 0 else 0.0

        # Synchronous beat pulse on beatgrid
        sec_beats = [b for b in beat_timestamps if p_sec <= b < p_end]
        beat_pulses = []
        for bt in sec_beats:
            b_i = int(bt * sr)
            w = env_sub[max(0, b_i - int(0.04*sr)):min(len(env_sub), b_i + int(0.08*sr))]
            if len(w) > 0:
                beat_pulses.append(float(np.max(w) - np.min(w)))
        beat_pulse = float(np.mean(beat_pulses)) if beat_pulses else 0.0

        flux = float(np.mean(np.square(np.diff(np.abs(seg_mh)))) * 1e6) if len(seg_mh) > 1 else 0.0

        phrase_records.append({
            "sec": p_sec,
            "end_sec": p_end,
            "e_rms": e_rms,
            "e_bass": e_bass,
            "e_mh": e_mh,
            "e_sub_o": e_sub_o,
            "trans_e": trans_e,
            "beat_pulse": beat_pulse,
            "flux": flux
        })

    # Reference levels: 90th percentile across the track to avoid single-peak skew
    all_trans = [p["trans_e"] for p in phrase_records]
    all_pulse = [p["beat_pulse"] for p in phrase_records]
    all_bass = [p["e_bass"] for p in phrase_records]

    ref_trans = float(np.percentile(all_trans, 90)) if all_trans else 1.0
    ref_pulse = float(np.percentile(all_pulse, 90)) if all_pulse else 1.0
    ref_bass = float(np.percentile(all_bass, 90)) if all_bass else 1.0

    ref_trans = max(1e-6, ref_trans)
    ref_pulse = max(1e-6, ref_pulse)
    ref_bass = max(1e-6, ref_bass)

    # Calculate normalized scores and continuous BreakScore
    for p in phrase_records:
        norm_trans = min(1.0, p["trans_e"] / ref_trans)
        norm_pulse = min(1.0, p["beat_pulse"] / ref_pulse)
        norm_bass = min(1.0, p["e_bass"] / ref_bass)

        # Break evidence: Low transient activity + low synchronous pulse in sub-bass
        t_inact = max(0.0, 1.0 - norm_trans / max(1e-6, config.rel_transient_threshold)) if norm_trans < config.rel_transient_threshold else 0.0
        p_inact = max(0.0, 1.0 - norm_pulse / max(1e-6, config.rel_beat_pulse_threshold)) if norm_pulse < config.rel_beat_pulse_threshold else 0.0

        break_score = 0.55 * t_inact + 0.45 * p_inact
        p["break_score"] = float(np.clip(break_score, 0.0, 1.0))
        p["norm_trans"] = float(norm_trans)
        p["norm_pulse"] = float(norm_pulse)
        p["norm_bass"] = float(norm_bass)

    return {
        "phrases": phrase_records,
        "ref_trans": ref_trans,
        "ref_pulse": ref_pulse,
        "ref_bass": ref_bass,
        "y": y,
        "y_bass": y_bass,
        "y_mh": y_mh,
        "y_sub_only": y_sub_only
    }


# ---------------------------------------------------------------------------
# LAYER 2: CANDIDATE GENERATION
# ---------------------------------------------------------------------------

def generate_structural_candidates(evidence, beat_timestamps, duration_secs, sr=SAMPLE_RATE, config=DEFAULT_EXPERIMENTAL_CONFIG):
    """
    Layer 2: Generates candidate structural events (DROP, BREAKDOWN, BUILDUP, OUTRO)
    based on evidence signals without coupling to final sequential HotCue numbering.
    """
    phrases = evidence["phrases"]
    y = evidence["y"]
    y_bass = evidence["y_bass"]
    y_mh = evidence["y_mh"]
    max_bass = evidence["ref_bass"]

    candidates = []
    first_beat = beat_timestamps[0] if beat_timestamps else 0.0

    candidates.append({
        "cue_type": "FIRST_BEAT",
        "position_secs": first_beat,
        "score": 1.0,
        "diagnostics": {"type": "anchor"}
    })

    # Step 1: Detect Drop candidates
    drop_candidates = []
    in_drop = False
    last_drop_sec = -999.0
    last_break_sec = -999.0

    for i, p in enumerate(phrases):
        sec = p["sec"]
        if sec >= duration_secs * 0.90:
            continue

        prev_b = phrases[i-1]["e_bass"] if i > 0 else 0.0
        delta_b = p["e_bass"] - prev_b

        # High-precision Drop Condition
        is_drop = (p["e_bass"] >= 0.58 * max_bass and delta_b >= 0.035) or \
                  (p["e_bass"] >= 0.80 * max_bass and delta_b >= 0.015) or \
                  (p["norm_pulse"] >= 0.60 and p["norm_trans"] >= 0.50 and delta_b >= 0.02)

        # Break condition: Transient & synchronous pulse drop in sub-bass
        is_break = (p["break_score"] >= 0.40) or (p["norm_trans"] <= 0.12 and p["norm_pulse"] <= 0.20)

        # Don't register a drop candidate at the very start (0.0s) unless it has full maximum bass
        is_intro_beat = (sec <= first_beat + 4.0) and (p["e_bass"] < 0.80 * max_bass)

        if is_drop and not in_drop and not is_intro_beat and (sec - last_drop_sec >= 14.0):
            # Snap position to peak sub-bass onset beat
            cand_beats = [b for b in beat_timestamps if sec - 1.0 <= b <= sec + 2.5]
            best_sec = sec
            if cand_beats:
                best_sec = max(cand_beats, key=lambda b: np.mean(y_bass[int(b*sr):int((b+0.2)*sr)]**2) if int((b+0.2)*sr) < len(y_bass) else 0)

            drop_cand = {
                "cue_type": "DROP",
                "position_secs": float(best_sec),
                "score": float(p["norm_bass"]),
                "diagnostics": {"e_bass": p["e_bass"], "trans_e": p["trans_e"], "beat_pulse": p["beat_pulse"]}
            }
            drop_candidates.append(drop_cand)
            candidates.append(drop_cand)
            in_drop = True
            last_drop_sec = float(best_sec)

        # Breakdown is structurally valid only AFTER the first consolidated Drop has occurred
        elif is_break and in_drop and len(drop_candidates) > 0 and (sec - last_drop_sec >= 12.0) and (sec - last_break_sec >= 14.0):
            break_cand = {
                "cue_type": "BREAKDOWN",
                "position_secs": float(sec),
                "score": float(p["break_score"]),
                "diagnostics": {"break_score": p["break_score"], "norm_trans": p["norm_trans"], "norm_pulse": p["norm_pulse"]}
            }
            candidates.append(break_cand)
            in_drop = False
            last_break_sec = float(sec)

    # Step 2: Detect Buildup candidates anchored to detected Drops
    bpm_est = 120.0
    if len(beat_timestamps) > 1:
        ibi_median = float(np.median(np.diff(beat_timestamps)))
        if ibi_median > 0:
            bpm_est = 60.0 / ibi_median
    beat_dur = 60.0 / max(1.0, bpm_est)

    for d_cand in drop_candidates:
        d_time = d_cand["position_secs"]
        # Analyze lookback window (up to 32 beats before drop)
        lookback_sec = min(d_time - first_beat, config.buildup_lookback_beats * beat_dur)
        if lookback_sec < 8 * beat_dur:
            continue

        t_start = max(first_beat, d_time - lookback_sec)
        s_i, e_i = int(t_start * sr), int(d_time * sr)
        seg_mh = y_mh[s_i:e_i]

        if len(seg_mh) < int(8 * beat_dur * sr):
            continue

        # Evaluate energy slope and flux acceleration across 4 sub-blocks
        num_sub = 4
        sub_len = len(seg_mh) // num_sub
        if sub_len <= 0:
            continue

        rms_sub = [float(np.sqrt(np.mean(seg_mh[k*sub_len:(k+1)*sub_len]**2))) for k in range(num_sub)]
        slope_mh = (rms_sub[-1] - rms_sub[0]) / (rms_sub[0] + 1e-6)

        half = len(seg_mh) // 2
        f1 = float(np.mean(np.square(np.diff(np.abs(seg_mh[:half])))) * 1e6) if half > 1 else 1.0
        f2 = float(np.mean(np.square(np.diff(np.abs(seg_mh[half:])))) * 1e6) if half > 1 else 1.0
        flux_accel = f2 / max(1e-6, f1)

        # Pre-drop gap (last 1 beat before drop vs segment mean)
        gap_samples = int(beat_dur * sr)
        last_beat_rms = float(np.sqrt(np.mean(y[max(0, e_i-gap_samples):e_i]**2))) if e_i > gap_samples else 1.0
        seg_mean_rms = float(np.sqrt(np.mean(y[s_i:e_i]**2))) if e_i > s_i else 1.0
        gap_ratio = last_beat_rms / max(1e-6, seg_mean_rms)

        is_buildup = (slope_mh >= config.buildup_slope_threshold) or (flux_accel >= config.buildup_flux_accel_threshold)

        if is_buildup:
            # Anchor buildup start at 16 beats before drop (or 32 beats if slope is extended)
            buildup_span_beats = 32 if (slope_mh > 1.2 and lookback_sec >= 30 * beat_dur) else 16
            buildup_sec = max(first_beat, d_time - (buildup_span_beats * beat_dur))

            # Snap to closest beat
            idx_b = int(np.argmin(np.abs(np.array(beat_timestamps) - buildup_sec)))
            snapped_buildup = float(beat_timestamps[idx_b])

            buildup_score = float(np.clip(0.50 + 0.25 * min(2.0, slope_mh) + 0.25 * min(2.0, flux_accel - 1.0), 0.0, 1.0))
            candidates.append({
                "cue_type": "BUILDUP",
                "position_secs": snapped_buildup,
                "score": buildup_score,
                "diagnostics": {
                    "slope_mh": slope_mh,
                    "flux_accel": flux_accel,
                    "gap_ratio": gap_ratio,
                    "target_drop_sec": d_time
                }
            })

    # Step 3: Outro candidate
    last_event_sec = max([c["position_secs"] for c in candidates]) if candidates else first_beat
    outro_candidates = [p["sec"] for p in phrases if p["sec"] >= max(first_beat + 30.0, max(last_event_sec + 15.0, duration_secs * 0.80)) and p["sec"] < duration_secs - 1.5]
    outro_sec = outro_candidates[0] if outro_candidates else max(first_beat + 10.0, duration_secs - max(2.0, duration_secs * 0.12))
    candidates.append({
        "cue_type": "OUTRO",
        "position_secs": float(outro_sec),
        "score": 1.0,
        "diagnostics": {"type": "outro_anchor"}
    })

    return candidates


# ---------------------------------------------------------------------------
# LAYER 3: HOTCUE CONVERSION & NUMBERING
# ---------------------------------------------------------------------------

def convert_candidates_to_hotcues(candidates, beat_timestamps, duration_secs, min_spacing_sec=6.0):
    """
    Layer 3: Snaps candidates to beatgrid, enforces bounds and minimum spacing,
    assigns clean sequential labels (DROP_1, DROP_2, BREAK_1, BUILDUP_1...),
    and numbers hotcue_num 1..8.
    """
    if not beat_timestamps or len(beat_timestamps) == 0:
        beat_timestamps = [0.0]

    first_beat_sec = float(beat_timestamps[0])
    cues_sorted = sorted(candidates, key=lambda x: x["position_secs"])

    filtered_cues = []
    last_pos = -999.0

    for c in cues_sorted:
        pos = float(c["position_secs"])

        if pos >= duration_secs:
            pos = max(first_beat_sec, round(max(0.0, duration_secs - 1.5), 3))

        if c["cue_type"] == "FIRST_BEAT":
            pos = round(first_beat_sec, 3)

        # Snap to nearest beatgrid timestamp
        idx = int(np.argmin(np.abs(np.array(beat_timestamps) - pos)))
        snapped_pos = round(float(beat_timestamps[idx]), 3)

        if (snapped_pos - last_pos >= min_spacing_sec) or c["cue_type"] == "FIRST_BEAT":
            c_copy = dict(c)
            c_copy["position_secs"] = snapped_pos
            filtered_cues.append(c_copy)
            last_pos = snapped_pos

    # Deduplicate positions
    unique_cues = []
    seen_pos = set()
    for c in filtered_cues:
        pos_key = round(c["position_secs"], 2)
        if pos_key not in seen_pos:
            seen_pos.add(pos_key)
            unique_cues.append(c)

    # Sequential semantic labeling
    drop_idx = 0
    break_idx = 0
    buildup_idx = 0
    for c in unique_cues:
        raw_type = c["cue_type"]
        if raw_type.startswith("DROP"):
            drop_idx += 1
            c["cue_type"] = f"DROP_{drop_idx}"
        elif raw_type.startswith("BREAK"):
            break_idx += 1
            c["cue_type"] = f"BREAK_{break_idx}"
        elif raw_type.startswith("BUILDUP"):
            buildup_idx += 1
            c["cue_type"] = f"BUILDUP_{buildup_idx}" if buildup_idx > 1 else "BUILDUP"

    # If no drop was found, ensure DROP_1 exists
    if drop_idx == 0 and len(unique_cues) > 0:
        unique_cues.insert(1, {"cue_type": "DROP_1", "position_secs": first_beat_sec})

    final_cues = sorted(unique_cues, key=lambda x: x["position_secs"])[:8]
    for idx, c in enumerate(final_cues, 1):
        c["hotcue_num"] = idx

    return final_cues


# ===========================================================================
# CUE DETECTION ENGINES (LEGACY & EXPERIMENTAL)
# ===========================================================================

def detect_cue_points_legacy(y, beat_timestamps, duration_secs, sr=SAMPLE_RATE):
    """
    Section 4.4 (Legacy): Original State Machine HotCue Analyzer.
    Preserved 100% intact for backward-compatibility and A/B benchmarking.
    """
    if not beat_timestamps or len(beat_timestamps) == 0:
        beat_timestamps = [0.0]

    # For short preview tracks (< 30s), keep simple FIRST_BEAT + OUTRO
    if duration_secs < 30.0:
        first_b = beat_timestamps[0]
        outro_b = max(first_b + 4.0, duration_secs - max(1.5, duration_secs * 0.20))
        return [
            {"cue_type": "FIRST_BEAT", "position_secs": round(float(first_b), 3), "hotcue_num": 1},
            {"cue_type": "OUTRO", "position_secs": round(float(outro_b), 3), "hotcue_num": 2}
        ]

    phrase_beats = beat_timestamps[::16] if len(beat_timestamps) >= 16 else beat_timestamps
    if not phrase_beats:
        phrase_beats = [beat_timestamps[0]]

    try:
        sos_bass = signal.butter(4, [20.0, 250.0], btype='bandpass', fs=sr, output='sos')
        y_bass = np.abs(signal.sosfilt(sos_bass, y))
    except Exception:
        y_bass = np.abs(y)

    phrase_states = []
    for i, p_sec in enumerate(phrase_beats):
        p_end = phrase_beats[i+1] if i+1 < len(phrase_beats) else duration_secs
        s_idx, e_idx = int(p_sec * sr), int(p_end * sr)

        segment_bass = y_bass[s_idx:e_idx]
        segment_full = y[s_idx:e_idx]

        e_bass = float(np.sqrt(np.mean(np.square(segment_bass)))) if len(segment_bass) > 0 else 0.0
        e_rms = float(np.sqrt(np.mean(np.square(segment_full)))) if len(segment_full) > 0 else 0.0
        phrase_states.append({"sec": p_sec, "e_bass": e_bass, "e_rms": e_rms})

    max_bass = max([p["e_bass"] for p in phrase_states]) if phrase_states else 1.0
    if max_bass <= 0:
        max_bass = 1e-6

    cues = []
    first_beat = beat_timestamps[0]
    cues.append({
        "cue_type": "FIRST_BEAT",
        "position_secs": round(float(first_beat), 3)
    })

    in_drop = False
    drop_count = 0
    break_count = 0
    last_drop_sec = -999.0
    last_break_sec = -999.0

    for i, p in enumerate(phrase_states):
        sec = p["sec"]
        if sec >= duration_secs * 0.90 and len(cues) > 1:
            continue

        prev_b = phrase_states[i-1]["e_bass"] if i > 0 else 0.0
        delta_b = p["e_bass"] - prev_b

        is_drop = (p["e_bass"] >= 0.58 * max_bass and delta_b >= 0.035) or (p["e_bass"] >= 0.80 * max_bass and delta_b >= 0.015)
        is_break = (p["e_bass"] <= 0.55 * max_bass) and (delta_b <= -0.035 or p["e_bass"] <= 0.35 * max_bass)

        if is_drop and not in_drop and (sec - last_drop_sec >= 14.0):
            drop_count += 1
            label = f"DROP_{drop_count}"

            candidate_beats = [b for b in beat_timestamps if sec - 1.0 <= b <= sec + 2.5]
            best_sec = sec
            if candidate_beats:
                best_sec = max(candidate_beats, key=lambda b: np.mean(y_bass[int(b*sr):int((b+0.2)*sr)]**2) if int((b+0.2)*sr) < len(y_bass) else 0)

            cues.append({
                "cue_type": label,
                "position_secs": round(float(best_sec), 3)
            })
            in_drop = True
            last_drop_sec = float(best_sec)

        elif is_break and in_drop and (sec - last_drop_sec >= 12.0) and (sec - last_break_sec >= 14.0):
            break_count += 1
            label = f"BREAK_{break_count}"
            cues.append({
                "cue_type": label,
                "position_secs": round(float(sec), 3)
            })
            in_drop = False
            last_break_sec = float(sec)

    if drop_count == 0:
        cues.append({"cue_type": "DROP_1", "position_secs": round(float(first_beat), 3)})

    outro_candidates = [p["sec"] for p in phrase_states if p["sec"] >= max(first_beat + 30.0, max(last_drop_sec + 15.0, duration_secs * 0.80)) and p["sec"] < duration_secs - 1.5]
    outro_sec = outro_candidates[0] if outro_candidates else max(first_beat + 10.0, duration_secs - max(2.0, duration_secs * 0.12))
    cues.append({
        "cue_type": "OUTRO",
        "position_secs": round(float(outro_sec), 3)
    })

    return _sanitize_and_number_cues(cues, beat_timestamps, duration_secs)


def detect_cue_points_experimental(y, beat_timestamps, duration_secs, sr=SAMPLE_RATE,
                                  config=DEFAULT_EXPERIMENTAL_CONFIG, return_diagnostics=False):
    """
    Experimental 3-Layer DSP HotCue Analyzer:
    Layer 1: extract_dsp_evidence (Transients, Pulses, Flux, continuous Scores)
    Layer 2: generate_structural_candidates (Drop, Break, Buildup, Outro candidates)
    Layer 3: convert_candidates_to_hotcues (Beatgrid snapping & 1..8 numbering)
    """
    if not beat_timestamps or len(beat_timestamps) == 0:
        beat_timestamps = [0.0]

    if duration_secs < 30.0:
        return detect_cue_points_legacy(y, beat_timestamps, duration_secs, sr)

    # Layer 1
    evidence = extract_dsp_evidence(y, sr, beat_timestamps, duration_secs, config=config)

    # Layer 2
    candidates = generate_structural_candidates(evidence, beat_timestamps, duration_secs, sr=sr, config=config)

    # Layer 3
    hotcues = convert_candidates_to_hotcues(candidates, beat_timestamps, duration_secs)

    if return_diagnostics:
        return hotcues, {"evidence": evidence, "candidates": candidates}
    return hotcues


def detect_cue_points(y, beat_timestamps, duration_secs, sr=SAMPLE_RATE, mode=None):
    """
    Central HotCue detection entry point with transparent A/B switching.
    Default mode is 'legacy'. Set environment variable AUDIOHARMONIX_DSP_MODE='experimental'
    or pass mode='experimental' to activate the experimental engine.
    """
    env_mode = os.getenv("AUDIOHARMONIX_DSP_MODE", "legacy").lower()
    selected_mode = mode.lower() if mode is not None else env_mode

    if selected_mode == "experimental":
        return detect_cue_points_experimental(y, beat_timestamps, duration_secs, sr=sr)
    else:
        return detect_cue_points_legacy(y, beat_timestamps, duration_secs, sr=sr)

def _sanitize_and_number_cues(cues, beat_timestamps, duration_secs):
    """Sorts cues chronologically, enforces bounds, filters near-duplicates, and assigns hotcue_num 1..8 with clean sequential labels"""
    cues_sorted = sorted(cues, key=lambda x: x["position_secs"])

    first_beat_sec = beat_timestamps[0] if beat_timestamps else 0.0

    filtered_cues = []
    last_pos = -999.0

    for c in cues_sorted:
        # Cap cue position to duration_secs - 0.5s
        if c["position_secs"] >= duration_secs:
            c["position_secs"] = max(first_beat_sec, round(max(0.0, duration_secs - 1.5), 3))

        if c["cue_type"] == "FIRST_BEAT":
            c["position_secs"] = round(first_beat_sec, 3)

        # Keep cues if spaced at least 6.0s apart, or if it's the main FIRST_BEAT
        if (c["position_secs"] - last_pos >= 6.0) or c["cue_type"] == "FIRST_BEAT":
            filtered_cues.append(c)
            last_pos = c["position_secs"]

    # Deduplicate by position_secs
    unique_cues = []
    seen_pos = set()
    for c in filtered_cues:
        pos_key = round(c["position_secs"], 2)
        if pos_key not in seen_pos:
            seen_pos.add(pos_key)
            unique_cues.append(c)

    # Re-label drops and breaks cleanly and sequentially (DROP_1, DROP_2..., BREAK_1, BREAK_2...)
    drop_idx = 0
    break_idx = 0
    for c in unique_cues:
        if c["cue_type"].startswith("DROP"):
            drop_idx += 1
            c["cue_type"] = f"DROP_{drop_idx}"
        elif c["cue_type"].startswith("BREAK"):
            break_idx += 1
            c["cue_type"] = f"BREAK_{break_idx}"

    final_cues = sorted(unique_cues, key=lambda x: x["position_secs"])[:8]
    for idx, c in enumerate(final_cues, 1):
        c["hotcue_num"] = idx

    return final_cues

def generate_3band_waveform_peaks(y, sr=SAMPLE_RATE, num_points=600):
    """
    Section 2.7: Fast 3-Band RGB Waveform Peak Generator
    """
    if len(y) == 0:
        return {"low": [], "mid": [], "high": []}

    # Downsample audio by factor of 5 for blazingly fast waveform visualization calculation
    ds_factor = 5
    y_ds = y[::ds_factor]
    sr_ds = sr // ds_factor

    # Fast bandpass filters on downsampled audio
    try:
        sos_low = signal.butter(2, [20.0, 250.0], btype='bandpass', fs=sr_ds, output='sos')
        sos_mid = signal.butter(2, [250.0, max(255.0, min(2000.0, sr_ds/2.1))], btype='bandpass', fs=sr_ds, output='sos')
        sos_high = signal.butter(2, [max(500.0, sr_ds/4.0), sr_ds/2.05], btype='bandpass', fs=sr_ds, output='sos')

        y_low = np.abs(signal.sosfilt(sos_low, y_ds))
        y_mid = np.abs(signal.sosfilt(sos_mid, y_ds))
        y_high = np.abs(signal.sosfilt(sos_high, y_ds))
    except Exception:
        y_low = np.abs(y_ds)
        y_mid = np.abs(y_ds) * 0.7
        y_high = np.abs(y_ds) * 0.4

    chunk_size = max(1, len(y_ds) // num_points)
    
    low_peaks = [float(np.max(y_low[i:i+chunk_size])) for i in range(0, len(y_ds), chunk_size)]
    mid_peaks = [float(np.max(y_mid[i:i+chunk_size])) for i in range(0, len(y_ds), chunk_size)]
    high_peaks = [float(np.max(y_high[i:i+chunk_size])) for i in range(0, len(y_ds), chunk_size)]

    max_l = max(1e-6, max(low_peaks)) if low_peaks else 1.0
    max_m = max(1e-6, max(mid_peaks)) if mid_peaks else 1.0
    max_h = max(1e-6, max(high_peaks)) if high_peaks else 1.0

    return {
        "low": [round(v / max_l, 3) for v in low_peaks],
        "mid": [round(v / max_m, 3) for v in mid_peaks],
        "high": [round(v / max_h, 3) for v in high_peaks]
    }

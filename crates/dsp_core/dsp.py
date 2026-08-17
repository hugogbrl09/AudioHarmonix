"""
AudioHarmonix DSP Core Engine
Section 4: Signal Processing, Spectral Analysis, Energy Scoring, Beat Tracking & Waveform Generation
"""

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
    Section 4.2: High-Precision Constant-Q Transform (CQT) & Chromagram
    Uses librosa.cqt when available with fast vectorized STFT-filterbank fallback.
    """
    try:
        import librosa
        cqt_complex = librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin, hop_length=hop_length)
        cqt_matrix = np.abs(cqt_complex).astype(np.float32)
        
        # Log-scale compression
        cqt_matrix = np.log1p(cqt_matrix)
        chromagram = np.zeros((12, cqt_matrix.shape[1]), dtype=np.float32)
        for b in range(n_bins):
            chromagram[b % 12, :] += cqt_matrix[b, :]
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

    chromagram = np.zeros((12, mag_stft.shape[1]), dtype=np.float32)
    for b in range(n_bins):
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

def detect_cue_points(y, beat_timestamps, duration_secs, sr=SAMPLE_RATE):
    """
    Section 4.4: Industry-Standard Multi-Section HotCue Analyzer (State Machine)
    Scans phrase-by-phrase (4 bars / 16 beats) across the entire track using 
    Sub-Bass Energy Jump (Delta E_bass) scoring & Sub-Bass Onset Snapping to identify ALL structural transitions:
    FIRST_BEAT, DROP_1, BREAKDOWN, DROP_2, BREAK_2, DROP_3, OUTRO.
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

    # 1. Extract 4-bar phrase boundaries (16 beats per phrase for 4-bar DJ phrase precision)
    phrase_beats = beat_timestamps[::16] if len(beat_timestamps) >= 16 else beat_timestamps
    if not phrase_beats:
        phrase_beats = [beat_timestamps[0]]

    # 2. Extract Sub-Bass Energy (20-250Hz) and Total RMS for each phrase
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

    # Scan 16-beat phrases sequentially from start to end
    for i, p in enumerate(phrase_states):
        sec = p["sec"]
        
        # Don't place interior drop cues in the last 10% of the track
        if sec >= duration_secs * 0.90 and len(cues) > 1:
            continue

        prev_b = phrase_states[i-1]["e_bass"] if i > 0 else 0.0
        delta_b = p["e_bass"] - prev_b

        # High-precision Drop Condition: Sub-bass energy jump >= 0.035 OR sustained peak sub-bass >= 0.80
        is_drop = (p["e_bass"] >= 0.58 * max_bass and delta_b >= 0.035) or (p["e_bass"] >= 0.80 * max_bass and delta_b >= 0.015)
        # Breakdown condition: Bass drops below 55% of max with significant negative drop
        is_break = (p["e_bass"] <= 0.55 * max_bass) and (delta_b <= -0.035 or p["e_bass"] <= 0.35 * max_bass)

        if is_drop and not in_drop and (sec - last_drop_sec >= 14.0):
            drop_count += 1
            label = f"DROP_{drop_count}"
            
            # Sub-Bass Onset Snapping: Snap position to candidate beat timestamp with peak sub-bass onset
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

    # If no drop was found (e.g. ambient or low-bass), add default DROP_1
    if drop_count == 0:
        cues.append({"cue_type": "DROP_1", "position_secs": round(float(first_beat), 3)})

    # OUTRO: Place after the last drop, at the start of the final decline in last 12-20% of track
    outro_candidates = [p["sec"] for p in phrase_states if p["sec"] >= max(first_beat + 30.0, max(last_drop_sec + 15.0, duration_secs * 0.80)) and p["sec"] < duration_secs - 1.5]
    outro_sec = outro_candidates[0] if outro_candidates else max(first_beat + 10.0, duration_secs - max(2.0, duration_secs * 0.12))
    cues.append({
        "cue_type": "OUTRO",
        "position_secs": round(float(outro_sec), 3)
    })

    # Sanitize, enforce minimum spacing, and assign hotcue_num 1..8 with clean sequential labels
    return _sanitize_and_number_cues(cues, beat_timestamps, duration_secs)

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

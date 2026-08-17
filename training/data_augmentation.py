"""
AudioHarmonix Advanced Audio & Spectrogram Data Augmentation Pipeline
Features:
- Harmonic Pitch-Shifting (12x exact CQT vertical shift with Camelot label rotation)
- Multi-Band SpecAugment (Time & Frequency Masking for CQT and Log-Mel)
- Dynamic Mixup (convex linear interpolation of features and probability targets)
- Time-Stretching (rescaling temporal axes without altering pitch)
"""

import numpy as np

def pitch_shift_cqt(cqt_matrix, semitones, label_id):
    """
    Shifts an 84-bin Constant-Q Spectrogram (12 bins/octave) by `semitones` (-6 to +6)
    and rotates the corresponding Key label (0..23) accurately.
    Returns:
        shifted_cqt (np.ndarray): (84, T)
        new_label_id (int): 0..23
    """
    if semitones == 0 or cqt_matrix is None:
        return cqt_matrix, label_id

    n_bins, n_frames = cqt_matrix.shape
    shifted_cqt = np.zeros_like(cqt_matrix, dtype=np.float32)

    if semitones > 0:
        shifted_cqt[semitones:, :] = cqt_matrix[:-semitones, :]
    else:
        shift_abs = abs(semitones)
        shifted_cqt[:-shift_abs, :] = cqt_matrix[shift_abs:, :]

    # Rotate Key Label in Modulo 12 space
    if label_id < 12:
        # Major Keys (0..11)
        new_label_id = (label_id + semitones) % 12
    else:
        # Minor Keys (12..23)
        new_label_id = 12 + ((label_id - 12 + semitones) % 12)

    return shifted_cqt, new_label_id


def apply_spec_augment(spec, num_freq_masks=2, num_time_masks=2, max_freq_mask=8, max_time_mask=12):
    """
    Applies SpecAugment masking directly on 2D Spectrograms (Freq, Time).
    """
    spec_aug = spec.copy()
    n_freqs, n_frames = spec_aug.shape

    # Frequency Masking
    for _ in range(num_freq_masks):
        f_width = np.random.randint(1, max_freq_mask + 1)
        f_start = np.random.randint(0, max(1, n_freqs - f_width))
        spec_aug[f_start:f_start + f_width, :] = 0.0

    # Time Masking
    for _ in range(num_time_masks):
        t_width = np.random.randint(1, max_time_mask + 1)
        t_start = np.random.randint(0, max(1, n_frames - t_width))
        spec_aug[:, t_start:t_start + t_width] = 0.0

    return spec_aug


def apply_mixup(feat_a, label_a, feat_b, label_b, alpha=0.2):
    """
    Applies Mixup augmentation: convex combination of feature pairs and one-hot targets.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    mixed_feat = lam * feat_a + (1.0 - lam) * feat_b

    # Construct one-hot targets if given as scalar indices
    if np.isscalar(label_a) or label_a.ndim == 0:
        target_a = np.zeros(24, dtype=np.float32)
        target_a[int(label_a)] = 1.0
        target_b = np.zeros(24, dtype=np.float32)
        target_b[int(label_b)] = 1.0
    else:
        target_a = label_a
        target_b = label_b

    mixed_target = lam * target_a + (1.0 - lam) * target_b
    return mixed_feat, mixed_target

"""
AudioHarmonix Audio Decoder Module
Section 4.1: Audio Decoding & Resampling (22.050 Hz Mono PCM Float32)
"""

import os
import wave
import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SAMPLE_RATE = 22050

def load_and_resample(file_path, target_sr=TARGET_SAMPLE_RATE):
    """
    Decodes MP3, WAV, FLAC, AAC, OGG, AIFF audio files.
    Converts audio to Mono PCM Float32 and resamples down to target_sr (22,050 Hz).
    Returns:
        samples (np.ndarray): 1D float32 array normalized to [-1.0, 1.0]
        sr (int): Target sample rate (22050)
        duration_secs (float): Total audio duration in seconds
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        data, orig_sr = sf.read(file_path, dtype='float32')
        # Convert stereo/multichannel to mono by averaging channels
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        duration_secs = float(len(data)) / float(orig_sr)

        # Resample to target_sr (22050Hz) if necessary
        if orig_sr != target_sr:
            num_target_samples = int(round(duration_secs * target_sr))
            data = signal.resample(data, num_target_samples).astype(np.float32)

        # Normalize amplitude peak to avoid clipping
        max_val = np.max(np.abs(data))
        if max_val > 1e-6:
            data = data / max_val

        return data, target_sr, duration_secs

    except Exception as e:
        # Fallback for standard WAV files if soundfile fails
        try:
            with wave.open(file_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                orig_sr = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)

                if sampwidth == 2:
                    dtype = np.int16
                elif sampwidth == 4:
                    dtype = np.int32
                else:
                    dtype = np.uint8

                data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                if dtype == np.uint8:
                    data = (data - 128.0) / 128.0
                elif dtype == np.int16:
                    data = data / 32768.0
                elif dtype == np.int32:
                    data = data / 2147483648.0

                if n_channels > 1:
                    data = data.reshape(-1, n_channels).mean(axis=1)

                duration_secs = len(data) / orig_sr
                if orig_sr != target_sr:
                    num_target_samples = int(round(duration_secs * target_sr))
                    data = signal.resample(data, num_target_samples).astype(np.float32)

                max_val = np.max(np.abs(data))
                if max_val > 1e-6:
                    data = data / max_val

                return data, target_sr, duration_secs
        except Exception as err:
            raise RuntimeError(f"Failed to decode audio file '{file_path}': {e} | Fallback err: {err}")

"""
AudioHarmonix Audio Test Suite & Track Synthesizer
Generates real audio files with distinct musical keys, BPMs, kick drums, basslines, and synth chords.
"""

import os
import wave
import numpy as np

SAMPLE_RATE = 22050

# Frequencies for musical notes (Hz)
NOTE_FREQS = {
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56,
    "E3": 164.81, "F3": 174.61, "F#3": 185.00, "G3": 196.00,
    "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00,
    "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "E5": 659.25, "G5": 783.99
}

# Scale notes for key simulation
KEY_SCALES = {
    "A Minor": ["A3", "C4", "E4", "A4"],     # Camelot 8A
    "C Major": ["C4", "E4", "G4", "C5"],     # Camelot 8B
    "C Minor": ["C3", "D#3", "G3", "C4"],    # Camelot 5A
    "E Minor": ["E3", "G3", "B3", "E4"],     # Camelot 9A
    "F Major": ["F3", "A3", "C4", "F4"]      # Camelot 7B
}

def generate_kick(duration_sec=0.15, sr=SAMPLE_RATE):
    """Synthesizes a punchy kick drum pulse with pitch sweep"""
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    freq_env = 150.0 * np.exp(-35.0 * t) + 40.0
    phase = 2.0 * np.pi * np.cumsum(freq_env) / sr
    amp_env = np.exp(-12.0 * t)
    return np.sin(phase) * amp_env

def generate_track(file_path, key_name, bpm, duration_sec=12.0, sr=SAMPLE_RATE):
    """Generates a complete synthesized electronic track with kick beat and synth chord progression"""
    num_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    
    y = np.zeros(num_samples, dtype=np.float32)

    # 1. Four-on-the-floor kick beat
    beat_interval = 60.0 / float(bpm)
    kick = generate_kick(duration_sec=0.15, sr=sr)
    beat_times = np.arange(0, duration_sec, beat_interval)

    for b in beat_times:
        start_idx = int(b * sr)
        end_idx = min(num_samples, start_idx + len(kick))
        y[start_idx:end_idx] += kick[:end_idx - start_idx] * 0.7

    # 2. Synthesize harmonic synth chord progression
    scale_notes = KEY_SCALES.get(key_name, ["A3", "C4", "E4"])
    for note in scale_notes:
        f = NOTE_FREQS.get(note, 440.0)
        # Add fundamental + harmonics
        synth = 0.3 * np.sin(2.0 * np.pi * f * t) + 0.1 * np.sin(2.0 * np.pi * f * 2.0 * t)
        # Apply gentle modulation envelope
        mod = 0.5 + 0.5 * np.sin(2.0 * np.pi * (bpm / 60.0) * t)
        y += synth * mod * 0.25

    # 3. Add high-frequency cymbal/hi-hat click on off-beats
    offbeat_times = beat_times + (beat_interval / 2.0)
    for ob in offbeat_times:
        if ob < duration_sec:
            start_idx = int(ob * sr)
            click_len = int(sr * 0.04)
            end_idx = min(num_samples, start_idx + click_len)
            noise = (np.random.rand(end_idx - start_idx) * 2.0 - 1.0) * np.exp(-40.0 * np.linspace(0, 0.04, end_idx - start_idx))
            y[start_idx:end_idx] += noise * 0.2

    # Peak normalization
    max_amp = np.max(np.abs(y))
    if max_amp > 1e-6:
        y = y / max_amp * 0.9

    # Save to WAV file (16-bit PCM)
    audio_int16 = (y * 32767).astype(np.int16)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

    print(f"Generated sample audio track: '{file_path}' ({bpm} BPM, Key: {key_name})")

def generate_all_samples(output_dir="sample_tracks"):
    os.makedirs(output_dir, exist_ok=True)
    
    samples = [
        ("track_01_starlight.wav", "A Minor", 126.0),   # Camelot 8A
        ("track_02_deep_rhythm.wav", "C Minor", 124.0), # Camelot 5A
        ("track_03_synth_wave.wav", "C Major", 120.0),   # Camelot 8B
        ("track_04_sunset_groove.wav", "E Minor", 128.0),# Camelot 9A
        ("track_05_funky_disco.wav", "F Major", 122.0)   # Camelot 7B
    ]

    paths = []
    for fname, key_name, bpm in samples:
        fp = os.path.join(output_dir, fname)
        generate_track(fp, key_name, bpm, duration_sec=10.0)
        paths.append(fp)

    return paths

if __name__ == "__main__":
    generate_all_samples()

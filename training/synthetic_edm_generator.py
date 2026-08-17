"""
AudioHarmonix Procedural Synthetic EDM Generator
Generates mathematically clean, high-fidelity synthetic electronic dance music (EDM) tracks
with exact ground-truth labels for:
1. Musical Key (all 24 Major & Minor keys with precise harmonic overtone series)
2. BPM & Exact Beatgrid Downbeats
3. Structural Boundaries & HotCues (FIRST_BEAT, INTRO, BUILDUP, DROP_1, BREAKDOWN, DROP_2, OUTRO)
4. Dynamic Dancefloor Energy Curves (1.0 to 10.0 scale)
"""

import os
import sys
import numpy as np
import scipy.signal as signal

KEY_NAMES = [
    "C Major", "C# Major", "D Major", "D# Major", "E Major", "F Major",
    "F# Major", "G Major", "G# Major", "A Major", "A# Major", "B Major",
    "C Minor", "C# Minor", "D Minor", "D# Minor", "E Minor", "F Minor",
    "F# Minor", "G Minor", "G# Minor", "A Minor", "A# Minor", "B Minor"
]

# Root frequencies for Octave 3 (Hz)
ROOT_FREQS = {
    0: 130.81,   # C3
    1: 138.59,   # C#3
    2: 146.83,   # D3
    3: 155.56,   # D#3
    4: 164.81,   # E3
    5: 174.61,   # F3
    6: 185.00,   # F#3
    7: 196.00,   # G3
    8: 207.65,   # G#3
    9: 220.00,   # A3
    10: 233.08,  # A#3
    11: 246.94   # B3
}

# Semitone intervals for chords
CHORD_INTERVALS = {
    "Major": [0, 4, 7, 11, 12, 16],      # Root, Major 3rd, 5th, Major 7th, Octave, 10th
    "Minor": [0, 3, 7, 10, 12, 15]       # Root, Minor 3rd, 5th, Minor 7th, Octave, Minor 10th
}

class ProceduralEDMTrackGenerator:
    def __init__(self, sr=22050):
        self.sr = sr

    def _generate_kick(self, duration_sec=0.25):
        """Synthesizes a punchy 909-style analog kick drum with sub sweep and transient click"""
        t = np.linspace(0, duration_sec, int(self.sr * duration_sec), endpoint=False)
        # Exponential pitch envelope from 160Hz down to 45Hz
        freq_env = 45.0 + (160.0 - 45.0) * np.exp(-t * 35.0)
        phase = 2 * np.pi * np.cumsum(freq_env) / self.sr
        body = np.sin(phase)
        # Amplitude envelope
        amp_env = np.exp(-t * 12.0)
        # High transient click
        click = np.random.normal(0, 1, len(t)) * np.exp(-t * 120.0) * 0.4
        kick = (body * amp_env + click).astype(np.float32)
        return kick / (np.max(np.abs(kick)) + 1e-6)

    def _generate_hihat(self, duration_sec=0.10, open_hat=False):
        """Synthesizes an open or closed electronic hi-hat using filtered high-frequency noise"""
        dur = duration_sec if not open_hat else duration_sec * 2.5
        t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        noise = np.random.normal(0, 1, len(t))
        decay_rate = 45.0 if not open_hat else 15.0
        amp = np.exp(-t * decay_rate)
        # High-pass filter > 7000 Hz
        sos = signal.butter(4, 7000.0, btype='highpass', fs=self.sr, output='sos')
        filtered_hat = signal.sosfilt(sos, noise * amp)
        return filtered_hat.astype(np.float32)

    def _generate_saw_chord(self, root_hz, is_minor=False, duration_sec=2.0):
        """Synthesizes rich supersaw harmonic chords with overtone series in specified key"""
        t = np.linspace(0, duration_sec, int(self.sr * duration_sec), endpoint=False)
        intervals = CHORD_INTERVALS["Minor"] if is_minor else CHORD_INTERVALS["Major"]
        chord_audio = np.zeros_like(t, dtype=np.float32)

        for semi in intervals:
            note_hz = root_hz * (2.0 ** (semi / 12.0))
            # 5-voice detuned supersaw
            detunes = [-0.015, -0.007, 0.0, 0.007, 0.015]
            for d in detunes:
                f = note_hz * (1.0 + d)
                saw = 2.0 * (t * f - np.floor(t * f + 0.5))
                chord_audio += saw * 0.08

        # Smooth envelope
        attack = int(self.sr * 0.03)
        release = int(self.sr * 0.08)
        env = np.ones_like(t)
        if len(t) > attack + release:
            env[:attack] = np.linspace(0, 1, attack)
            env[-release:] = np.linspace(1, 0, release)
        
        return (chord_audio * env).astype(np.float32)

    def generate_track(self, key_id=None, bpm=None, bars=32):
        """
        Generates a complete multi-section synthetic EDM track:
        Structure: INTRO (4 bars) -> BUILDUP (4 bars) -> DROP 1 (8 bars) -> BREAKDOWN (4 bars) -> DROP 2 (8 bars) -> OUTRO (4 bars)
        Total: 32 bars (~64 seconds at 120-128 BPM).
        """
        if key_id is None:
            key_id = np.random.randint(0, 24)
        if bpm is None:
            bpm = float(np.random.choice([120.0, 124.0, 126.0, 128.0, 130.0, 132.0, 138.0, 140.0]))

        is_minor = (key_id >= 12)
        pitch_class = key_id % 12
        root_hz = ROOT_FREQS[pitch_class]

        sec_per_beat = 60.0 / bpm
        sec_per_bar = sec_per_beat * 4.0
        total_duration = sec_per_bar * bars
        total_samples = int(self.sr * total_duration)

        y = np.zeros(total_samples, dtype=np.float32)
        kick_sample = self._generate_kick()
        closed_hat = self._generate_hihat(open_hat=False)
        open_hat = self._generate_hihat(open_hat=True)

        # Structural Section Boundaries (in Bars)
        bar_intro_end = 4
        bar_buildup_end = 8
        bar_drop1_end = 16
        bar_break_end = 20
        bar_drop2_end = 28
        bar_outro_end = bars

        # Beat timestamps
        beat_timestamps = []
        cues = []
        energy_profile = np.zeros(total_samples, dtype=np.float32)

        for b in range(bars * 4):
            t_beat = b * sec_per_beat
            beat_timestamps.append(round(t_beat, 3))

        # HotCues Ground Truth
        cues.append({"cue_type": "FIRST_BEAT", "position_secs": 0.0, "hotcue_num": 1})
        cues.append({"cue_type": "BUILDUP", "position_secs": round(bar_intro_end * sec_per_bar, 3), "hotcue_num": 2})
        cues.append({"cue_type": "DROP_1", "position_secs": round(bar_buildup_end * sec_per_bar, 3), "hotcue_num": 3})
        cues.append({"cue_type": "BREAKDOWN", "position_secs": round(bar_drop1_end * sec_per_bar, 3), "hotcue_num": 4})
        cues.append({"cue_type": "DROP_2", "position_secs": round(bar_break_end * sec_per_bar, 3), "hotcue_num": 5})
        cues.append({"cue_type": "OUTRO", "position_secs": round(bar_drop2_end * sec_per_bar, 3), "hotcue_num": 6})

        # Assemble Audio & Energy Layer by Layer
        for bar in range(bars):
            t_bar_start = bar * sec_per_bar
            s_bar_start = int(t_bar_start * self.sr)

            # Determine Active Section
            if bar < bar_intro_end:
                sec_type = "INTRO"
                target_energy = 3.5
            elif bar < bar_buildup_end:
                sec_type = "BUILDUP"
                # Rising energy curve
                progress = (bar - bar_intro_end) / float(bar_buildup_end - bar_intro_end)
                target_energy = 4.0 + progress * 4.0
            elif bar < bar_drop1_end:
                sec_type = "DROP_1"
                target_energy = 9.0
            elif bar < bar_break_end:
                sec_type = "BREAKDOWN"
                target_energy = 4.0
            elif bar < bar_drop2_end:
                sec_type = "DROP_2"
                target_energy = 9.5
            else:
                sec_type = "OUTRO"
                target_energy = 3.0

            s_bar_end = min(total_samples, int((t_bar_start + sec_per_bar) * self.sr))
            energy_profile[s_bar_start:s_bar_end] = target_energy

            # 1. Chords & Harmonic Content
            chord = self._generate_saw_chord(root_hz, is_minor=is_minor, duration_sec=sec_per_bar)
            c_len = min(len(chord), total_samples - s_bar_start)
            if sec_type in ["INTRO", "BREAKDOWN"]:
                # Soft filtered chords
                y[s_bar_start:s_bar_start + c_len] += chord[:c_len] * 0.45
            elif sec_type in ["DROP_1", "DROP_2"]:
                # Powerful wide chords
                y[s_bar_start:s_bar_start + c_len] += chord[:c_len] * 0.85
            elif sec_type == "BUILDUP":
                # High-pass filter rising chord
                y[s_bar_start:s_bar_start + c_len] += chord[:c_len] * 0.55

            # 2. Drums & Rhythmic Grid
            for beat in range(4):
                s_beat = int((t_bar_start + beat * sec_per_beat) * self.sr)
                if s_beat >= total_samples:
                    continue

                # Kicks on every beat in Drops and Outro
                if sec_type in ["DROP_1", "DROP_2", "OUTRO", "INTRO"]:
                    k_len = min(len(kick_sample), total_samples - s_beat)
                    y[s_beat:s_beat + k_len] += kick_sample[:k_len] * 0.90

                # Offbeat open hi-hat (on the 'and' of each beat)
                if sec_type in ["DROP_1", "DROP_2"]:
                    s_offbeat = int((t_bar_start + (beat + 0.5) * sec_per_beat) * self.sr)
                    if s_offbeat < total_samples:
                        h_len = min(len(open_hat), total_samples - s_offbeat)
                        y[s_offbeat:s_offbeat + h_len] += open_hat[:h_len] * 0.40

                # Closed hats on 16th notes
                if sec_type in ["DROP_1", "DROP_2", "INTRO"]:
                    for step in [0.25, 0.75]:
                        s_16th = int((t_bar_start + (beat + step) * sec_per_beat) * self.sr)
                        if s_16th < total_samples:
                            c_len = min(len(closed_hat), total_samples - s_16th)
                            y[s_16th:s_16th + c_len] += closed_hat[:c_len] * 0.25

            # 3. Buildup Riser Snare Roll
            if sec_type == "BUILDUP":
                progress = (bar - bar_intro_end) / float(bar_buildup_end - bar_intro_end)
                roll_subdiv = 2 if progress < 0.5 else 4
                for sub in range(4 * roll_subdiv):
                    s_snare = int((t_bar_start + (sub / float(roll_subdiv)) * sec_per_beat) * self.sr)
                    if s_snare < total_samples:
                        sn_amp = 0.2 + progress * 0.6
                        sn_len = min(len(closed_hat), total_samples - s_snare)
                        y[s_snare:s_snare + sn_len] += closed_hat[:sn_len] * sn_amp

        # Master Limiting / Normalization
        max_peak = np.max(np.abs(y))
        if max_peak > 1e-6:
            y = (y / max_peak) * 0.95

        return {
            "audio": y.astype(np.float32),
            "sr": self.sr,
            "duration_sec": total_duration,
            "bpm": bpm,
            "key_id": key_id,
            "key_name": KEY_NAMES[key_id],
            "beat_timestamps": beat_timestamps,
            "cues": cues,
            "energy_profile": energy_profile
        }

if __name__ == "__main__":
    generator = ProceduralEDMTrackGenerator()
    track = generator.generate_track(key_id=0, bpm=128.0)
    print(f"[+] Successfully generated synthetic EDM track: {track['key_name']} @ {track['bpm']} BPM ({track['duration_sec']:.1f}s)")
    print(f"    HotCues generated: {len(track['cues'])} cues -> {[c['cue_type'] for c in track['cues']]}")

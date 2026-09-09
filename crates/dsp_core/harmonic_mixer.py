"""
AudioHarmonix — HarmonicMixer Engine
Section 4.5 & Roadmap Phase 2: Camelot 24-Position Harmonic Transition & Smart Track Recommendation Engine

Provides mathematical scoring and contextual classification for DJ transitions:
- Same Key (Harmonic Match, 1.0)
- Relative Key (Major/Minor, 0.98)
- Subdominant / Dominant (+1 / -1 Clockwise/Counter-Clockwise, 0.90 - 0.92)
- Energy Boost (+1 / +2 Camelot Steps, 0.78 - 0.85)
- Climax Semitone Jump (+1 Semitone modulation, 0.82)
- Diagonal Mix (Key Shift + Mode Change, 0.85)
- Parallel Scale Shift (Same root note, 0.75)
- Tempo & Pitch tolerance calculation with Half-Time / Double-Time detection
"""

import re
from typing import Dict, List, Optional, Tuple, Any

# Map Camelot to Semitone Index (0 to 11, relative to C)
# 1A = Abm / G#m (8), 1B = B (11), etc.
CAMELOT_TO_SEMITONE_OFFSET = {
    # Minor keys (A)
    "1A": 8,   # G#m
    "2A": 3,   # D#m
    "3A": 10,  # A#m / Bbm
    "4A": 5,   # Fm
    "5A": 0,   # Cm
    "6A": 7,   # Gm
    "7A": 2,   # Dm
    "8A": 9,   # Am
    "9A": 4,   # Em
    "10A": 11, # Bm
    "11A": 6,  # F#m
    "12A": 1,  # C#m
    # Major keys (B)
    "1B": 11,  # B
    "2B": 6,   # F#
    "3B": 1,   # C# / Db
    "4B": 8,   # G# / Ab
    "5B": 3,   # D# / Eb
    "6B": 10,  # A# / Bb
    "7B": 5,   # F
    "8B": 0,   # C
    "9B": 7,   # G
    "10B": 2,  # D
    "11B": 9,  # A
    "12B": 4   # E
}

class HarmonicMixer:
    """
    Intelligent DJ transition engine evaluating harmonic compatibility,
    tempo pitch stretch, and energy progression across library tracks.
    """

    @staticmethod
    def parse_camelot(camelot_str: str) -> Optional[Tuple[int, str]]:
        """Parses a Camelot string like '8A' or '11b' into (number, letter)."""
        if not camelot_str:
            return None
        m = re.match(r"^(\d{1,2})([a-bA-B])$", str(camelot_str).strip())
        if not m:
            return None
        num = int(m.group(1))
        letter = m.group(2).upper()
        if 1 <= num <= 12 and letter in ("A", "B"):
            return (num, letter)
        return None

    @classmethod
    def evaluate_harmonic_transition(cls, key_from: str, key_to: str) -> Dict[str, Any]:
        """
        Evaluates the harmonic relationship between two Camelot keys.
        Returns a dictionary containing score (0.0 to 1.0), transition category,
        and user-facing label.
        """
        parsed_from = cls.parse_camelot(key_from)
        parsed_to = cls.parse_camelot(key_to)

        if not parsed_from or not parsed_to:
            return {
                "score": 0.30,
                "type": "UNKNOWN",
                "label": "Unknown Key",
                "description": "Unrecognized musical key",
                "color": "#64748b"
            }

        num1, let1 = parsed_from
        num2, let2 = parsed_to

        # 1. SAME KEY (Harmonic Match)
        if num1 == num2 and let1 == let2:
            return {
                "score": 1.0,
                "type": "SAME_KEY",
                "label": "Harmonic Match",
                "description": "Perfect tonal match in identical key",
                "color": "#10b981" # Green
            }

        # 2. RELATIVE KEY (Same number, different mode: A <-> B)
        if num1 == num2 and let1 != let2:
            rel_type = "Relative Major" if let2 == "B" else "Relative Minor"
            return {
                "score": 0.98,
                "type": "RELATIVE_KEY",
                "label": f"Relative Key ({rel_type})",
                "description": "Seamless mood shift sharing the exact same scale notes",
                "color": "#06b6d4" # Cyan
            }

        # Calculate clock distances on 12-hour wheel
        step_forward = (num2 - num1) % 12
        step_backward = (num1 - num2) % 12

        # 3. SUBDOMINANT / DOMINANT (+1 / -1 in same mode)
        if let1 == let2:
            if step_forward == 1:
                return {
                    "score": 0.92,
                    "type": "ENERGY_LIFT_1",
                    "label": "+1 Energy Lift",
                    "description": "Smooth 1-step energy elevation (Subdominant/Dominant)",
                    "color": "#38bdf8" # Sky blue
                }
            elif step_backward == 1:
                return {
                    "score": 0.90,
                    "type": "CHILL_DROP_1",
                    "label": "-1 Deep Chill",
                    "description": "Gentle 1-step release of tension (Subdominant/Dominant)",
                    "color": "#818cf8" # Indigo
                }
            elif step_forward == 2:
                return {
                    "score": 0.80,
                    "type": "ENERGY_BOOST_2",
                    "label": "+2 Energy Boost",
                    "description": "Noticeable positive energy surge on the dancefloor",
                    "color": "#f59e0b" # Amber
                }
            elif step_backward == 2:
                return {
                    "score": 0.75,
                    "type": "ENERGY_DROP_2",
                    "label": "-2 Energy Drop",
                    "description": "Significant energy reset for breakdowns or warm-down",
                    "color": "#a855f7" # Purple
                }

        # 4. CLIMAX SEMITONE JUMP (+1 Semitone modulation = +7 Camelot steps in same mode)
        # e.g. 8A (Am) -> 3A (Bbm): (8 + 7) % 12 = 3.
        # e.g. 1B (B) -> 8B (C): (1 + 7) % 12 = 8.
        if let1 == let2 and step_forward == 7:
            return {
                "score": 0.82,
                "type": "SEMITONE_JUMP",
                "label": "Climax (+1 Semitone)",
                "description": "Dramatic half-step upward modulation for peak euphoria",
                "color": "#ec4899" # Pink / Magenta
            }

        # 5. DIAGONAL MIX (+1 or -1 Step with Mode Change)
        if let1 != let2:
            if step_forward == 1 or step_backward == 1:
                return {
                    "score": 0.85,
                    "type": "DIAGONAL_MIX",
                    "label": "Diagonal Mix",
                    "description": "Expressive transition modulating both key and mode",
                    "color": "#14b8a6" # Teal
                }

        # 6. PARALLEL SCALE SHIFT (Same tonic root note, changing Major/Minor)
        # e.g. Am (8A) to A (11B): offset difference is 3 steps
        if let1 != let2:
            if (let1 == "A" and let2 == "B" and step_forward == 3) or \
               (let1 == "B" and let2 == "A" and step_backward == 3):
                return {
                    "score": 0.78,
                    "type": "PARALLEL_SHIFT",
                    "label": "Parallel Scale Shift",
                    "description": "Transitions between major and minor around the same root note",
                    "color": "#eab308" # Yellow
                }

        # 7. DISTANT / HARMONIC CLASH
        return {
            "score": 0.25,
            "type": "DISCORDANT",
            "label": "Tone Clash (Distant Key)",
            "description": "Incompatible scale notes; use an echo freeze or drum-only transition",
            "color": "#ef4444" # Red
        }

    @staticmethod
    def evaluate_tempo_compatibility(bpm_from: float, bpm_to: float) -> Dict[str, Any]:
        """
        Calculates pitch differential percentage on standard DJ equipment.
        Handles direct BPM as well as Half-Time (x0.5) and Double-Time (x2.0).
        """
        if not bpm_from or bpm_from <= 0 or not bpm_to or bpm_to <= 0:
            return {
                "score": 0.50,
                "pitch_pct": 0.0,
                "pitch_display": "0.0%",
                "is_half_time": False,
                "is_double_time": False,
                "effective_target_bpm": bpm_to or 120.0
            }

        # Direct comparison
        diff_direct = bpm_to - bpm_from
        pitch_direct = (diff_direct / bpm_from) * 100.0

        # Half-time comparison (e.g. 140 to 70)
        diff_half = (bpm_to * 2.0) - bpm_from
        pitch_half = (diff_half / bpm_from) * 100.0

        # Double-time comparison (e.g. 70 to 140)
        diff_double = (bpm_to / 2.0) - bpm_from
        pitch_double = (diff_double / bpm_from) * 100.0

        candidates = [
            (abs(pitch_direct), pitch_direct, False, False, bpm_to),
            (abs(pitch_half), pitch_half, True, False, bpm_to * 2.0),
            (abs(pitch_double), pitch_double, False, True, bpm_to / 2.0)
        ]
        candidates.sort(key=lambda x: x[0])
        best_abs_pitch, best_pitch, is_half, is_double, effective_bpm = candidates[0]

        # Score based on CDJ pitch limits
        # <= 2%: 1.0 (Flawless)
        # <= 4%: 0.90 (Great)
        # <= 6%: 0.75 (Good, typical range)
        # <= 8%: 0.55 (Noticeable stretch)
        # > 8%: decaying
        if best_abs_pitch <= 2.0:
            bpm_score = 1.0 - (best_abs_pitch / 20.0)
        elif best_abs_pitch <= 4.0:
            bpm_score = 0.90 - ((best_abs_pitch - 2.0) / 20.0)
        elif best_abs_pitch <= 6.0:
            bpm_score = 0.80 - ((best_abs_pitch - 4.0) / 15.0)
        elif best_abs_pitch <= 8.0:
            bpm_score = 0.65 - ((best_abs_pitch - 6.0) / 10.0)
        else:
            bpm_score = max(0.10, 0.45 - ((best_abs_pitch - 8.0) / 12.0))

        sign = "+" if best_pitch > 0 else ""
        pitch_formatted = f"{sign}{best_pitch:.1f}%"
        if is_half:
            pitch_formatted += " (Half-Time)"
        elif is_double:
            pitch_formatted += " (Double-Time)"

        return {
            "score": round(bpm_score, 3),
            "pitch_pct": round(best_pitch, 2),
            "pitch_display": pitch_formatted,
            "is_half_time": is_half,
            "is_double_time": is_double,
            "effective_target_bpm": round(effective_bpm, 1)
        }

    @staticmethod
    def evaluate_energy_progression(energy_from: float, energy_to: float) -> Dict[str, Any]:
        """Evaluates energy delta between current and target track (1-10 scale)."""
        e_from = float(energy_from or 5.0)
        e_to = float(energy_to or 5.0)
        delta = e_to - e_from

        if delta >= 1.5:
            label = f"Rising Energy (+{delta:.1f})"
            trend = "RISE"
            score = 0.90
        elif delta <= -1.5:
            label = f"Cool Down ({delta:.1f})"
            trend = "DROP"
            score = 0.85
        else:
            label = f"Steady Energy ({'+' if delta >= 0 else ''}{delta:.1f})"
            trend = "STEADY"
            score = 1.0

        return {
            "score": score,
            "delta": round(delta, 1),
            "label": label,
            "trend": trend
        }

    @classmethod
    def rank_candidates(cls, source_track: Dict[str, Any], library_tracks: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
        """
        Ranks library tracks by mixability score relative to source_track.
        Weights:
        - 55% Harmonic Compatibility
        - 35% Tempo / Pitch Tolerance
        - 10% Energy Flow
        """
        src_id = source_track.get("id") or source_track.get("track_id")
        src_key = source_track.get("camelot_key", "")
        src_bpm = float(source_track.get("bpm") or 120.0)
        src_energy = float(source_track.get("energy_score") or 5.0)

        results = []

        for candidate in library_tracks:
            cand_id = candidate.get("id") or candidate.get("track_id")
            if src_id and cand_id == src_id:
                continue

            cand_key = candidate.get("camelot_key", "")
            cand_bpm = float(candidate.get("bpm") or 120.0)
            cand_energy = float(candidate.get("energy_score") or 5.0)

            # 1. Harmonic evaluation
            harm_res = cls.evaluate_harmonic_transition(src_key, cand_key)

            # 2. Tempo evaluation
            tempo_res = cls.evaluate_tempo_compatibility(src_bpm, cand_bpm)

            # 3. Energy evaluation
            energy_res = cls.evaluate_energy_progression(src_energy, cand_energy)

            # Composite Score (0 to 100)
            composite_score = (
                (0.55 * harm_res["score"]) +
                (0.35 * tempo_res["score"]) +
                (0.10 * energy_res["score"])
            ) * 100.0

            match_pct = int(round(composite_score))

            results.append({
                "track": candidate,
                "match_score": match_pct,
                "harmonic": harm_res,
                "tempo": tempo_res,
                "energy": energy_res,
                "recommended_reason": f"{harm_res['label']} • {tempo_res['pitch_display']}"
            })

        # Sort descending by match_score, then by harmonic score
        results.sort(key=lambda x: (x["match_score"], x["harmonic"]["score"]), reverse=True)

        return results[:limit]

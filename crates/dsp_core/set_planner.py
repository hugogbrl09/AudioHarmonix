"""
AudioHarmonix Intelligent Set Planner & Energy Curve Optimizer
Roadmap Phase 4: Graph-Based Harmonic Route Optimization & Narrative Energy Arcs
"""

try:
    from harmonic_mixer import HarmonicMixer
except ImportError:
    from crates.dsp_core.harmonic_mixer import HarmonicMixer


class SetPlanner:
    """
    Plans and optimizes a multi-track DJ setlist by combining:
    1. Harmonic compatibility on the 24-position Camelot Wheel (via HarmonicMixer)
    2. Tempo & Pitch locking (minimizing BPM jumps)
    3. Narrative Energy Arc adherence (Warm-Up -> Peak -> Reset, Festival Peak, Deep Hypnotic, Progressive)
    """

    PROFILES = {
        "warmup_peak_reset": {
            "id": "warmup_peak_reset",
            "name": "Warm-Up -> Peak -> Reset",
            "description": "Classic club narrative arc: smooth opening (4.0), building to peak time (9.0) at 70% of the set, and a controlled closing reset (5.5)."
        },
        "festival_peak": {
            "id": "festival_peak",
            "name": "Festival Peak Time",
            "description": "High-octane mainstage energy (8.0-9.5) prioritizing +1/+2 Camelot energy lifts and climax modulations."
        },
        "deep_hypnotic": {
            "id": "deep_hypnotic",
            "name": "Continuous Deep / Hypnotic",
            "description": "Seamless, hypnotic groove (5.0-6.5) locking BPM and favoring pure Same-Key (8A->8A) and Relative (8A<->8B) blends."
        },
        "progressive_journey": {
            "id": "progressive_journey",
            "name": "Progressive Ascending",
            "description": "Steady, continuous upward climb in energy (3.5 -> 9.5) and tempo from opening track to finale."
        }
    }

    @classmethod
    def get_profiles(cls):
        return list(cls.PROFILES.values())

    @classmethod
    def compute_target_energy_curve(cls, profile_id: str, num_slots: int) -> list:
        """
        Generates the target energy level (1.0 to 10.0) for each slot in the setlist.
        """
        n = max(1, int(num_slots))
        if n == 1:
            if profile_id == "festival_peak":
                return [9.0]
            if profile_id == "deep_hypnotic":
                return [5.5]
            return [6.5]

        curve = []
        for i in range(n):
            p = i / float(n - 1)  # Normalized position in set [0.0 .. 1.0]
            if profile_id == "festival_peak":
                # 8.0 -> 9.6 at p=0.65 -> 8.8 at p=1.0
                if p <= 0.65:
                    val = 8.0 + (p / 0.65) * 1.6
                else:
                    val = 9.6 - ((p - 0.65) / 0.35) * 0.8
            elif profile_id == "deep_hypnotic":
                # Smooth hypnotic wave: 5.0 -> 6.5 at p=0.5 -> 5.5 at p=1.0
                if p <= 0.5:
                    val = 5.0 + (p / 0.5) * 1.5
                else:
                    val = 6.5 - ((p - 0.5) / 0.5) * 1.0
            elif profile_id == "progressive_journey":
                # Continuous linear climb: 3.5 -> 9.5
                val = 3.5 + p * 6.0
            else:
                # Default: warmup_peak_reset
                # 4.0 -> 9.0 at p=0.70 -> 5.5 at p=1.0
                if p <= 0.70:
                    val = 4.0 + (p / 0.70) * 5.0
                else:
                    val = 9.0 - ((p - 0.70) / 0.30) * 3.5

            curve.append(round(max(1.0, min(10.0, val)), 2))

        return curve

    @classmethod
    def _step_utility(cls, curr_track: dict, next_track: dict, target_energy: float, profile_id: str) -> tuple:
        """
        Computes combined step utility (0..100) and transition details between curr_track and next_track
        given the slot's target energy and active profile.
        """
        trans = HarmonicMixer.evaluate_transition(curr_track, next_track)
        h_type = trans["harmonic"]["type"]
        h_score = trans["harmonic"]["score"]
        t_score = trans["tempo"]["score"]

        cand_energy = float(next_track.get("energy_score") or 5.0)
        energy_diff = abs(cand_energy - target_energy)
        energy_adherence = max(0.0, 1.0 - (energy_diff / 7.5))

        # Profile-specific weighting & bonuses
        bonus = 0.0
        if profile_id == "deep_hypnotic":
            if h_type in ("SAME_KEY", "RELATIVE_KEY"):
                bonus += 0.08
            if abs(trans["tempo"]["pitch_pct"]) <= 1.5:
                bonus += 0.04
            combined = 0.52 * h_score + 0.28 * t_score + 0.20 * energy_adherence + bonus
        elif profile_id == "festival_peak":
            if h_type in ("ENERGY_LIFT_1", "ENERGY_BOOST_2", "SEMITONE_JUMP", "SAME_KEY"):
                bonus += 0.06
            combined = 0.48 * h_score + 0.22 * t_score + 0.30 * energy_adherence + bonus
        else:
            combined = 0.50 * h_score + 0.24 * t_score + 0.26 * energy_adherence + bonus

        # Heavy penalty for discordant key clashes so the planner avoids them whenever possible
        if h_type == "DISCORDANT":
            combined *= 0.45

        return round(min(100.0, max(0.0, combined * 100.0)), 2), trans

    @classmethod
    def _build_greedy_route(cls, start_track: dict, pool: list, target_curve: list, profile_id: str) -> list:
        """Constructs a route starting from start_track of length len(target_curve)."""
        route = [start_track]
        used_ids = {start_track.get("id")}
        n = len(target_curve)

        for slot_idx in range(1, n):
            curr = route[-1]
            target_e = target_curve[slot_idx]
            best_cand = None
            best_util = -1.0

            for cand in pool:
                cid = cand.get("id")
                if cid in used_ids:
                    continue
                util, _ = cls._step_utility(curr, cand, target_e, profile_id)
                if util > best_util:
                    best_util = util
                    best_cand = cand

            if best_cand is None:
                break
            route.append(best_cand)
            used_ids.add(best_cand.get("id"))

        return route

    @classmethod
    def _route_total_objective(cls, route: list, target_curve: list, profile_id: str) -> float:
        if not route:
            return 0.0
        first_e = float(route[0].get("energy_score") or 5.0)
        start_adherence = max(0.0, 1.0 - abs(first_e - target_curve[0]) / 7.5) * 100.0
        if len(route) == 1:
            return start_adherence

        total = start_adherence * 0.5
        for i in range(1, len(route)):
            util, _ = cls._step_utility(route[i - 1], route[i], target_curve[i], profile_id)
            total += util
        return total

    @classmethod
    def _optimize_2opt(cls, route: list, target_curve: list, profile_id: str, lock_start: bool) -> list:
        """
        Applies local swap optimization to eliminate any remaining tonal friction and improve curve fit.
        """
        best_route = list(route)
        best_obj = cls._route_total_objective(best_route, target_curve, profile_id)
        n = len(best_route)
        start_idx = 1 if lock_start else 0

        improved = True
        passes = 0
        while improved and passes < 3:
            improved = False
            passes += 1
            for i in range(start_idx, n - 1):
                for j in range(i + 1, n):
                    candidate = list(best_route)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    cand_obj = cls._route_total_objective(candidate, target_curve, profile_id)
                    if cand_obj > best_obj + 0.05:
                        best_route = candidate
                        best_obj = cand_obj
                        improved = True

        return best_route

    @classmethod
    def plan_set(cls, tracks: list, profile: str = "warmup_peak_reset", max_tracks: int = 12, start_track_id: str = None) -> dict:
        """
        Computes an optimal DJ setlist for the given tracks and energy profile.
        """
        if profile not in cls.PROFILES:
            profile = "warmup_peak_reset"
        profile_meta = cls.PROFILES[profile]

        valid_tracks = [t for t in (tracks or []) if t and t.get("id")]
        if not valid_tracks:
            return {
                "profile": profile_meta,
                "target_energy_curve": [],
                "actual_energy_curve": [],
                "bpm_curve": [],
                "setlist": [],
                "summary": {
                    "total_tracks": 0,
                    "total_duration_secs": 0.0,
                    "avg_transition_score": 0,
                    "harmonic_clashes": 0,
                    "energy_adherence_pct": 0
                }
            }

        n = min(len(valid_tracks), max(1, int(max_tracks or 12)))
        target_curve = cls.compute_target_energy_curve(profile, n)

        # Determine starting track candidates
        locked_start = None
        if start_track_id:
            for t in valid_tracks:
                if str(t.get("id")) == str(start_track_id):
                    locked_start = t
                    break

        if locked_start:
            start_candidates = [locked_start]
        else:
            # Rank tracks by proximity to opening target energy
            sorted_by_open = sorted(
                valid_tracks,
                key=lambda t: abs(float(t.get("energy_score") or 5.0) - target_curve[0])
            )
            start_candidates = sorted_by_open[:min(6, len(sorted_by_open))]

        best_route = None
        best_obj = -1.0
        for st in start_candidates:
            initial_route = cls._build_greedy_route(st, valid_tracks, target_curve, profile)
            refined_route = cls._optimize_2opt(
                initial_route, target_curve, profile, lock_start=(locked_start is not None)
            )
            obj = cls._route_total_objective(refined_route, target_curve, profile)
            if obj > best_obj:
                best_obj = obj
                best_route = refined_route

        # Format output setlist and transition metrics
        setlist = []
        actual_energy_curve = []
        bpm_curve = []
        transition_scores = []
        clashes = 0
        total_duration = 0.0
        energy_errors = []

        for idx, tr in enumerate(best_route):
            t_energy = float(tr.get("energy_score") or 5.0)
            t_bpm = round(float(tr.get("bpm") or 120.0), 2)
            t_dur = float(tr.get("duration_secs") or 0.0)
            target_e = target_curve[idx]

            actual_energy_curve.append(t_energy)
            bpm_curve.append(t_bpm)
            total_duration += t_dur
            energy_errors.append(abs(t_energy - target_e))

            trans_info = None
            if idx > 0:
                prev_tr = best_route[idx - 1]
                trans_info = HarmonicMixer.evaluate_transition(prev_tr, tr)
                transition_scores.append(trans_info["match_score"])
                if trans_info["harmonic"]["type"] == "DISCORDANT":
                    clashes += 1

            t_label = "OPENER"
            t_score = 100.0
            if trans_info:
                t_label = trans_info["harmonic"]["label"]
                t_score = trans_info["match_score"]

            setlist.append({
                "step": idx + 1,
                "track_id": tr.get("id"),
                "title": tr.get("title") or tr.get("file_name", "Unknown"),
                "artist": tr.get("artist", "Unknown Artist"),
                "camelot_key": tr.get("camelot_key", "8A"),
                "bpm": t_bpm,
                "actual_energy": t_energy,
                "target_energy": target_e,
                "transition_label": t_label,
                "transition_score": t_score,
                "track": {
                    "id": tr.get("id"),
                    "title": tr.get("title") or tr.get("file_name", "Unknown"),
                    "artist": tr.get("artist", "Unknown Artist"),
                    "file_name": tr.get("file_name", ""),
                    "file_path": tr.get("file_path", ""),
                    "camelot_key": tr.get("camelot_key", "8A"),
                    "detected_key": tr.get("detected_key", "A Minor"),
                    "bpm": t_bpm,
                    "energy_score": t_energy,
                    "duration_secs": round(t_dur, 1),
                    "has_artwork": bool(tr.get("has_artwork", False)),
                    "cues": tr.get("cues", [])
                },
                "transition_from_prev": trans_info
            })

        avg_trans = round(sum(transition_scores) / len(transition_scores), 1) if transition_scores else 100.0
        avg_e_err = sum(energy_errors) / len(energy_errors) if energy_errors else 0.0
        energy_adherence_pct = round(max(0.0, min(100.0, (1.0 - avg_e_err / 7.5) * 100.0)), 1)
        harm_compat_pct = round(((max(1, len(transition_scores)) - clashes) / max(1, len(transition_scores))) * 100.0, 1) if transition_scores else 100.0
        bpm_range = [min(bpm_curve), max(bpm_curve)] if bpm_curve else [0.0, 0.0]

        summary_obj = {
            "total_tracks": len(setlist),
            "total_duration_secs": round(total_duration, 1),
            "total_duration_mins": round(total_duration / 60.0, 1),
            "avg_transition_score": avg_trans,
            "harmonic_compatibility_pct": harm_compat_pct,
            "harmonic_clashes": clashes,
            "energy_adherence_pct": energy_adherence_pct,
            "bpm_range": bpm_range,
        }

        return {
            "profile": profile_meta,
            "profile_id": profile,
            "target_energy_curve": target_curve,
            "target_curve": target_curve,
            "actual_energy_curve": actual_energy_curve,
            "actual_curve": actual_energy_curve,
            "bpm_curve": bpm_curve,
            "setlist": setlist,
            "summary": summary_obj,
            "curve_summary": summary_obj,
        }


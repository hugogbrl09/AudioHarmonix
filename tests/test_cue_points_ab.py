"""
AudioHarmonix A/B Cue Detection Audit Test
Compares Legacy DSP vs Experimental 3-Layer DSP on real audio files.
Audits:
- Common events (Legacy & Experimental)
- New events (Buildups, Breaks with heavy sub-bass)
- Lost events (regressions)
- Temporal Delta (time alignment)
- Feature diagnostics (TransientEnergy, BeatPulse, Slope, FluxAcceleration, BreakScore, BuildupScore)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "crates", "audio_decoder")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "crates", "dsp_core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "crates", "ml_engine")))

import decoder
import dsp
import ml

TEST_TRACKS = [
    ("Tech House (Soma)", "sample_tracks/Return Of The Jaded - Soma (Extended Mix).mp3"),
    ("Melodic House (Fire Desire)", "sample_tracks/Fire Desire (Original Mix) - RÜFÜS DU SOL.mp3"),
    ("EDM / Electro (blaster)", "sample_tracks/blaster.mp3"),
    ("Drum & Bass (1052744)", "dataset/giantsteps-key-dataset-master/audio/1052744.LOFI.mp3"),
    ("Techno (1026478)", "dataset/giantsteps-key-dataset-master/audio/1026478.LOFI.mp3")
]

def run_ab_comparison(track_label, fpath):
    if not os.path.exists(fpath):
        return None

    y, sr, dur = decoder.load_and_resample(fpath)
    bpm, conf, beats, _ = dsp.estimate_bpm_and_beatgrid(y, sr=sr)

    # Legacy cues
    legacy_cues = dsp.detect_cue_points_legacy(y, beats, dur, sr=sr)

    # Experimental cues + diagnostics
    exp_cues, diag = dsp.detect_cue_points_experimental(y, beats, dur, sr=sr, return_diagnostics=True)

    # Compare events
    common_events = []
    new_events = []
    lost_events = []

    exp_matched = set()
    for l_cue in legacy_cues:
        l_pos = l_cue["position_secs"]
        l_type = l_cue["cue_type"].split("_")[0] # DROP, BREAK, OUTRO, FIRST

        # Find closest match in experimental within 4.0s
        best_match = None
        best_diff = 999.0
        best_idx = -1

        for i, e_cue in enumerate(exp_cues):
            e_pos = e_cue["position_secs"]
            e_type = e_cue["cue_type"].split("_")[0]
            diff = abs(e_pos - l_pos)

            if diff < 4.0 and (e_type == l_type or (e_type in ["DROP", "BUILDUP"] and l_type == "DROP")):
                if diff < best_diff:
                    best_diff = diff
                    best_match = e_cue
                    best_idx = i

        if best_match is not None:
            exp_matched.add(best_idx)
            common_events.append({
                "legacy_type": l_cue["cue_type"],
                "legacy_pos": l_pos,
                "exp_type": best_match["cue_type"],
                "exp_pos": best_match["position_secs"],
                "delta_sec": round(best_match["position_secs"] - l_pos, 3)
            })
        else:
            lost_events.append(l_cue)

    for i, e_cue in enumerate(exp_cues):
        if i not in exp_matched:
            new_events.append(e_cue)

    return {
        "label": track_label,
        "duration": dur,
        "bpm": bpm,
        "legacy_cues": legacy_cues,
        "exp_cues": exp_cues,
        "common_events": common_events,
        "new_events": new_events,
        "lost_events": lost_events,
        "evidence": diag["evidence"],
        "candidates": diag["candidates"]
    }


def test_ab_comparison_all_tracks():
    print("\n" + "=" * 90)
    print("           AUDIOHARMONIX DSP A/B BENCHMARK AUDIT (LEGACY VS EXPERIMENTAL)")
    print("=" * 90)

    for label, path in TEST_TRACKS:
        res = run_ab_comparison(label, path)
        if res is None:
            continue

        print(f"\n>>> FAIXA: {res['label']} (BPM: {res['bpm']:.1f}, Duracao: {res['duration']:.1f}s)")
        print("-" * 90)
        print("  LEGACY CUES      : " + ", ".join([f"{c['cue_type']}@{c['position_secs']}s" for c in res["legacy_cues"]]))
        print("  EXPERIMENTAL CUES: " + ", ".join([f"{c['cue_type']}@{c['position_secs']}s" for c in res["exp_cues"]]))
        
        print("\n  [OK] EVENTOS COMUNS:")
        for ev in res["common_events"]:
            delta_str = f"{ev['delta_sec']:+.2f}s" if ev["delta_sec"] != 0 else "0.00s (Exato)"
            print(f"      * {ev['legacy_type']} ({ev['legacy_pos']}s) <---> {ev['exp_type']} ({ev['exp_pos']}s) | Delta = {delta_str}")

        print("\n  [+] NOVOS EVENTOS (Detectados apenas pelo Experimental):")
        if res["new_events"]:
            for ne in res["new_events"]:
                print(f"      * {ne['cue_type']} @ {ne['position_secs']}s")
        else:
            print("      (Nenhum novo evento)")

        print("\n  [-] EVENTOS PERDIDOS (Regressoes em relacao ao Legacy):")
        if res["lost_events"]:
            for le in res["lost_events"]:
                print(f"      * {le['cue_type']} @ {le['position_secs']}s")
        else:
            print("      (Zero regressoes - todos os eventos legacy foram mantidos/refinados)")

        print("-" * 90)

    # -----------------------------------------------------------------------
    # SPECIFIC REGRESSION TEST GATES (PROMPT3.MD MANDATES)
    # -----------------------------------------------------------------------
    # 1. Soma: nao gerar BREAK na regiao de 15.6s, manter BREAK em 124.8s, manter BUILDUP em 173.1s
    res_soma = run_ab_comparison("Soma", "sample_tracks/Return Of The Jaded - Soma (Extended Mix).mp3")
    if res_soma:
        soma_exp_cues = res_soma["exp_cues"]
        soma_cue_types = [c["cue_type"] for c in soma_exp_cues]
        soma_cue_positions = [c["position_secs"] for c in soma_exp_cues]
        
        # Must NOT generate BREAK in 15.6s region
        assert not any(c["cue_type"].startswith("BREAK") and abs(c["position_secs"] - 15.6) < 5.0 for c in soma_exp_cues), "Regression: False BREAK at 15.6s detected in Soma!"
        # Must maintain BREAK at 124.8s
        assert any(c["cue_type"].startswith("BREAK") and abs(c["position_secs"] - 124.8) < 3.0 for c in soma_exp_cues), "Regression: Missing real BREAK at 124.8s in Soma!"
        # Must maintain BUILDUP at 173.1s
        assert any(c["cue_type"].startswith("BUILDUP") and abs(c["position_secs"] - 173.1) < 3.0 for c in soma_exp_cues), "Regression: Missing BUILDUP at 173.1s in Soma!"
        # Must maintain DROP at 188.7s
        assert any(c["cue_type"].startswith("DROP") and abs(c["position_secs"] - 188.7) < 3.0 for c in soma_exp_cues), "Regression: Missing DROP at 188.7s in Soma!"

    # 2. Fire Desire: manter Drops em 53.8s e 222.7s
    res_fd = run_ab_comparison("Fire Desire", "sample_tracks/Fire Desire (Original Mix) - RÜFÜS DU SOL.mp3")
    if res_fd:
        fd_exp_cues = res_fd["exp_cues"]
        assert any(c["cue_type"].startswith("DROP") and abs(c["position_secs"] - 53.8) < 2.0 for c in fd_exp_cues), "Regression: Missing Drop 1 in Fire Desire!"
        assert any(c["cue_type"].startswith("DROP") and abs(c["position_secs"] - 222.7) < 2.0 for c in fd_exp_cues), "Regression: Missing Drop 2 in Fire Desire!"

    # 3. blaster: manter os tres Drops e o Buildup
    res_blaster = run_ab_comparison("blaster", "sample_tracks/blaster.mp3")
    if res_blaster:
        blaster_cues = res_blaster["exp_cues"]
        assert any(c["cue_type"].startswith("DROP") and abs(c["position_secs"] - 32.2) < 2.0 for c in blaster_cues), "Regression: Missing Drop 1 in blaster!"
        assert any(c["cue_type"].startswith("DROP") and abs(c["position_secs"] - 94.1) < 2.0 for c in blaster_cues), "Regression: Missing Drop 2 in blaster!"
        assert any(c["cue_type"].startswith("DROP") and abs(c["position_secs"] - 201.9) < 2.0 for c in blaster_cues), "Regression: Missing Drop 3 in blaster!"
        assert any(c["cue_type"].startswith("BUILDUP") and abs(c["position_secs"] - 194.1) < 2.0 for c in blaster_cues), "Regression: Missing Buildup in blaster!"

    # 4. Default mode is strictly legacy
    default_cues = dsp.detect_cue_points(np.zeros(22050 * 5), [0.0, 0.5, 1.0, 1.5, 2.0], 5.0)
    assert default_cues[0]["cue_type"] == "FIRST_BEAT"
    print("\n[ALL REGRESSION GATES PASSED 100% SUCCESSFULLY]")


if __name__ == "__main__":
    test_ab_comparison_all_tracks()

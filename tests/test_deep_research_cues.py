import os
import glob
import urllib.request
import json

API_BASE = "http://127.0.0.1:8888"

def test_deep_research_cue_engine():
    # 1. Fetch tracks
    req = urllib.request.urlopen(f"{API_BASE}/api/tracks")
    assert req.getcode() == 200
    data = json.loads(req.read().decode("utf-8"))
    tracks = data.get("tracks", [])
    assert len(tracks) > 0, "No tracks found in database!"

    for t in tracks:
        fname = t["file_name"]
        dur = t["duration_secs"]
        cues = t["cues"]
        assert len(cues) > 0, f"Track {fname} has no cues!"

        # Rule 1: FIRST_BEAT must be in range [0.0s, 1.5s]
        first_beat = next((c for c in cues if c["cue_type"] == "FIRST_BEAT"), None)
        assert first_beat is not None, f"Track {fname} missing FIRST_BEAT cue!"
        assert 0.0 <= first_beat["position_secs"] <= 1.5, f"Track {fname} FIRST_BEAT ({first_beat['position_secs']}s) is not at start of track!"

        # Rule 2: All HotCues must be within physical duration bounds
        for c in cues:
            assert c["position_secs"] < dur, f"Track {fname} Cue {c['cue_type']} ({c['position_secs']}s) exceeds duration ({dur}s)!"

        # Rule 3: Strict chronological ordering
        cue_times = [c["position_secs"] for c in cues]
        assert cue_times == sorted(cue_times), f"Track {fname} HotCues are not chronologically sorted: {cue_times}"

        # Rule 4: Clean sequential numbering 1..N
        cue_nums = [c["hotcue_num"] for c in cues]
        assert cue_nums == list(range(1, len(cues) + 1)), f"Track {fname} hotcue_num not 1..N: {cue_nums}"

        print(f"VERIFIED {fname} (Dur: {dur:.1f}s): {[(c['cue_type'], c['position_secs']) for c in cues]}")

if __name__ == "__main__":
    test_deep_research_cue_engine()

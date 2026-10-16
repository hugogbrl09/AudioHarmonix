"""
AudioHarmonix Tag Writer & Multi-Platform DJ Software Exporter
Section 8, 11 & Roadmap Phase 3:
- Writes ID3v2.4 Tags (TKEY, TBPM, COMM) & Serato Markers2 binary GEOB frames
- Exports Pioneer Rekordbox XML (with RGB section colors & Camelot Smart Crates)
- Exports Native Instruments Traktor Pro NML (with <CUE_V2> millisecond markers)
- Exports Serato DJ Pro binary .crate files (UTF-16BE vrsn/otrk/ptrk structure)
"""

import os
import struct
import base64
import xml.etree.ElementTree as ET
from xml.dom import minidom
import mutagen
from mutagen.id3 import ID3, TKEY, TBPM, COMM, GEOB, ID3NoHeaderError


# Official Pioneer CDJ / Rekordbox / Serato RGB Pad Colors by Musical Section
CUE_SECTION_COLORS = {
    "INTRO": (40, 226, 20),       # Green
    "FIRST_BEAT": (40, 226, 20),  # Green
    "DROP": (230, 30, 30),        # Red
    "DROP_1": (230, 30, 30),      # Red
    "DROP_2": (255, 60, 20),      # Crimson / Orange-Red
    "BREAK": (0, 114, 255),       # Blue
    "BREAKDOWN": (0, 114, 255),   # Blue
    "BREAK_1": (0, 114, 255),     # Blue
    "BREAK_2": (30, 144, 255),    # Dodger Blue
    "BUILDUP": (255, 190, 0),     # Gold / Yellow
    "BUILD": (255, 190, 0),       # Gold / Yellow
    "BUILD_1": (255, 190, 0),     # Gold / Yellow
    "OUTRO": (0, 200, 180),       # Cyan / Teal
    "VERSE": (160, 60, 240),      # Purple
}


def get_cue_rgb_color(cue_type):
    """Returns (R, G, B) tuple for a given cue_type string."""
    if not cue_type:
        return (255, 165, 0)
    clean = str(cue_type).strip().upper()
    if clean in CUE_SECTION_COLORS:
        return CUE_SECTION_COLORS[clean]
    for prefix, rgb in CUE_SECTION_COLORS.items():
        if clean.startswith(prefix):
            return rgb
    return (255, 165, 0)


def encode_serato_markers2(cues, bpm=120.0, camelot_key="8A", energy_score=5):
    """
    Builds binary payload for GEOB:Serato Markers2 containing up to 8 HotCues with RGB colors
    and millisecond positions according to Serato's Markers2 specification.
    """
    inner_stream = bytearray(b"\x01\x01")

    for idx, c in enumerate((cues or [])[:8]):
        cue_type = str(c.get("cue_type", f"CUE {idx + 1}"))
        pos_secs = max(0.0, float(c.get("position_secs", 0.0)))
        pos_ms = int(round(pos_secs * 1000.0))
        r, g, b = get_cue_rgb_color(cue_type)
        pad_idx = int(c.get("hotcue_num", idx + 1)) - 1
        pad_idx = max(0, min(7, pad_idx))

        label_bytes = cue_type.encode("utf-8") + b"\x00"
        # Entry payload: [0x00, pad_idx(1B), pos_ms(4B BE), 0x00, R, G, B, 0x00, 0x00, label_utf8_null]
        entry_data = (
            struct.pack(">BBI", 0, pad_idx, pos_ms)
            + b"\x00"
            + struct.pack(">BBB", r, g, b)
            + b"\x00\x00"
            + label_bytes
        )
        inner_stream.extend(b"CUE\x00")
        inner_stream.extend(struct.pack(">I", len(entry_data)))
        inner_stream.extend(entry_data)

    # Add BPMLOCK entry (unlocked = 0)
    inner_stream.extend(b"BPMLOCK\x00")
    inner_stream.extend(struct.pack(">IB", 1, 0))

    b64_payload = base64.b64encode(bytes(inner_stream))
    return b"\x01\x01" + b64_payload + b"\x00"


def write_id3_tags(file_path, bpm, camelot_key, detected_key, energy_score, cues=None):
    """
    Section 8.2 & Phase 3: Writes ID3v2.4 tags to MP3 / AIFF / FLAC files.
    - TKEY: Camelot Key + Musical Key (e.g., "8A - A Minor")
    - TBPM: BPM value formatted to 2 decimals (e.g., "124.00")
    - COMM: Comment string formatted for DJ software (e.g., "Energy 7 - 8A")
    - GEOB:Serato Markers2 & GEOB:Serato Markers_2: HotCue pad markers + metadata
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    try:
        try:
            audio = ID3(file_path)
        except ID3NoHeaderError:
            audio = ID3()

        key_str = f"{camelot_key} - {detected_key}"
        bpm_str = f"{float(bpm):.2f}"
        comm_str = f"Energy {energy_score} - {camelot_key}"

        audio["TKEY"] = TKEY(encoding=3, text=key_str)
        audio["TBPM"] = TBPM(encoding=3, text=bpm_str)
        audio["COMM::eng"] = COMM(encoding=3, lang="eng", desc="", text=comm_str)

        # Legacy summary marker frame
        serato_marker_data = f"AudioHarmonix|{camelot_key}|{bpm_str}|{energy_score}".encode("utf-8")
        audio["GEOB:Serato Markers_2"] = GEOB(
            encoding=0,
            mime="application/octet-stream",
            filename="Serato Markers_2",
            desc="Serato Markers_2",
            data=serato_marker_data
        )

        # Official Serato Markers2 binary frame with 8 colored HotCue pads
        markers2_bin = encode_serato_markers2(cues or [], bpm=bpm, camelot_key=camelot_key, energy_score=energy_score)
        audio["GEOB:Serato Markers2"] = GEOB(
            encoding=0,
            mime="application/octet-stream",
            filename="Serato Markers2",
            desc="Serato Markers2",
            data=markers2_bin
        )

        audio.save(file_path)
        return True, "ID3 tags and Serato Markers2 saved successfully"

    except Exception as e:
        return False, f"Error saving ID3 tags: {str(e)}"


def export_rekordbox_xml(output_xml_path, tracks_data):
    """
    Section 8.1 & Phase 3: Generates official Pioneer Rekordbox XML Export format
    with section-mapped RGB HotCue colors and Camelot Smart Playlists.
    """
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="AudioHarmonix", Version="2.0.0", Company="AudioHarmonix Team")

    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks_data)))
    camelot_groups = {}

    for idx, item in enumerate(tracks_data, 1):
        tr = item.get("track") or item
        an = item.get("analysis") or item
        cues = item.get("cues", [])

        file_url = "file://localhost/" + os.path.abspath(tr.get("file_path", "")).replace("\\", "/")
        cam_key = an.get("camelot_key", "8A") or "8A"
        camelot_groups.setdefault(cam_key, []).append(str(idx))

        track_elem = ET.SubElement(
            collection,
            "TRACK",
            TrackID=str(idx),
            Name=tr.get("title", tr.get("file_name", "Unknown")),
            Artist=tr.get("artist", "Unknown Artist"),
            Composer="",
            Album=tr.get("album", ""),
            Grouping="",
            Genre="",
            Kind="Audio File",
            Size=str(tr.get("file_size", 0)),
            TotalTime=str(int(tr.get("duration_secs", 0))),
            DiscNumber="0",
            TrackNumber="0",
            Year="2026",
            AverageBpm=f"{an.get('bpm', 120.0):.2f}",
            DateAdded="2026-08-11",
            BitRate="320",
            SampleRate="44100",
            Comments=f"Energy {an.get('energy_score', 5)} - {cam_key}",
            PlayCount="0",
            Rating="0",
            Location=file_url,
            Tonality=cam_key,
            Label="",
            Mix=""
        )

        # Add Tempo Grid Mark (First Beat anchored)
        bpm_val = float(an.get("bpm", 120.0))
        first_beat = 0.0
        if cues:
            first_beat = float(cues[0].get("position_secs", 0.0))
        ET.SubElement(track_elem, "TEMPO", Inizio=f"{first_beat:.3f}", Bpm=f"{bpm_val:.2f}", Metro="4/4", Battito="1")

        # Add Position / Cue Markers with section-accurate RGB colors
        for c_idx, c in enumerate(cues[:8], 1):
            cue_name = c.get("cue_type", "CUE")
            r, g, b = get_cue_rgb_color(cue_name)
            ET.SubElement(
                track_elem,
                "POSITION_MARK",
                Name=cue_name,
                Type="0",
                Start=f"{float(c.get('position_secs', 0.0)):.3f}",
                Num=str(c.get("hotcue_num", c_idx)),
                Red=str(r),
                Green=str(g),
                Blue=str(b)
            )

    # Add Playlists Node with Root -> AudioHarmonix Smart Crates
    playlists_elem = ET.SubElement(root, "PLAYLISTS")
    root_node = ET.SubElement(playlists_elem, "NODE", Type="0", Name="ROOT", Count="2")

    all_playlist = ET.SubElement(
        root_node, "NODE", Name="AudioHarmonix All Tracks", Type="1", KeyType="0", Entries=str(len(tracks_data))
    )
    for idx in range(1, len(tracks_data) + 1):
        ET.SubElement(all_playlist, "TRACK", Key=str(idx))

    camelot_folder = ET.SubElement(
        root_node, "NODE", Name="AudioHarmonix Camelot Crates", Type="0", Count=str(len(camelot_groups))
    )
    for cam_key in sorted(camelot_groups.keys()):
        t_ids = camelot_groups[cam_key]
        crate_node = ET.SubElement(
            camelot_folder, "NODE", Name=f"Key {cam_key}", Type="1", KeyType="0", Entries=str(len(t_ids))
        )
        for tid in t_ids:
            ET.SubElement(crate_node, "TRACK", Key=tid)

    # Format XML with indentation
    xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    return True, f"Rekordbox XML exported successfully to {output_xml_path} ({len(tracks_data)} tracks)"


def export_traktor_nml(output_nml_path, tracks_data):
    """
    Section 8.1 & Phase 3: Export Native Instruments Traktor Pro NML Format
    including <CUE_V2> HotCues in milliseconds.
    """
    root = ET.Element("NML", VERSION="19")
    ET.SubElement(root, "HEAD", COMPANY="AudioHarmonix", VERSION="2.0.0")
    collection = ET.SubElement(root, "COLLECTION", ENTRIES=str(len(tracks_data)))

    for idx, item in enumerate(tracks_data, 1):
        tr = item.get("track") or item
        an = item.get("analysis") or item
        cues = item.get("cues", [])

        entry = ET.SubElement(
            collection,
            "ENTRY",
            TITLE=tr.get("title", tr.get("file_name", "Unknown")),
            ARTIST=tr.get("artist", "Unknown Artist")
        )
        abs_path = os.path.abspath(tr.get("file_path", ""))
        drive, tail = os.path.splitdrive(abs_path)
        file_dir, file_name = os.path.split(tail)
        traktor_dir = "/:" + file_dir.strip("\\/").replace("\\", "/:").replace("/", "/:") + "/:"
        ET.SubElement(entry, "LOCATION", DIR=traktor_dir, FILE=file_name, VOLUME=drive or "C:")
        ET.SubElement(
            entry,
            "INFO",
            BITRATE="320000",
            KEY=an.get("camelot_key", "8A"),
            PLAYTIME=str(int(tr.get("duration_secs", 0))),
            COMMENT=f"Energy {an.get('energy_score', 5)} - {an.get('camelot_key', '8A')}"
        )
        ET.SubElement(entry, "TEMPO", BPM=f"{float(an.get('bpm', 120.0)):.2f}", BPM_QUALITY="100.000000")

        for c_idx, c in enumerate(cues[:8]):
            pos_ms = float(c.get("position_secs", 0.0)) * 1000.0
            hotcue_slot = int(c.get("hotcue_num", c_idx + 1)) - 1
            hotcue_slot = max(0, min(7, hotcue_slot))
            ET.SubElement(
                entry,
                "CUE_V2",
                NAME=str(c.get("cue_type", f"CUE {c_idx + 1}")),
                DISPL_ORDER="0",
                TYPE="0",
                START=f"{pos_ms:.3f}",
                LEN="0.000000",
                REPEATS="-1",
                HOTCUE=str(hotcue_slot)
            )

    xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    with open(output_nml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    return True, f"Traktor NML exported successfully to {output_nml_path} ({len(tracks_data)} tracks)"


def export_serato_crate(output_crate_path, tracks_data, write_tags=True):
    """
    Phase 3: Exports a binary Serato DJ Pro .crate file (UTF-16BE TLV structure)
    and optionally embeds GEOB:Serato Markers2 HotCues into each audio file's ID3 tags.
    """
    try:
        crate_bytes = bytearray()

        # 1. Version tag ('vrsn')
        vrsn_str = "81.0/Serato ScratchLive Crate".encode("utf-16-be")
        crate_bytes.extend(b"vrsn")
        crate_bytes.extend(struct.pack(">I", len(vrsn_str)))
        crate_bytes.extend(vrsn_str)

        updated_tags_count = 0

        # 2. Track entries ('otrk' -> 'ptrk')
        for item in tracks_data:
            tr = item.get("track") or item
            an = item.get("analysis") or item
            cues = item.get("cues", [])
            fp = tr.get("file_path", "")
            if not fp:
                continue

            abs_fp = os.path.abspath(fp)
            # Strip drive letter for Serato ptrk specification if on Windows, normalize slashes
            _, rel_no_drive = os.path.splitdrive(abs_fp)
            serato_path = rel_no_drive.lstrip("\\/").replace("\\", "/")
            ptrk_payload = serato_path.encode("utf-16-be")

            ptrk_block = b"ptrk" + struct.pack(">I", len(ptrk_payload)) + ptrk_payload
            otrk_block = b"otrk" + struct.pack(">I", len(ptrk_block)) + ptrk_block
            crate_bytes.extend(otrk_block)

            if write_tags and os.path.exists(abs_fp) and abs_fp.lower().endswith(".mp3"):
                ok, _ = write_id3_tags(
                    abs_fp,
                    an.get("bpm", 120.0),
                    an.get("camelot_key", "8A"),
                    an.get("detected_key", "A Minor"),
                    an.get("energy_score", 5),
                    cues=cues
                )
                if ok:
                    updated_tags_count += 1

        with open(output_crate_path, "wb") as f:
            f.write(bytes(crate_bytes))

        return True, f"Serato Crate exported to {output_crate_path} ({len(tracks_data)} tracks, {updated_tags_count} MP3 Markers2 updated)"
    except Exception as e:
        return False, f"Error exporting Serato Crate: {str(e)}"


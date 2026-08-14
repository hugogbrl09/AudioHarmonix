"""
AudioHarmonix Tag Writer & DJ Software Exporter
Section 8 & 11: Writes ID3v2.4 Tags (TKEY, TBPM, COMM) and Exports Rekordbox XML & Traktor NML
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import mutagen
from mutagen.id3 import ID3, TKEY, TBPM, COMM, GEOB, ID3NoHeaderError

def write_id3_tags(file_path, bpm, camelot_key, detected_key, energy_score):
    """
    Section 8.2: Writes ID3v2.4 tags to MP3 / AIFF / FLAC files.
    - TKEY: Camelot Key + Musical Key (e.g., "8A - A Minor")
    - TBPM: BPM value formatted to 2 decimals (e.g., "124.00")
    - COMM: Comment string formatted for DJ software (e.g., "Energy 7 - 8A")
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

        # Private GEOB frame for Serato Markers compatibility
        serato_marker_data = f"AudioHarmonix|{camelot_key}|{bpm_str}|{energy_score}".encode("utf-8")
        audio["GEOB:Serato Markers_2"] = GEOB(
            encoding=0,
            mime="application/octet-stream",
            filename="Serato Markers_2",
            desc="Serato Markers_2",
            data=serato_marker_data
        )

        audio.save(file_path)
        return True, "ID3 tags saved successfully"

    except Exception as e:
        return False, f"Error saving ID3 tags: {str(e)}"

def export_rekordbox_xml(output_xml_path, tracks_data):
    """
    Section 8.1: Generates official Pioneer Rekordbox XML Export format.
    tracks_data: list of dicts containing track, analysis, and cue points
    """
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="AudioHarmonix", Version="1.0.0", Company="AudioHarmonix Team")

    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks_data)))

    for idx, item in enumerate(tracks_data, 1):
        tr = item.get("track", {})
        an = item.get("analysis", {})
        cues = item.get("cues", [])

        file_url = "file://localhost/" + os.path.abspath(tr.get("file_path", "")).replace("\\", "/")

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
            Comments=f"Energy {an.get('energy_score', 5)} - {an.get('camelot_key', '8A')}",
            PlayCount="0",
            Rating="0",
            Location=file_url,
            Tonality=an.get("camelot_key", "8A"),
            Label="",
            Mix=""
        )

        # Add Tempo Grid Mark
        bpm_val = float(an.get("bpm", 120.0))
        ET.SubElement(track_elem, "TEMPO", Inizio="0.000", Bpm=f"{bpm_val:.2f}", Metro="4/4", Battito="1")

        # Add Position / Cue Markers
        for c_idx, c in enumerate(cues, 1):
            ET.SubElement(
                track_elem,
                "POSITION_MARK",
                Name=c.get("cue_type", "CUE"),
                Type="0",
                Start=f"{c.get('position_secs', 0.0):.3f}",
                Num=str(c.get("hotcue_num", c_idx)),
                Red="255",
                Green="165",
                Blue="0"
            )

    # Format XML with indentation
    xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    return True, f"Rekordbox XML exported successfully to {output_xml_path}"

def export_traktor_nml(output_nml_path, tracks_data):
    """
    Section 8.1: Export Native Instruments Traktor NML Format
    """
    root = ET.Element("NML", VERSION="19")
    ET.SubElement(root, "HEAD", COMPANY="AudioHarmonix", VERSION="1.0.0")
    collection = ET.SubElement(root, "COLLECTION", ENTRIES=str(len(tracks_data)))

    for idx, item in enumerate(tracks_data, 1):
        tr = item.get("track", {})
        an = item.get("analysis", {})

        entry = ET.SubElement(
            collection,
            "ENTRY",
            TITLE=tr.get("title", tr.get("file_name", "Unknown")),
            ARTIST=tr.get("artist", "Unknown Artist")
        )
        file_dir, file_name = os.path.split(os.path.abspath(tr.get("file_path", "")))
        ET.SubElement(entry, "LOCATION", DIR=file_dir.replace("\\", "/"), FILE=file_name, VOLUME="C:")
        ET.SubElement(entry, "INFO", BITRATE="320000", KEY=an.get("camelot_key", "8A"), COMMENT=f"Energy {an.get('energy_score', 5)}")
        ET.SubElement(entry, "TEMPO", BPM=f"{an.get('bpm', 120.0):.2f}")

    xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    with open(output_nml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    return True, f"Traktor NML exported to {output_nml_path}"

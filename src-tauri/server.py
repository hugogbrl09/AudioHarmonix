"""
AudioHarmonix Backend Core Server & IPC Handler
Section 3 & 10: Multithreaded Worker Pool, Realtime Progress, API & Local IPC Server
"""

import os
import sys
import time
import json
import glob
import base64
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Add project sub-crates to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "audio_decoder"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "dsp_core"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "ml_engine"))
sys.path.insert(0, os.path.join(BASE_DIR, "crates", "tag_writer"))
sys.path.insert(0, os.path.join(BASE_DIR, "src-tauri"))

import decoder
import dsp
import ml
from tagger import write_id3_tags, export_rekordbox_xml, export_traktor_nml
from db import DatabaseManager

db_manager = DatabaseManager()
ml_key_detector = ml.KeyDetector()
ml_structure_detector = ml.StructureDetector()
ml_energy_detector = ml.EnergyDetector()

# Global Batch Processing state
batch_state = {
    "is_running": False,
    "total_files": 0,
    "processed_files": 0,
    "current_file": "",
    "start_time": 0.0,
    "tracks_per_sec": 0.0,
    "eta_seconds": 0,
    "cpu_usage_pct": 0.0
}

# Use N-1 threads for background processing to keep UI thread responsive
NUM_THREADS = max(1, os.cpu_count() - 1)
thread_pool = ThreadPoolExecutor(max_workers=NUM_THREADS)

def extract_audio_metadata(file_path):
    """
    Extracts real ID3 / Vorbis / MP4 / WAV metadata and embedded artwork from audio files.
    Falls back gracefully to filename without extension only if metadata is absent.
    """
    file_name = os.path.basename(file_path)
    fallback_title = os.path.splitext(file_name)[0]
    
    title = None
    artist = None
    album = None
    genre = None
    has_artwork = False

    try:
        import mutagen
        audio = mutagen.File(file_path)
        if audio:
            # 1. Title Extraction
            for k in ['TIT2', 'title', '\xa9nam', 'TITLE', 'Title']:
                val = audio.get(k)
                if val:
                    val_str = str(val[0] if isinstance(val, (list, tuple)) else val).strip()
                    if val_str:
                        title = val_str
                        break

            # 2. Artist Extraction
            for k in ['TPE1', 'artist', '\xa9ART', 'ARTIST', 'Artist', 'TPE2']:
                val = audio.get(k)
                if val:
                    val_str = str(val[0] if isinstance(val, (list, tuple)) else val).strip()
                    if val_str:
                        artist = val_str
                        break

            # 3. Album Extraction
            for k in ['TALB', 'album', '\xa9alb', 'ALBUM', 'Album']:
                val = audio.get(k)
                if val:
                    val_str = str(val[0] if isinstance(val, (list, tuple)) else val).strip()
                    if val_str:
                        album = val_str
                        break

            # 4. Check for embedded artwork
            if hasattr(audio, 'tags') and audio.tags:
                for key in audio.tags.keys():
                    if str(key).startswith('APIC'):
                        has_artwork = True
                        break
            if hasattr(audio, 'pictures') and audio.pictures:
                has_artwork = True
            if 'covr' in audio and audio['covr']:
                has_artwork = True
    except Exception as e:
        print(f"[-] Notice during metadata extraction for {file_path}: {e}")

    if not title:
        title = fallback_title
    if not artist:
        artist = "Unknown Artist"
    if not album:
        album = "Unknown Album"

    return {
        "title": title,
        "artist": artist,
        "album": album,
        "genre": genre,
        "has_artwork": has_artwork
    }

def get_track_artwork(file_path):
    """Extracts raw image bytes and mime type from embedded audio file artwork"""
    try:
        import mutagen
        audio = mutagen.File(file_path)
        if not audio:
            return None, None

        if hasattr(audio, 'tags') and audio.tags:
            for k, v in audio.tags.items():
                if str(k).startswith('APIC'):
                    mime = getattr(v, 'mime', 'image/jpeg') or 'image/jpeg'
                    return v.data, mime

        if hasattr(audio, 'pictures') and audio.pictures:
            pic = audio.pictures[0]
            mime = getattr(pic, 'mime', 'image/jpeg') or 'image/jpeg'
            return pic.data, mime

        if 'covr' in audio and audio['covr']:
            covr = audio['covr'][0]
            mime = 'image/png' if covr.startswith(b'\x89PNG') else 'image/jpeg'
            return bytes(covr), mime
    except Exception as e:
        print(f"[-] Notice extracting artwork from {file_path}: {e}")

    return None, None

def refresh_all_tracks_metadata():
    """Scans existing database tracks and updates metadata/artwork flags if missing"""
    try:
        tracks = db_manager.get_all_tracks()
        for t in tracks:
            fp = t.get("file_path", "")
            if fp and os.path.exists(fp):
                meta = extract_audio_metadata(fp)
                # Update if artist was Unknown Artist or title was raw filename
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                    UPDATE tracks SET title = ?, artist = ?, album = ?
                    WHERE id = ?
                    """, (meta["title"], meta["artist"], meta["album"], t["id"]))
                    conn.commit()
    except Exception as e:
        print(f"[-] Notice refreshing track metadata: {e}")

def resolve_file_path(file_path):
    """Resolves absolute or relative file paths in project directory"""
    if not file_path:
        return None
    if os.path.exists(file_path):
        return file_path
    # Try resolving relative to BASE_DIR
    rel_path = os.path.join(BASE_DIR, file_path)
    if os.path.exists(rel_path):
        return rel_path
    # Try sample_tracks
    sample_path = os.path.join(BASE_DIR, "sample_tracks", os.path.basename(file_path))
    if os.path.exists(sample_path):
        return sample_path
    return None

def process_single_file(file_path):
    """Full AudioHarmonix Analysis Pipeline for a single track"""
    resolved_path = resolve_file_path(file_path)
    if not resolved_path:
        raise FileNotFoundError(f"Audio file '{file_path}' not found on server disk.")

    file_path = resolved_path
    t0 = time.time()
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    # 1. Decode PCM & Resample to 22.050Hz Mono
    y, sr, duration_secs = decoder.load_and_resample(file_path)

    # 2. Extract Energy Score (1-10) via Neural EnergyNet (with DSP fallback)
    dsp_energy = dsp.compute_energy_score(y, sr=sr)
    energy_score = ml_energy_detector.predict_energy(y, sr=sr, dsp_fallback_energy=dsp_energy)

    # 3. CQT & Key Detection via Neural KeyNet + Harmonic Chromagram
    cqt_matrix, chromagram = dsp.compute_cqt(y, sr=sr)
    detected_key, camelot_key, open_key, key_confidence = ml_key_detector.predict_key(cqt_matrix, chromagram=chromagram)

    # 4. BPM & Beat Tracking
    bpm, bpm_confidence, beat_timestamps, is_variable_bpm = dsp.estimate_bpm_and_beatgrid(y, sr=sr)

    # 5. Cue Points Detection via Neural StructureNet (with DSP State Machine fallback)
    cues = ml_structure_detector.predict_cues(y, beat_timestamps, duration_secs, sr=sr, dsp_fallback_fn=dsp.detect_cue_points)

    # 6. 3-Band Waveform Peak extraction
    waveform_peaks = dsp.generate_3band_waveform_peaks(y, sr=sr, num_points=600)

    # 7. Real Metadata Extraction from audio file
    meta = extract_audio_metadata(file_path)
    title = meta["title"]
    artist = meta["artist"]
    album = meta["album"]
    has_artwork = meta["has_artwork"]

    analysis_dict = {
        "bpm": bpm,
        "bpm_confidence": bpm_confidence,
        "detected_key": detected_key,
        "camelot_key": camelot_key,
        "key_confidence": key_confidence,
        "energy_score": energy_score,
        "is_variable_bpm": is_variable_bpm
    }

    # 8. Save results to SQLite Database
    track_id = db_manager.upsert_track_analysis(
        file_path, file_name, file_size, duration_secs, title, artist, album, analysis_dict, cues, waveform_peaks=waveform_peaks
    )

    # 9. Write ID3 Tags to File
    write_id3_tags(file_path, bpm, camelot_key, detected_key, energy_score)

    processing_time = time.time() - t0

    return {
        "track_id": track_id,
        "file_path": file_path,
        "file_name": file_name,
        "title": title,
        "artist": artist,
        "album": album,
        "has_artwork": has_artwork,
        "duration_secs": duration_secs,
        "bpm": bpm,
        "bpm_confidence": bpm_confidence,
        "detected_key": detected_key,
        "camelot_key": camelot_key,
        "key_confidence": key_confidence,
        "energy_score": energy_score,
        "is_variable_bpm": is_variable_bpm,
        "cues": cues,
        "waveform_peaks": waveform_peaks,
        "processing_time_secs": round(processing_time, 3)
    }

def scan_available_audio_files():
    """Scans sample_tracks and root project folder for audio files"""
    patterns = ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.aiff", "*.ogg"]
    found = []
    
    # Root dir
    for p in patterns:
        found.extend(glob.glob(os.path.join(BASE_DIR, p)))
        
    # sample_tracks dir
    sample_dir = os.path.join(BASE_DIR, "sample_tracks")
    if os.path.exists(sample_dir):
        for p in patterns:
            found.extend(glob.glob(os.path.join(sample_dir, p)))

    # Filter duplicates and normalize paths
    unique_paths = list(dict.fromkeys(os.path.normpath(f) for f in found))
    return unique_paths

def run_batch_analysis(file_paths):
    """Executes batch processing in parallel using N-1 threads"""
    global batch_state
    batch_state["is_running"] = True
    batch_state["total_files"] = len(file_paths)
    batch_state["processed_files"] = 0
    batch_state["start_time"] = time.time()

    futures = []
    for fp in file_paths:
        futures.append(thread_pool.submit(process_single_file, fp))

    for idx, f in enumerate(futures, 1):
        try:
            res = f.result()
            batch_state["processed_files"] = idx
            batch_state["current_file"] = res["file_name"]
            elapsed = time.time() - batch_state["start_time"]
            speed = idx / max(0.1, elapsed)
            batch_state["tracks_per_sec"] = round(speed, 2)
            remaining = batch_state["total_files"] - idx
            batch_state["eta_seconds"] = int(round(remaining / max(0.01, speed)))
        except Exception as err:
            print(f"Error analyzing track {idx}: {err}")

    batch_state["is_running"] = False
    batch_state["current_file"] = "Complete"

class AudioHarmonixHTTPHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/tracks":
            search_text = query.get("search", [""])[0]
            camelot_filter = query.get("camelot", [""])[0]
            bpm_min = float(query.get("bpm_min", [0])[0])
            bpm_max = float(query.get("bpm_max", [300])[0])
            energy_min = int(query.get("energy_min", [1])[0])

            tracks = db_manager.get_all_tracks(
                search_text=search_text,
                camelot_filter=camelot_filter,
                bpm_min=bpm_min,
                bpm_max=bpm_max,
                energy_min=energy_min
            )

            # Attach has_artwork flag to tracks
            for t in tracks:
                fp = t.get("file_path", "")
                t["has_artwork"] = False
                if fp and os.path.exists(fp):
                    try:
                        meta = extract_audio_metadata(fp)
                        t["has_artwork"] = meta["has_artwork"]
                        t["title"] = meta["title"]
                        t["artist"] = meta["artist"]
                        t["album"] = meta["album"]
                    except Exception:
                        pass

            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "tracks": tracks}).encode("utf-8"))

        elif path == "/api/artwork":
            track_id = query.get("id", [""])[0]
            track = db_manager.get_track_by_id(track_id) if track_id else None
            if track and os.path.exists(track.get("file_path", "")):
                img_data, mime = get_track_artwork(track["file_path"])
                if img_data:
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(img_data)))
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(img_data)
                    return

            self._set_headers(404, "text/plain")
            self.wfile.write(b"Artwork not found")

        elif path == "/api/batch_status":
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "batch": batch_state}).encode("utf-8"))

        elif path == "/api/scan_files":
            files = scan_available_audio_files()
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "files": files}).encode("utf-8"))

        elif path.startswith("/ui/") or path.startswith("/src/") or path.endswith(".css") or path.endswith(".js") or path.endswith(".png") or path.endswith(".svg") or path.endswith(".jpg") or path.endswith(".jpeg"):
            rel_path = path[4:] if path.startswith("/ui/") else path.lstrip("/")
            full_path = os.path.join(BASE_DIR, "ui", rel_path)
            if not os.path.exists(full_path) or os.path.isdir(full_path):
                full_path = os.path.join(BASE_DIR, "ui", "index.html")

            ctype = "text/html"
            if full_path.endswith(".css"):
                ctype = "text/css"
            elif full_path.endswith(".js"):
                ctype = "application/javascript"
            elif full_path.endswith(".png"):
                ctype = "image/png"
            elif full_path.endswith(".svg"):
                ctype = "image/svg+xml"
            elif full_path.endswith(".jpg") or full_path.endswith(".jpeg"):
                ctype = "image/jpeg"

            try:
                with open(full_path, "rb") as f:
                    content = f.read()
                self._set_headers(200, ctype)
                self.wfile.write(content)
            except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._set_headers(404, "text/plain")
                    self.wfile.write(f"Not found: {e}".encode("utf-8"))
                except Exception:
                    pass

        elif path.startswith("/audio/"):
            file_id = query.get("id", [""])[0]
            track = db_manager.get_track_by_id(file_id) if file_id else None
            if track and os.path.exists(track["file_path"]):
                try:
                    file_path = track["file_path"]
                    file_size = os.path.getsize(file_path)
                    range_header = self.headers.get("Range")

                    if range_header and range_header.startswith("bytes="):
                        byte_range = range_header.replace("bytes=", "").split("-")
                        start_byte = int(byte_range[0]) if byte_range[0] else 0
                        end_byte = int(byte_range[1]) if (len(byte_range) > 1 and byte_range[1]) else file_size - 1
                        end_byte = min(end_byte, file_size - 1)
                        length = end_byte - start_byte + 1

                        self.send_response(206)
                        self.send_header("Content-Type", "audio/mpeg")
                        self.send_header("Accept-Ranges", "bytes")
                        self.send_header("Content-Range", f"bytes {start_byte}-{end_byte}/{file_size}")
                        self.send_header("Content-Length", str(length))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()

                        with open(file_path, "rb") as f:
                            f.seek(start_byte)
                            chunk = f.read(length)
                            self.wfile.write(chunk)
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "audio/mpeg")
                        self.send_header("Accept-Ranges", "bytes")
                        self.send_header("Content-Length", str(file_size))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()

                        with open(file_path, "rb") as f:
                            self.wfile.write(f.read())
                except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:
                    try:
                        self._set_headers(500, "text/plain")
                        self.wfile.write(f"Audio read error: {e}".encode("utf-8"))
                    except Exception:
                        pass
            else:
                try:
                    self._set_headers(404, "text/plain")
                    self.wfile.write(b"Audio file not found")
                except Exception:
                    pass

        else:
            index_path = os.path.join(BASE_DIR, "ui", "index.html")
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self._set_headers(200, "text/html")
                self.wfile.write(content)
            else:
                self._set_headers(404)
                self.wfile.write(b"Not found")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        data = json.loads(body.decode("utf-8"))

        if self.path == "/api/analyze_file":
            file_path = data.get("file_path", "")
            if not file_path:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "No file_path provided"}).encode("utf-8"))
                return

            try:
                res = process_single_file(file_path)
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "ok", "result": res}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode("utf-8"))

        elif self.path == "/api/upload_and_analyze":
            file_name = data.get("file_name", "upload.mp3")
            b64_content = data.get("base64_data", "")
            if "," in b64_content:
                b64_content = b64_content.split(",", 1)[1]

            raw_bytes = base64.b64decode(b64_content)
            upload_dir = os.path.join(BASE_DIR, "sample_tracks")
            os.makedirs(upload_dir, exist_ok=True)
            saved_path = os.path.join(upload_dir, file_name)

            with open(saved_path, "wb") as f:
                f.write(raw_bytes)

            try:
                res = process_single_file(saved_path)
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "ok", "result": res}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode("utf-8"))

        elif self.path == "/api/analyze_batch":
            file_paths = data.get("file_paths", [])
            if not file_paths:
                file_paths = scan_available_audio_files()

            threading.Thread(target=run_batch_analysis, args=(file_paths,), daemon=True).start()
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "message": "Batch processing started", "total": len(file_paths)}).encode("utf-8"))

        elif self.path == "/api/delete_track":
            track_id = data.get("track_id", "")
            if not track_id:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "No track_id provided"}).encode("utf-8"))
                return

            db_manager.delete_track(track_id)
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "message": f"Track {track_id} deleted successfully"}).encode("utf-8"))

        elif self.path == "/api/export_rekordbox":
            out_xml = data.get("output_path", os.path.join(BASE_DIR, "rekordbox.xml"))
            tracks = db_manager.get_all_tracks()
            tracks_data = []
            for t in tracks:
                tracks_data.append({
                    "track": t,
                    "analysis": t,
                    "cues": t.get("cues", [])
                })
            success, msg = export_rekordbox_xml(out_xml, tracks_data)
            self._set_headers(200 if success else 500)
            self.wfile.write(json.dumps({"status": "ok" if success else "error", "message": msg}).encode("utf-8"))

        elif self.path == "/api/save_user_cues":
            track_id = data.get("track_id", "")
            cues = data.get("cues", [])
            teach_ai = bool(data.get("teach_ai", False))
            if not track_id or not isinstance(cues, list):
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Invalid request: track_id and cues required"}).encode("utf-8"))
                return

            cues = sorted(cues, key=lambda c: float(c.get("position_secs", 0)))
            for idx, c in enumerate(cues, 1):
                c["hotcue_num"] = idx

            db_manager.update_user_cues(track_id, cues)
            
            # Update BPM if provided
            bpm_val = data.get("bpm")
            if bpm_val:
                try:
                    db_manager.update_track_bpm(track_id, float(bpm_val))
                except Exception as e:
                    print(f"[-] Notice updating BPM: {e}")

            track = db_manager.get_track_by_id(track_id)
            if bpm_val and track and os.path.exists(track.get("file_path", "")):
                try:
                    write_id3_tags(track["file_path"], float(bpm_val), track.get("camelot_key", ""), track.get("detected_key", ""), track.get("energy_score", 5))
                except Exception:
                    pass

            # Active Learning Online Fine-Tuning in background only if teach_ai is True
            if teach_ai and track and os.path.exists(track.get("file_path", "")):
                threading.Thread(target=ml.adapt_structure_model, args=(track["file_path"], cues), daemon=True).start()

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "ok",
                "message": "HotCues e BPM salvos com sucesso e IA adaptada ao seu estilo!" if teach_ai else "HotCues e BPM salvos com sucesso no banco de dados!",
                "cues": cues,
                "bpm": track.get("bpm") if track else bpm_val,
                "taught_ai": teach_ai
            }).encode("utf-8"))

        else:
            self._set_headers(404)
            self.wfile.write(b"Endpoint not found")

def start_server(port=8888):
    refresh_all_tracks_metadata()
    server = ThreadingHTTPServer(("127.0.0.1", port), AudioHarmonixHTTPHandler)
    print(f"AudioHarmonix Backend running at http://127.0.0.1:{port}")
    server.serve_forever()

if __name__ == "__main__":
    start_server(8888)

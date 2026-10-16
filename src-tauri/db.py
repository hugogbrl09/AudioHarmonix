"""
AudioHarmonix Database Module
Section 17 & Roadmap Phase 1: SQLite Schema, High-Speed WAL/MMAP Engine & Binary Waveform BLOB Cache
"""

import os
import json
import struct
import sqlite3
import uuid


def pack_waveform_blob(waveform_peaks):
    """
    Packs waveform peaks (3-band RGB dict or 1D list of floats) into a compact IEEE-754 float32 binary BLOB.
    Reduces storage footprint by ~78% compared to JSON text serialization.
    """
    if not waveform_peaks:
        return None

    if isinstance(waveform_peaks, dict) and "low" in waveform_peaks and "mid" in waveform_peaks and "high" in waveform_peaks:
        low = [float(x) for x in waveform_peaks["low"]]
        mid = [float(x) for x in waveform_peaks["mid"]]
        high = [float(x) for x in waveform_peaks["high"]]
        n = min(len(low), len(mid), len(high))
        header = struct.pack("<4sI", b"AHW3", n)
        body = (
            struct.pack(f"<{n}f", *low[:n])
            + struct.pack(f"<{n}f", *mid[:n])
            + struct.pack(f"<{n}f", *high[:n])
        )
        return header + body

    if isinstance(waveform_peaks, (list, tuple)):
        vals = [float(x) for x in waveform_peaks]
        n = len(vals)
        header = struct.pack("<4sI", b"AHW1", n)
        body = struct.pack(f"<{n}f", *vals)
        return header + body

    return None


def unpack_waveform_blob(blob_bytes):
    """
    Unpacks a binary BLOB created by pack_waveform_blob back into Python structures.
    """
    if not blob_bytes or len(blob_bytes) < 8:
        return None

    magic, n = struct.unpack("<4sI", blob_bytes[:8])
    payload = blob_bytes[8:]

    if magic == b"AHW3":
        expected_bytes = n * 4 * 3
        if len(payload) < expected_bytes:
            return None
        floats = struct.unpack(f"<{n * 3}f", payload[:expected_bytes])
        return {
            "low": [round(x, 4) for x in floats[0:n]],
            "mid": [round(x, 4) for x in floats[n:2 * n]],
            "high": [round(x, 4) for x in floats[2 * n:3 * n]],
        }
    elif magic == b"AHW1":
        expected_bytes = n * 4
        if len(payload) < expected_bytes:
            return None
        floats = struct.unpack(f"<{n}f", payload[:expected_bytes])
        return [round(x, 4) for x in floats]

    return None


class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "audioharmonix.db")

        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA mmap_size = 268435456;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Creates tables, indexes, and runs automatic binary BLOB migration (Phase 1)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tracks table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                id TEXT PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                duration_secs REAL NOT NULL,
                title TEXT,
                artist TEXT,
                album TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Analysis Results table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                track_id TEXT PRIMARY KEY,
                bpm REAL NOT NULL,
                bpm_confidence REAL NOT NULL,
                detected_key TEXT NOT NULL,
                camelot_key TEXT NOT NULL,
                key_confidence REAL NOT NULL,
                energy_score INTEGER NOT NULL,
                is_variable_bpm BOOLEAN DEFAULT 0,
                waveform_peaks TEXT,
                waveform_blob BLOB,
                analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            );
            """)

            # Ensure waveform_peaks & waveform_blob columns exist if table was created earlier
            try:
                cursor.execute("ALTER TABLE analysis_results ADD COLUMN waveform_peaks TEXT;")
            except Exception:
                pass

            try:
                cursor.execute("ALTER TABLE analysis_results ADD COLUMN waveform_blob BLOB;")
            except Exception:
                pass

            # Cue Points table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS cue_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT NOT NULL,
                cue_type TEXT NOT NULL,
                position_secs REAL NOT NULL,
                hotcue_num INTEGER,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            );
            """)

            # Indexes for fast search
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_file_path ON tracks(file_path);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_bpm ON analysis_results(bpm);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_camelot ON analysis_results(camelot_key);")

            # Transparent Migration: Convert legacy JSON waveform_peaks to binary waveform_blob
            cursor.execute("""
            SELECT track_id, waveform_peaks FROM analysis_results
            WHERE waveform_blob IS NULL AND waveform_peaks IS NOT NULL
            """)
            legacy_rows = cursor.fetchall()
            for row in legacy_rows:
                try:
                    peaks_obj = json.loads(row["waveform_peaks"])
                    blob_data = pack_waveform_blob(peaks_obj)
                    if blob_data:
                        cursor.execute("""
                        UPDATE analysis_results
                        SET waveform_blob = ?, waveform_peaks = NULL
                        WHERE track_id = ?
                        """, (sqlite3.Binary(blob_data), row["track_id"]))
                except Exception:
                    pass
            
            conn.commit()

    def upsert_track_analysis(self, file_path, file_name, file_size, duration_secs, title, artist, album, analysis_dict, cues_list, waveform_peaks=None):
        """Atomically inserts or updates track, analysis, cue points, and binary waveform BLOB"""
        file_path = os.path.abspath(file_path)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            if row:
                track_id = row['id']
                cursor.execute("""
                UPDATE tracks SET file_name=?, file_size=?, duration_secs=?, title=?, artist=?, album=?
                WHERE id=?
                """, (file_name, file_size, duration_secs, title, artist, album, track_id))
            else:
                track_id = str(uuid.uuid4())
                cursor.execute("""
                INSERT INTO tracks (id, file_path, file_name, file_size, duration_secs, title, artist, album)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (track_id, file_path, file_name, file_size, duration_secs, title, artist, album))

            wf_blob = pack_waveform_blob(waveform_peaks) if waveform_peaks else None

            cursor.execute("""
            INSERT INTO analysis_results (track_id, bpm, bpm_confidence, detected_key, camelot_key, key_confidence, energy_score, is_variable_bpm, waveform_peaks, waveform_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(track_id) DO UPDATE SET
                bpm=excluded.bpm,
                bpm_confidence=excluded.bpm_confidence,
                detected_key=excluded.detected_key,
                camelot_key=excluded.camelot_key,
                key_confidence=excluded.key_confidence,
                energy_score=excluded.energy_score,
                is_variable_bpm=excluded.is_variable_bpm,
                waveform_peaks=NULL,
                waveform_blob=excluded.waveform_blob,
                analyzed_at=CURRENT_TIMESTAMP
            """, (
                track_id,
                analysis_dict['bpm'],
                analysis_dict['bpm_confidence'],
                analysis_dict['detected_key'],
                analysis_dict['camelot_key'],
                analysis_dict['key_confidence'],
                analysis_dict['energy_score'],
                1 if analysis_dict.get('is_variable_bpm') else 0,
                sqlite3.Binary(wf_blob) if wf_blob else None
            ))

            cursor.execute("DELETE FROM cue_points WHERE track_id = ?", (track_id,))
            for cue in cues_list:
                cursor.execute("""
                INSERT INTO cue_points (track_id, cue_type, position_secs, hotcue_num)
                VALUES (?, ?, ?, ?)
                """, (track_id, cue['cue_type'], cue['position_secs'], cue.get('hotcue_num')))

            conn.commit()
            return track_id

    def get_all_tracks(self, search_text="", camelot_filter="", bpm_min=0, bpm_max=300, energy_min=1):
        """Queries library tracks with optional text search and filters"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
            SELECT 
                t.id, t.file_path, t.file_name, t.file_size, t.duration_secs, t.title, t.artist, t.album,
                a.bpm, a.bpm_confidence, a.detected_key, a.camelot_key, a.key_confidence, a.energy_score, a.is_variable_bpm, a.waveform_peaks, a.waveform_blob, a.analyzed_at
            FROM tracks t
            LEFT JOIN analysis_results a ON t.id = a.track_id
            WHERE 1=1
            """
            params = []

            if search_text:
                query += " AND (t.title LIKE ? OR t.artist LIKE ? OR t.file_name LIKE ? OR a.camelot_key LIKE ?)"
                pattern = f"%{search_text}%"
                params.extend([pattern, pattern, pattern, pattern])

            if camelot_filter:
                query += " AND a.camelot_key = ?"
                params.append(camelot_filter)

            if bpm_min > 0:
                query += " AND a.bpm >= ?"
                params.append(bpm_min)

            if bpm_max < 300:
                query += " AND a.bpm <= ?"
                params.append(bpm_max)

            if energy_min > 1:
                query += " AND a.energy_score >= ?"
                params.append(energy_min)

            query += " ORDER BY t.created_at DESC"
            cursor.execute(query, params)
            rows = [dict(r) for r in cursor.fetchall()]

            for r in rows:
                wf_blob = r.pop('waveform_blob', None)
                if wf_blob:
                    r['waveform_peaks'] = unpack_waveform_blob(wf_blob)
                elif r.get('waveform_peaks'):
                    try:
                        r['waveform_peaks'] = json.loads(r['waveform_peaks'])
                    except Exception:
                        r['waveform_peaks'] = None
                if r['id']:
                    cursor.execute("SELECT cue_type, position_secs, hotcue_num FROM cue_points WHERE track_id = ?", (r['id'],))
                    r['cues'] = [dict(c) for c in cursor.fetchall()]
                else:
                    r['cues'] = []

            return rows

    def get_track_by_id(self, track_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT t.*, a.bpm, a.bpm_confidence, a.detected_key, a.camelot_key, a.key_confidence, a.energy_score, a.is_variable_bpm, a.waveform_peaks, a.waveform_blob
            FROM tracks t
            LEFT JOIN analysis_results a ON t.id = a.track_id
            WHERE t.id = ?
            """, (track_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            wf_blob = res.pop('waveform_blob', None)
            if wf_blob:
                res['waveform_peaks'] = unpack_waveform_blob(wf_blob)
            elif res.get('waveform_peaks'):
                try:
                    res['waveform_peaks'] = json.loads(res['waveform_peaks'])
                except Exception:
                    res['waveform_peaks'] = None
            cursor.execute("SELECT cue_type, position_secs, hotcue_num FROM cue_points WHERE track_id = ?", (track_id,))
            res['cues'] = [dict(c) for c in cursor.fetchall()]
            return res

    def get_waveform_blob(self, track_id):
        """Returns the raw IEEE-754 binary waveform BLOB for zero-copy streaming."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT waveform_blob, waveform_peaks FROM analysis_results WHERE track_id = ?", (track_id,))
            row = cursor.fetchone()
            if not row:
                return None
            if row["waveform_blob"]:
                return bytes(row["waveform_blob"])
            if row["waveform_peaks"]:
                try:
                    return pack_waveform_blob(json.loads(row["waveform_peaks"]))
                except Exception:
                    return None
            return None

    def update_user_cues(self, track_id, cues_list):
        """Updates cue points for a track and registers annotation for active learning."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_cue_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT NOT NULL,
                cues_json TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cursor.execute("DELETE FROM cue_points WHERE track_id = ?", (track_id,))
            for idx, cue in enumerate(cues_list, 1):
                cursor.execute("""
                INSERT INTO cue_points (track_id, cue_type, position_secs, hotcue_num)
                VALUES (?, ?, ?, ?)
                """, (track_id, cue['cue_type'], cue['position_secs'], cue.get('hotcue_num', idx)))

            cursor.execute("""
            INSERT INTO user_cue_annotations (track_id, cues_json)
            VALUES (?, ?)
            """, (track_id, json.dumps(cues_list)))
            conn.commit()
            return True

    def update_track_bpm(self, track_id, bpm):
        """Updates track BPM in analysis_results table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE analysis_results SET bpm = ? WHERE track_id = ?
            """, (float(bpm), track_id))
            conn.commit()
            return True

    def delete_track(self, track_id):
        """Deletes a track and all cascade foreign keys (analysis, cues)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
            conn.commit()
            return True

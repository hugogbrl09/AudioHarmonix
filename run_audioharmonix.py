#!/usr/bin/env python3
"""
AudioHarmonix Master Launcher
Initializes DB & ONNX model, starts local server and launches GUI interface.
"""

import os
import sys
import time
import webbrowser
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src-tauri"))

from server import start_server

def main():
    print("=" * 80)
    print("  AUDIOHARMONIX (v1.0.0) - High-Performance DJ Audio Analyzer Studio")
    print("  Starting local API backend & Web Interface...")
    print("=" * 80)

    # Launch server in thread
    server_thread = threading.Thread(target=start_server, args=(8888,), daemon=True)
    server_thread.start()

    time.sleep(1.0)

    url = "http://127.0.0.1:8888/ui/index.html"
    print(f"\n[+] AudioHarmonix Web UI ready at: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    print("\nPress Ctrl+C to stop AudioHarmonix.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nAudioHarmonix shut down cleanly.")

if __name__ == "__main__":
    main()

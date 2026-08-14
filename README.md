# 🎛️ AudioHarmonix

> **High-Performance Intelligent DJ Audio Analysis Studio & Harmonic Mixing Engine**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AudioHarmonix** is a professional-grade standalone DJ audio analysis workstation built for electronic music producers, DJs, and audio engineers. It combines advanced digital signal processing (DSP), deep learning neural networks (ONNX), real-time 60 FPS interactive multi-band RGB waveform rendering, active learning fine-tuning, and seamless Rekordbox XML export.

---

## 🌟 Key Features

### 1. 🎼 Neural Musical Key & Camelot Wheel Detection (`KeyNet`)
- **Deep CQT Architecture**: Computes Constant-Q Transforms (84 bins, 12 bins/octave, 32.7 Hz to 4186 Hz) to generate precise harmonic chroma representations.
- **Accurate Classification**: Detects all 24 major and minor keys mapped directly to the **Camelot Wheel** (`1A` - `12B`) with sub-5ms ONNX latency.
- **Trained on GiantSteps Dataset**: Validated against industry-standard benchmarks for electronic dance music.

### 2. 🌊 Real-Time 3-Band RGB Waveform Display (60 FPS)
- **Multi-Frequency Spectrum**:
  - 🔵 **Cyan / Highs**: Hi-hats, cymbals, air, and high synth transients (> 2.5 kHz).
  - 🟢 **Green / Mids**: Vocals, lead synths, snare bodies, and melodic instruments (400 Hz - 2.5 kHz).
  - 🔴 **Red / Lows**: Kick drums and sub-bass frequencies (< 400 Hz).
- **Dynamic Anchored Zoom**: <kbd>Ctrl + Scroll</kbd> zooms centered exactly on the mouse cursor position; toolbar controls center on the playhead needle.
- **Collapsible Waveform Panel**: Press <kbd>W</kbd> or click the centered divider handle to toggle between expanded waveform view and playlist view.

### 3. 🎯 Intelligent HotCue Detection & Active Learning (`StructureNet`)
- **Automated Cue Placement**: Identifies `FIRST_BEAT (1.1)`, `INTRO`, `BUILDUP`, `DROP_1`, `BREAK_1`, `DROP_2`, and `OUTRO`.
- **Interactive Drag & Drop**: Drag cue markers directly on the waveform with magnetic snap to beatgrid transients.
- **Active Learning Online Fine-Tuning**: When you reposition or create HotCues and click **`Save HotCues`**, the PyTorch active learning engine fine-tunes `StructureNet` with your custom annotations and re-exports the ONNX model in the background.

### 4. 📐 Precision Beatgrid & First Beat Calibration (`Set 1.1`)
- **Set 1.1 Downbeat**: Calibrate the exact first beat of bar 1 at the current needle position (<kbd>Shift + M</kbd>).
- **Micro-Nudge Grid (±5ms)**: Nudge the entire beatgrid with continuous click-and-hold buttons (`◀` / `▶`) to lock onto kick attacks.
- **Phrase Markers**: Visualizes 4-beat bars, 16-beat major sections, and 32-beat EDM phrase downbeats (`1.1`, `5.1`, `9.1`, `17.1`, `33.1`).
- **Interactive BPM Editor**: Direct BPM manual input, `/2`, `x2`, and Tap Tempo.

### 5. ⚡ Neural Energy Level Scoring (`EnergyNet`)
- Evaluates RMS energy, spectral centroid flux, low-frequency density, and rhythmic dynamism to rate dancefloor energy from `1` (Chill/Ambient) to `10` (Peak-Time Festival Banger).

### 6. 📁 Rekordbox 5.4.3 XML Export & ID3 Tagging
- Export your entire analyzed library with Camelot keys, exact BPM, and HotCue points directly into **Pioneer Rekordbox**, ready for USB export to CDJ-2000NXS2, CDJ-3000, and Opus-Quad players.
- Automatically writes ID3v2 tags (Key, BPM, Energy rating) directly into audio files.

---

## 🏗️ Architecture & Engine Overview

```
audioharmonix/
├── crates/
│   ├── audio_decoder/      # PCM Decoding, WAV/MP3 parsing, 22.05kHz resampling
│   ├── dsp_core/           # CQT transforms, 3-Band FFT extraction, Mel spectrograms
│   ├── ml_engine/          # KeyNet, StructureNet, EnergyNet, and Active Learning loop
│   └── tag_writer/         # ID3v2 metadata injection (Camelot, BPM, Energy)
├── models/                 # Pre-trained & ONNX Runtime models (Key, Structure, Energy)
├── training/               # PyTorch training pipelines, dataset loaders & benchmarks
├── src-tauri/              # Local High-Performance API backend & SQLite DB layer
├── ui/                     # Native Web UI with 60 FPS HTML5 Canvas RGB Waveform
├── sample_tracks/          # Sample audio tracks for immediate evaluation
├── tests/                  # Automated unit and integration test suite
└── run_audioharmonix.py    # Master single-command launcher
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/SEU_USUARIO/AudioHarmonix.git
cd AudioHarmonix
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `torch`, `torchaudio`, `librosa`, `onnxruntime`, `numpy`, `scipy`, `mutagen`, `soundfile`)*

### 3. Launch AudioHarmonix
```bash
python run_audioharmonix.py
```
AudioHarmonix will automatically start the local backend server and open the studio interface at `http://127.0.0.1:8888/ui/index.html`.

---

## ⌨️ Global DJ Keyboard Shortcuts

Press <kbd>?</kbd> or <kbd>F1</kbd> anywhere inside the application to open the interactive shortcuts cheatsheet:

| Category | Action | Shortcut | Description |
|---|---|---|---|
| 🔵 **Transport** | **Play / Pause** | <kbd>Space</kbd> | Toggle audio playback at 60 FPS |
| | **Nudge / Scrub Beat** | <kbd>←</kbd> / <kbd>→</kbd> | Seek ±1 beat on the beatgrid |
| | **Bar Seek** | <kbd>Shift</kbd> + <kbd>←</kbd> / <kbd>→</kbd> | Jump ±4 beats (1 full measure) |
| 🟢 **HotCues** | **Add HotCue** | <kbd>M</kbd> | Drop a HotCue at current playhead |
| | **Trigger HotCues** | <kbd>1</kbd> to <kbd>8</kbd> | Jump instantly to HotCue pads 1 through 8 |
| 🟣 **Beatgrid** | **Set First Beat (1.1)** | <kbd>Shift</kbd> + <kbd>M</kbd> | Calibrate Beatgrid 1.1 downbeat to needle |
| | **Toggle Beatgrid** | <kbd>B</kbd> | Show/hide 32-beat and 16-beat phrase lines |
| | **Toggle Snap** | <kbd>S</kbd> | Toggle magnetic beatgrid quantization |
| 🟠 **View** | **Zoom Waveform** | <kbd>+</kbd> / <kbd>-</kbd> / <kbd>0</kbd> | Zoom In / Zoom Out / Reset Zoom to 1x |
| | **Toggle Waveform / Playlist** | <kbd>W</kbd> | Collapse/expand waveform drawer |
| ⚪ **General** | **Shortcuts Cheat Sheet** | <kbd>?</kbd> / <kbd>F1</kbd> | Toggle keyboard shortcuts modal |
| | **Close Modal / Dismiss** | <kbd>Esc</kbd> | Close any open dialog window |

---

## 🧪 Running Automated Tests

Run the full automated test suite covering Key Detection, Structure Detection, Active Learning, and Audio Decoding:

```bash
python -m unittest discover -s tests
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

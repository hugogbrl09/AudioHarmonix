# 🎛️ AudioHarmonix

> **High-Performance Intelligent DJ Audio Analysis Studio & Harmonic Mixing Engine**  
> *Estúdio Inteligente de Alta Performance para Análise de Áudio DJ e Mixagem Harmônica*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#english) | [Português](#português)

---

<a name="english"></a>
## 🇬🇧 English

### 🌟 Key Features

#### 1. 🎼 Neural Musical Key & Camelot Wheel Detection (`KeyNet`)
- **Deep CQT Architecture**: Computes Constant-Q Transforms (84 bins, 12 bins/octave, 32.7 Hz to 4186 Hz) to generate precise harmonic chroma representations.
- **Accurate Classification**: Detects all 24 major and minor keys mapped directly to the **Camelot Wheel** (`1A` - `12B`) with sub-5ms ONNX latency.
- **Trained on GiantSteps Dataset**: Validated against industry-standard benchmarks for electronic dance music.

#### 2. 🌊 Real-Time 3-Band RGB Waveform Display (60 FPS)
- **Multi-Frequency Spectrum**:
  - 🔵 **Cyan / Highs**: Hi-hats, cymbals, air, and high synth transients (> 2.5 kHz).
  - 🟢 **Green / Mids**: Vocals, lead synths, snare bodies, and melodic instruments (400 Hz - 2.5 kHz).
  - 🔴 **Red / Lows**: Kick drums and sub-bass frequencies (< 400 Hz).
- **Dynamic Anchored Zoom**: <kbd>Ctrl + Scroll</kbd> zooms centered exactly on the mouse cursor position; toolbar controls center on the playhead needle.
- **Collapsible Waveform Panel**: Press <kbd>W</kbd> or click the centered divider handle to toggle between expanded waveform view and playlist view.

#### 3. 🎯 Intelligent HotCue Detection & Active Learning (`StructureNet`)
- **Automated Cue Placement**: Identifies `FIRST_BEAT (1.1)`, `INTRO`, `BUILDUP`, `DROP_1`, `BREAK_1`, `DROP_2`, and `OUTRO`.
- **Interactive Drag & Drop**: Drag cue markers directly on the waveform with magnetic snap to beatgrid transients.
- **Active Learning Online Fine-Tuning**: When you reposition or create HotCues and click **`Save HotCues`**, the PyTorch active learning engine fine-tunes `StructureNet` with your custom annotations and re-exports the ONNX model in the background.

#### 4. 📐 Precision Beatgrid & First Beat Calibration (`Set 1.1`)
- **Set 1.1 Downbeat**: Calibrate the exact first beat of bar 1 at the current needle position (<kbd>Shift + M</kbd>).
- **Micro-Nudge Grid (±5ms)**: Nudge the entire beatgrid with continuous click-and-hold buttons (`◀` / `▶`) to lock onto kick attacks.
- **Phrase Markers**: Visualizes 4-beat bars, 16-beat major sections, and 32-beat EDM phrase downbeats (`1.1`, `5.1`, `9.1`, `17.1`, `33.1`).
- **Interactive BPM Editor**: Direct BPM manual input, `/2`, `x2`, and Tap Tempo.

#### 5. ⚡ Neural Energy Level Scoring (`EnergyNet`)
- Evaluates RMS energy, spectral centroid flux, low-frequency density, and rhythmic dynamism to rate dancefloor energy from `1` (Chill/Ambient) to `10` (Peak-Time Festival Banger).

#### 6. 📁 Rekordbox 5.4.3 XML Export & ID3 Tagging
- Export your entire analyzed library with Camelot keys, exact BPM, and HotCue points directly into **Pioneer Rekordbox**, ready for USB export to CDJ-2000NXS2, CDJ-3000, and Opus-Quad players.
- Automatically writes ID3v2 tags (Key, BPM, Energy rating) directly into audio files.

---

### 🏗️ Architecture & Engine Overview

```
AudioHarmonix/
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

### 🚀 Quick Start (English)

```bash
# 1. Clone the repository
git clone https://github.com/hugogbrl09/AudioHarmonix.git
cd AudioHarmonix

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start AudioHarmonix
python run_audioharmonix.py
```
AudioHarmonix will start the local backend server and open the studio interface at `http://127.0.0.1:8888/ui/index.html`.

---

<a name="português"></a>
## 🇧🇷 Português

### 🌟 Funcionalidades Principais

#### 1. 🎼 Detecção Neural de Tonalidade & Camelot Wheel (`KeyNet`)
- **Arquitetura Baseada em CQT**: Processa Transformadas de Q-Constante (84 bins, 12 bins/oitava) para extrair o perfil harmônico exato da música.
- **Classificação Precisa**: Identifica todos os 24 tons maiores e menores mapeados diretamente na **Roda de Camelot** (`1A` a `12B`) com latência inferior a 5ms em ONNX.
- **Treinado no Dataset GiantSteps**: Validado com dados de referência da indústria de música eletrônica.

#### 2. 🌊 Waveform RGB de 3 Bandas em Tempo Real (60 FPS)
- **Separação Espectral Tri-Banda**:
  - 🔵 **Ciano / Agudos**: Hi-hats, pratos, ar e transientes de sintetizadores (> 2.5 kHz).
  - 🟢 **Verde / Médios**: Vocais, synths de liderança, caixas e instrumentos melódicos (400 Hz - 2.5 kHz).
  - 🔴 **Vermelho / Graves**: Bumbo/kick e frequências de sub-grave (< 400 Hz).
- **Zoom com Âncora Dinâmica**: <kbd>Ctrl + Scroll</kbd> dá zoom exatamente sobre o cursor do mouse; os botões da barra superior centralizam na agulha.
- **Painel Recolhível**: Pressione <kbd>W</kbd> para recolher a waveform e maximizar a visualização da playlist.

#### 3. 🎯 Detecção de HotCues & Active Learning Neural (`StructureNet`)
- **Marcação Automática**: Identifica `FIRST_BEAT (1.1)`, `INTRO`, `BUILDUP`, `DROP_1`, `BREAK_1`, `DROP_2` e `OUTRO`.
- **Arrasto Interativo com Snap Magnético**: Arraste os marcadores diretamente na onda sonora com magnetismo para as batidas da grade.
- **Treinamento Online (Active Learning)**: Ao ajustar pontos de HotCue e clicar em **`Save HotCues`**, a rede neural `StructureNet` passa por fine-tuning online no PyTorch com as suas anotações e re-exporta o modelo ONNX automaticamente.

#### 4. 📐 Calibração de Beatgrid & Primeira Batida (`Set 1.1`)
- **Definir Primeira Batida (1.1)**: Calibre o início real do compasso na posição da agulha (<kbd>Shift + M</kbd>).
- **Micro-Nudge de Grade (±5ms)**: Desloque a grade inteira em passos de 5ms com botões contínuos (`◀` / `▶`) para casar com o transiente do kick.
- **Marcadores de Frases**: Visualização de compassos de 4 batidas, seções de 16 batidas e frases EDM de 32 batidas (`1.1`, `5.1`, `9.1`, `17.1`, `33.1`).

#### 5. ⚡ Pontuação de Energia Neural (`EnergyNet`)
- Avalia energia RMS, fluxo de centroide espectral e densidade de graves para pontuar a energia da pista de `1` (Chill/Ambient) a `10` (Peak-Time Festival).

#### 6. 📁 Exportação para Rekordbox XML 5.4.3 & Tags ID3
- Exporte sua biblioteca com tom Camelot, BPM e HotCues diretamente para o **Pioneer Rekordbox** para envio a pendrives compatíveis com CDJ-2000NXS2, CDJ-3000 e Opus-Quad.
- Gravação automática de tags ID3v2 diretamente nos arquivos de áudio.

---

### 🚀 Guia Rápido de Instalação (Português)

```bash
# 1. Clonar o repositório
git clone https://github.com/hugogbrl09/AudioHarmonix.git
cd AudioHarmonix

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar o AudioHarmonix
python run_audioharmonix.py
```

---

## ⌨️ Atalhos de Teclado de DJ / Keyboard Shortcuts

| Categoria / Category | Ação / Action | Atalho / Shortcut | Descrição / Description |
|---|---|---|---|
| 🔵 **Transport** | **Play / Pause** | <kbd>Space</kbd> | Toggle audio playback at 60 FPS |
| | **Nudge / Scrub Beat** | <kbd>←</kbd> / <kbd>→</kbd> | Seek ±1 beat on beatgrid |
| | **Bar Seek** | <kbd>Shift</kbd> + <kbd>←</kbd> / <kbd>→</kbd> | Jump ±4 beats (1 full measure) |
| 🟢 **HotCues** | **Add HotCue** | <kbd>M</kbd> | Drop HotCue at playhead needle |
| | **Trigger HotCues** | <kbd>1</kbd> to <kbd>8</kbd> | Jump to HotCue pads 1 through 8 |
| 🟣 **Beatgrid** | **Set First Beat (1.1)** | <kbd>Shift</kbd> + <kbd>M</kbd> | Calibrate Beatgrid 1.1 downbeat |
| | **Toggle Beatgrid** | <kbd>B</kbd> | Show/hide 32-beat & 16-beat phrase lines |
| | **Toggle Snap** | <kbd>S</kbd> | Toggle magnetic beatgrid snapping |
| 🟠 **View** | **Zoom Waveform** | <kbd>+</kbd> / <kbd>-</kbd> / <kbd>0</kbd> | Zoom In / Zoom Out / Reset Zoom |
| | **Toggle Waveform** | <kbd>W</kbd> | Collapse/expand waveform drawer |
| ⚪ **General** | **Shortcuts Cheat Sheet** | <kbd>?</kbd> / <kbd>F1</kbd> | Open keyboard shortcuts cheatsheet |
| | **Close Modal** | <kbd>Esc</kbd> | Close any open dialog window |

---

## 🧪 Testes Automatizados / Automated Tests

```bash
python -m unittest discover -s tests
```

---

## 📄 Licença / License

Distribuído sob a licença **MIT License** - consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

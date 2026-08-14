# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

HTML5, Vanilla CSS3 (Dark Neon / DJ Audio Theme), Vanilla JavaScript (ES6+), Python Backend (FastAPI, PyTorch, ONNX Runtime, Librosa), Tauri (Desktop Packaging).

## Users

DJs profissionais e amadores que buscam preparação rápida de sets, curadoria de mixagem harmônica (Camelot Wheel), detecção precisa de tom/BPM/energia e posicionamento cirúrgico de HotCues automatizado e calibrável via IA.

## Product Purpose

O AudioHarmonix (Mixado no Tom) existe para eliminar o trabalho manual repetitivo de catalogação e marcação de faixas para DJs. Ele analisa arquivos de áudio em lote ou individualmente, prevê o tom musical exato (KeyNet), classifica o nível de energia da pista de 1 a 10 (EnergyNet), detecta a estrutura da música e insere HotCues automaticamente (StructureNet), permitindo edição interativa com aprendizado ativo online e exportação direta para o Rekordbox.

## Positioning

Diferente de analisadores estáticos convencionais (como Mixed In Key ou Rekordbox padrão), o AudioHarmonix possui um pipeline de aprendizado ativo integrado: o DJ pode ajustar manualmente os HotCues, BPM e beatgrid diretamente na Waveform, e clicar em "Salvar & Ensinar IA" para que as redes neurais calibrem seus pesos em tempo real de acordo com as preferências do DJ.

## Operating Context

- Ambiente de preparação de sets em estúdio ou em trânsito antes de apresentações.
- Interface de alto contraste escuro com feedback tátil/visual para leitura rápida sob diferentes condições de iluminação.
- Visualização de forma de onda RGB com zoom fluido de 1x a 16x, agulha de reprodução a 60 FPS com aceleração de GPU e roda harmônica Camelot interativa.
- Integração com pen drives e pastas de música locais com exportação em formato XML padrão Rekordbox.

## Capabilities and Constraints

- **Capacidades**:
  - Análise em lote multi-thread de pastas com arquivos MP3/WAV/FLAC.
  - Predição instantânea de Tom e Confiança via KeyNet ONNX (<6ms de latência).
  - Predição de Nível de Energia (1-10) com gradiente visual dinâmico via EnergyNet ONNX.
  - Predição e marcação de HotCues (First Beat, Break 1/2, Drop 1/2, Outro) via StructureNet ONNX.
  - Waveform RGB com seek milimétrico, linhas de Beatgrid (compassos 32/16/4 beats), Snap-to-grid magnético e controle de BPM (tap tempo, x2, /2).
  - Active Learning Online: ajuste fino imediato do modelo PyTorch com exportação contínua para ONNX.
  - Visualização da Roda Camelot interativa com indicação de faixas harmonicamente compatíveis.
- **Restrições**:
  - Processamento e armazenamento 100% locais (privacidade total e sem necessidade de conexão externa).
  - Interface limpa, profissional e sem emojis (utilizando apenas ícones vetoriais SVG e tipografia moderna).

## Brand Commitments

- **Nome**: AudioHarmonix (Mixado no Tom)
- **Tom de Voz**: Profissional, técnico, direto e focado em alta produtividade para DJs.
- **Identidade Visual**: Tema escuro com estética de equipamentos de DJ e estúdio profissional (fundo grafite profundo, acentos ciano neon, verde esmeralda, âmbar e vermelho coral para energia, e roxo elétrico para Camelot).

## Evidence on Hand

- Modelos ONNX compilados e otimizados (`models/key_detector.onnx`, `models/structure_detector.onnx`, `models/energy_detector.onnx`).
- Biblioteca local indexada em SQLite (`audio_tracks.db`) com faixas de teste reais.
- Servidor local HTTP em `127.0.0.1:8888` com streaming de áudio parcial HTTP 206 para scrubbing instantâneo.
- Suíte completa de testes unitários com 100% de aprovação (`tests/test_active_learning.py`, etc.).

## Product Principles

1. **Precisão Cirúrgica**: Cada HotCue e batida na Waveform deve respeitar a grade rítmica e a física do áudio.
2. **Velocidade sem Fricção**: Latência de predição em milissegundos e resposta instantânea na interface.
3. **Controle Total do Artista**: A IA auxilia e propõe, mas o DJ tem a palavra final para ajustar e ensinar o sistema.
4. **Design Impecável e Funcional**: Interfaces limpas, vetoriais, com alto contraste e legibilidade impecável sem poluição visual.

## Accessibility & Inclusion

- Alto contraste de texto e formas de onda sobre o fundo escuro (#0a0d14).
- Indicadores visuais redundantes (cor + texto/número) para Tom, Energia e Confiança.
- Atalhos e áreas clicáveis generosas com suporte a arraste e toques contínuos de precisão.

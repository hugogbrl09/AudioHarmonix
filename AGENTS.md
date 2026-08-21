# 🤖 AudioHarmonix — Manual de Orquestração Multi-Agente

> **Documentação de Referência Operacional para Agentes de IA e Desenvolvedores**  
> **Status do Projeto**: DSP Experimental CONGELADO | Legacy = DEFAULT | 30/30 Testes PASSANDO.

---

## 🏛️ 1. Estrutura dos Agentes Especializados

O AudioHarmonix utiliza uma arquitetura de **7 agentes especializados**, evitando enxames desnecessários e mantendo responsabilidades estritas:

| Agente | Identificador | Especialidade Principal | Ações Proibidas |
|---|---|---|---|
| **Orchestrator** | `audioharmonix-orchestrator` | Triagem, planejamento, delegação e fluxo de aprovação | Fazer alterações diretas sem análise de escopo |
| **DSP Specialist** | `audioharmonix-dsp` | Filtros, envelopes, beatgrid, Drops, Breaks, Buildups, HotCues | Mudar thresholds às cegas; alterar Legacy; mexer no StructureNet |
| **ML Specialist** | `audioharmonix-ml` | PyTorch, ONNX, KeyNet, StructureNet, EnergyNet | Retreinar sem justificativa; colocar StructureNet no caminho crítico |
| **Audio QA** | `audioharmonix-audio-qa` | Testes adversariais, contraexemplos, sincopação, false positives | Aceitar validação superficial com poucas faixas |
| **UI Specialist** | `audioharmonix-ui` | Waveform, timeline, pads 1..8, Tauri IPC, renderização | Modificar DSP para corrigir bugs visuais da UI |
| **Security & Perf** | `audioharmonix-security-performance` | `.env`, SQLite WAL/mmap, CPU/RAM, latência de análise | Sacrificar precisão acústica para ganhar performance |
| **Code Review** | `audioharmonix-code-review` | Auditoria independente, 9 perguntas de checklist, regressões | Escrever código de features ou aprovar sem testes |

---

## 🎯 2. Como Solicitar Tarefas aos Agentes

### 🔬 A. Solicitar Investigação DSP
```text
@audioharmonix-orchestrator Preciso de uma investigação DSP na faixa X:
1. Reproduzir o comportamento na faixa
2. Medir TransientEnergy, BeatPulse, MidHighSlope e e_bass
3. Gerar diagnóstico acústico antes de propor qualquer alteração
```

### 🧠 B. Solicitar Investigação de ML
```text
@audioharmonix-orchestrator Preciso de uma auditoria no modelo KeyNet / StructureNet:
1. Verificar acurácia no Gold Set
2. Medir latência de inferência ONNX
3. Analisar se há mismatch entre pré-processamento de treino e inferência
```

### 🥊 C. Solicitar Teste Adversarial (Audio QA)
```text
@audioharmonix-audio-qa Execute uma bateria de stress test com faixas adversariais:
1. Testar ritmos sincopados (Dubstep/DnB)
2. Testar faixas com Reese Bass contínuo
3. Gerar tabela de Falsos Positivos, Falsos Negativos e Δ temporal
```

### 🖥️ D. Solicitar Validação da UI
```text
@audioharmonix-ui Execute o aplicativo e faça a validação visual:
1. Verificar ordenação cronológica e ausência de duplicatas nos pads 1..8
2. Verificar se todos os cues estão dentro da duração da faixa
3. Testar a alternância entre modo Legacy e Experimental
```

### 🧐 E. Solicitar Code Review
```text
@audioharmonix-code-review Audite as alterações propostas:
1. Aplicar o checklist obrigatório das 9 perguntas
2. Verificar se os 30 testes unitários continuam passando
3. Confirmar que o motor Legacy não sofreu alterações
```

---

## 📜 3. Project Constitution & Regras Mandatórias

1. **Legacy = Padrão de Produção**: O modo padrão permanece estritamente `legacy`. Nenhuma promoção para default pode ser feita sem aprovação humana expressa.
2. **DSP Experimental Congelado**: O motor experimental foi auditado e validado. O próximo passo oficial é **validação manual na UI**, e não refatoração de código DSP.
3. **Princípio da Mudança Mínima**: Se um problema puder ser resolvido sem alterar código (ex: ajuste de interface ou documentação), nenhuma linha de código de produção deve ser alterada.
4. **Sem Ajustes Arbitrários de Thresholds**: É proibido alterar thresholds para fazer testes passarem sem diagnóstico físico comprovado.

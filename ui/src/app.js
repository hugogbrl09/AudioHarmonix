/**
 * AudioHarmonix Frontend Application Engine
 * Section 7: UI Component Logic, RGB Waveform Canvas, Modern Modals & Realtime API Sync
 */

const API_BASE = window.location.origin;

let tracksData = [];
let selectedTrack = null;
let currentAudio = null;
let isPlaying = false;
let batchPollInterval = null;
let pendingDeleteTrack = null;
let pendingSeekSec = null;
let simulatedCurTime = 0;
let simulatedInterval = null;

/// Waveform & Zoom & Beatgrid State
let zoomLevel = 1.0;
let showBeatgrid = true;
let snapToGrid = true;
let tapTimes = [];

// DOM Elements
const searchInput = document.getElementById("search-input");
const filterCamelot = document.getElementById("filter-camelot");
const filterEnergy = document.getElementById("filter-energy");
const trackTableBody = document.getElementById("track-table-body");
const trackCount = document.getElementById("track-count");

const waveformCanvasContainer = document.getElementById("waveform-canvas-container");
const waveformScrollWrapper = document.getElementById("waveform-scroll-wrapper");
const waveformCanvas = document.getElementById("waveform-canvas");
const playhead = document.getElementById("playhead");
const cueOverlay = document.getElementById("cue-overlay");
const selectedTitle = document.getElementById("selected-title");
const selectedMeta = document.getElementById("selected-meta");
const btnPlay = document.getElementById("btn-play");
const timeDisplay = document.getElementById("time-display");

// BPM & Toolbar DOM
const inputBpm = document.getElementById("input-bpm");
const btnBpmHalf = document.getElementById("btn-bpm-half");
const btnBpmDouble = document.getElementById("btn-bpm-double");
const btnBpmTap = document.getElementById("btn-bpm-tap");
const btnToggleBeatgrid = document.getElementById("btn-toggle-beatgrid");
const btnToggleSnap = document.getElementById("btn-toggle-snap");
const btnZoomIn = document.getElementById("btn-zoom-in");
const btnZoomOut = document.getElementById("btn-zoom-out");
const btnZoomReset = document.getElementById("btn-zoom-reset");
const zoomLevelLabel = document.getElementById("zoom-level-label");

// Grid 1.1 Alignment DOM
const btnSetFirstBeat = document.getElementById("btn-set-first-beat");
const btnGridNudgeLeft = document.getElementById("btn-grid-nudge-left");
const btnGridNudgeRight = document.getElementById("btn-grid-nudge-right");

// Waveform Collapse DOM
const waveformPanel = document.querySelector(".waveform-panel");
const btnToggleWaveformCollapse = document.getElementById("btn-toggle-waveform-collapse");

// Header Buttons
const btnImport = document.getElementById("btn-import");
const btnBatch = document.getElementById("btn-batch");
const btnExportXml = document.getElementById("btn-export-xml");

// Delete Modal DOM
const modalDeleteConfirm = document.getElementById("modal-delete-confirm");
const btnCloseDeleteModal = document.getElementById("btn-close-delete-modal");
const btnCancelDelete = document.getElementById("btn-cancel-delete");
const btnConfirmDelete = document.getElementById("btn-confirm-delete");
const deleteTrackTitleText = document.getElementById("delete-track-title-text");

// Camelot Picker Modal DOM
const modalCamelotPicker = document.getElementById("modal-camelot-picker");
const btnToggleCamelotWheel = document.getElementById("btn-toggle-camelot-wheel");
const btnCloseCamelotModal = document.getElementById("btn-close-camelot-modal");
const btnClearCamelotFilter = document.getElementById("btn-clear-camelot-filter");
const gridMinorKeys = document.getElementById("grid-minor-keys");
const gridMajorKeys = document.getElementById("grid-major-keys");

// Import Modal DOM
const modalImport = document.getElementById("modal-import");
const btnCloseImportModal = document.getElementById("btn-close-import-modal");
const dropzone = document.getElementById("dropzone");
const filePicker = document.getElementById("file-picker");
const importPathInput = document.getElementById("import-path-input");
const btnRunSingleImport = document.getElementById("btn-run-single-import");
const discoveredFilesList = document.getElementById("discovered-files-list");
const importAnalysisCard = document.getElementById("import-analysis-card");
const importStatusTitle = document.getElementById("import-status-title");
const importStatusStep = document.getElementById("import-status-step");

// Batch Modal DOM
const modalBatch = document.getElementById("modal-batch");
const btnCloseBatchModal = document.getElementById("btn-close-batch-modal");
const btnStartBatch = document.getElementById("btn-start-batch");
const batchModalFile = document.getElementById("batch-modal-file");
const batchModalPct = document.getElementById("batch-modal-pct");
const batchModalProgressBar = document.getElementById("batch-modal-progress-bar");
const batchModalCount = document.getElementById("batch-modal-count");
const batchModalSpeed = document.getElementById("batch-modal-speed");
const batchModalEta = document.getElementById("batch-modal-eta");

// Export Modal DOM
const modalExport = document.getElementById("modal-export");
const btnCloseExportModal = document.getElementById("btn-close-export-modal");
const btnConfirmExport = document.getElementById("btn-confirm-export");
const exportPathInput = document.getElementById("export-path-input");

// Shortcuts Modal DOM
const modalShortcuts = document.getElementById("modal-shortcuts");
const btnShortcuts = document.getElementById("btn-shortcuts");
const btnCloseShortcutsModal = document.getElementById("btn-close-shortcuts-modal");
const btnCloseShortcutsFooter = document.getElementById("btn-close-shortcuts-footer");

// Toast Container
const toastContainer = document.getElementById("toast-container");

// Status Footer
const batchProgressContainer = document.getElementById("batch-progress-container");
const batchProgressInner = document.getElementById("batch-progress-inner");
const batchSpeed = document.getElementById("batch-speed");
const batchCount = document.getElementById("batch-count");

const CAMELOT_MINORS = [
  { code: '1A', name: 'G#m / Abm' }, { code: '2A', name: 'D#m / Ebm' }, { code: '3A', name: 'A#m / Bbm' },
  { code: '4A', name: 'Fm' },        { code: '5A', name: 'Cm' },         { code: '6A', name: 'Gm' },
  { code: '7A', name: 'Dm' },        { code: '8A', name: 'Am' },         { code: '9A', name: 'Em' },
  { code: '10A', name: 'Bm' },       { code: '11A', name: 'F#m' },       { code: '12A', name: 'C#m' }
];

const CAMELOT_MAJORS = [
  { code: '1B', name: 'B' },  { code: '2B', name: 'F#' }, { code: '3B', name: 'Db / C#' },
  { code: '4B', name: 'Ab' }, { code: '5B', name: 'Eb' }, { code: '6B', name: 'Bb' },
  { code: '7B', name: 'F' },  { code: '8B', name: 'C' },  { code: '9B', name: 'G' },
  { code: '10B', name: 'D' }, { code: '11B', name: 'A' }, { code: '12B', name: 'E' }
];

/// Modern Toast Notification System
function showToast(message, type = "info", duration = 4000) {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${message}</span>
    <div style="cursor:pointer; margin-left:12px; display:inline-flex; align-items:center;" onclick="this.parentElement.remove()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </div>
  `;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    if (toast.parentElement) toast.remove();
  }, duration);
}

// Fetch Tracks from Backend API
async function loadTracks() {
  const search = searchInput.value;
  const camelot = filterCamelot.value;
  const energy = filterEnergy.value;

  try {
    const res = await fetch(`${API_BASE}/api/tracks?search=${encodeURIComponent(search)}&camelot=${camelot}&energy_min=${energy}`);
    const data = await res.json();
    if (data.status === "ok") {
      tracksData = data.tracks;
      renderTable(tracksData);
    }
  } catch (err) {
    console.error("Error loading tracks:", err);
    showToast("Failed to connect to AudioHarmonix backend engine.", "error");
  }
}

// Render Track Table
function renderTable(tracks) {
  trackTableBody.innerHTML = "";
  trackCount.textContent = `${tracks.length} Tracks`;

  if (tracks.length === 0) {
    trackTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:30px; color:#8a96ab;">No tracks analyzed yet. Click "Import Audio File" or "Analyze All (Batch)" to get started!</td></tr>`;
    return;
  }

  tracks.forEach((t, idx) => {
    const tr = document.createElement("tr");
    tr.className = `track-row ${selectedTrack && selectedTrack.id === t.id ? 'selected' : ''}`;

    const confVal = typeof t.key_confidence === 'number' ? t.key_confidence : 0.95;
    const confPct = Math.round(confVal * 100);
    let confColor = "#00e676"; // Green (>= 85%)
    let confBg = "rgba(0, 230, 118, 0.15)";
    if (confPct < 70) {
      confColor = "#ff5252"; // Red (< 70%)
      confBg = "rgba(255, 82, 82, 0.15)";
    } else if (confPct < 85) {
      confColor = "#ff9100"; // Orange (70% - 84%)
      confBg = "rgba(255, 145, 0, 0.15)";
    }

    const energyScore = t.energy_score || 5;
    const energyColor = getEnergyColor(energyScore);

    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>
        <div style="font-weight:600; color:var(--text-main);">${escapeHtml(t.title || t.file_name)}</div>
        <div style="font-size:11px; color:var(--text-muted);">${escapeHtml(t.artist || 'Unknown')}</div>
      </td>
      <td>
        <span style="font-family:monospace; font-weight:700;">${t.bpm ? t.bpm.toFixed(1) : '---'}</span>
        ${t.is_variable_bpm ? '<span style="font-size:10px; color:#ff9100; margin-left:4px;">(VAR)</span>' : ''}
      </td>
      <td>
        <span class="key-badge key-${t.camelot_key ? t.camelot_key.toLowerCase() : '8a'}">${t.camelot_key || '---'}</span>
        <span style="font-size:11px; color:var(--text-muted); margin-left:4px;">${t.detected_key || ''}</span>
      </td>
      <td>
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="flex:1; height:6px; background:#1b2230; border-radius:3px; overflow:hidden;">
            <div style="width:${energyScore * 10}%; height:100%; background:${energyColor};"></div>
          </div>
          <span style="font-size:11px; font-weight:700; color:${energyColor};">${energyScore}/10</span>
        </div>
      </td>
      <td>
        <span style="font-size:11px; font-weight:700; color:${confColor}; background:${confBg}; padding:2px 7px; border-radius:4px; display:inline-block;">${confPct}%</span>
      </td>
      <td>
        <div style="display:flex; gap:6px;">
          <button class="btn btn-secondary btn-table-play" style="padding:4px 8px; font-size:11px;" onclick="selectAndPlayTrack('${t.id}')">Play</button>
          <button class="btn btn-secondary btn-table-delete" style="padding:4px 8px; font-size:11px; color:#ff5252;" onclick="openDeleteModal('${t.id}', '${escapeHtml(t.title || t.file_name)}')">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
          </button>
        </div>
      </td>
    `;
    tr.addEventListener("click", (e) => {
      if (!e.target.closest("button")) {
        selectTrack(t);
      }
    });
    trackTableBody.appendChild(tr);
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function getEnergyColor(score) {
  if (score <= 3) return "#00e5ff"; // 1-3: Chill / Ambient / Intro (Cyan)
  if (score <= 5) return "#00e676"; // 4-5: Melodic / Groove (Green)
  if (score <= 7) return "#ff9100"; // 6-7: Driving Club / Mainstage (Orange)
  return "#ff1744";                // 8-10: Peak Energy / Festival Drop (Red)
}

// Prompt Delete Confirmation Modal
function openDeleteModal(id, title) {
  pendingDeleteTrack = { id, title };
  deleteTrackTitleText.textContent = `"${title}"`;
  modalDeleteConfirm.classList.remove("hidden");
}

btnCloseDeleteModal.addEventListener("click", () => {
  modalDeleteConfirm.classList.add("hidden");
  pendingDeleteTrack = null;
});

btnCancelDelete.addEventListener("click", () => {
  modalDeleteConfirm.classList.add("hidden");
  pendingDeleteTrack = null;
});

btnConfirmDelete.addEventListener("click", async () => {
  if (!pendingDeleteTrack) return;
  const { id, title } = pendingDeleteTrack;
  modalDeleteConfirm.classList.add("hidden");
  await executeDeleteTrack(id, title);
  pendingDeleteTrack = null;
});

// Delete Track Function
async function executeDeleteTrack(id, title) {
  try {
    const res = await fetch(`${API_BASE}/api/delete_track`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_id: id })
    });
    const data = await res.json();
    if (data.status === "ok") {
      showToast(`Track '${title}' removed from library.`, "info", 4000);
      if (selectedTrack && selectedTrack.id === id) {
        selectedTrack = null;
        selectedTitle.textContent = "Select a track to preview waveform";
        selectedMeta.textContent = "--- BPM | --- Key | --- Energy";
      }
      loadTracks();
    } else {
      showToast(`Delete Error: ${data.error}`, "error");
    }
  } catch (err) {
    showToast(`Error deleting track: ${err.message}`, "error");
  }
}

// Select Track & Render Waveform
function selectTrack(track) {
  selectedTrack = track;
  renderTable(tracksData);

  selectedTitle.textContent = `${track.title || track.file_name} - ${track.artist || 'Unknown Artist'}`;
  
  if (inputBpm) {
    inputBpm.value = (track.bpm ? track.bpm : 120.0).toFixed(2);
  }
  selectedMeta.textContent = `Key: ${track.camelot_key} (${track.detected_key}) | Energy: [${track.energy_score}/10]`;

  drawRGBWaveform(track);
  renderCueMarkers(track);
  renderCuePills(track);
  updateCamelotWheel(track);

  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  isPlaying = false;
  stopPlayheadAnimation();
  btnPlay.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;

  currentAudio = new Audio(`${API_BASE}/audio/?id=${track.id}`);
  currentAudio.addEventListener("ended", () => {
    isPlaying = false;
    stopPlayheadAnimation();
    btnPlay.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
    playhead.style.left = "0%";
    pendingSeekSec = null;
    simulatedCurTime = 0;
  });

  currentAudio.addEventListener("loadedmetadata", () => {
    if (pendingSeekSec !== null && currentAudio) {
      try {
        currentAudio.currentTime = pendingSeekSec;
      } catch (e) {}
    }
  });

  if (pendingSeekSec === null) {
    playhead.style.left = "0%";
    timeDisplay.textContent = `00:00 / ${formatTime(track.duration_secs || 0)}`;
  }
}

function selectAndPlayTrack(id) {
  const tr = tracksData.find(t => t.id === id);
  if (tr) {
    selectTrack(tr);
    togglePlay();
  }
}

// Update Camelot Harmonic Wheel Badges
function updateCamelotWheel(track) {
  const curKey = track.camelot_key || '8A';
  const badgeCurrent = document.getElementById("badge-current");
  const badgeRelative = document.getElementById("badge-relative");
  const badgeSubdom = document.getElementById("badge-subdom");
  const badgeDom = document.getElementById("badge-dom");

  badgeCurrent.textContent = `Current: ${curKey} (${track.detected_key || ''})`;

  const num = parseInt(curKey);
  const letter = curKey.slice(-1);
  const otherLetter = letter === "A" ? "B" : "A";

  if (!isNaN(num)) {
    const rel = `${num}${otherLetter}`;
    const subdom = `${((num - 2 + 12) % 12) + 1}${letter}`;
    const dom = `${((num % 12)) + 1}${letter}`;

    badgeRelative.textContent = `Relative: ${rel}`;
    badgeSubdom.textContent = `Subdominant: ${subdom}`;
    badgeDom.textContent = `Dominant: ${dom}`;
  }
}

// Render Camelot Wheel Interactive Grid Cells
function renderCamelotGrid() {
  gridMinorKeys.innerHTML = "";
  gridMajorKeys.innerHTML = "";

  const selectedKey = filterCamelot.value;

  CAMELOT_MINORS.forEach(k => {
    const cell = document.createElement("div");
    cell.className = `camelot-key-cell ${selectedKey === k.code ? 'active' : ''}`;
    cell.innerHTML = `
      <span class="key-code" style="color:var(--cyan-glow);">${k.code}</span>
      <span class="key-name">${k.name}</span>
    `;
    cell.addEventListener("click", () => {
      filterCamelot.value = k.code;
      modalCamelotPicker.classList.add("hidden");
      loadTracks();
      showToast(`Filtering library by Camelot key ${k.code} (${k.name})`, "info");
    });
    gridMinorKeys.appendChild(cell);
  });

  CAMELOT_MAJORS.forEach(k => {
    const cell = document.createElement("div");
    cell.className = `camelot-key-cell ${selectedKey === k.code ? 'active' : ''}`;
    cell.innerHTML = `
      <span class="key-code" style="color:var(--pink-glow);">${k.code}</span>
      <span class="key-name">${k.name}</span>
    `;
    cell.addEventListener("click", () => {
      filterCamelot.value = k.code;
      modalCamelotPicker.classList.add("hidden");
      loadTracks();
      showToast(`Filtering library by Camelot key ${k.code} (${k.name})`, "info");
    });
    gridMajorKeys.appendChild(cell);
  });
}

// Draw 3-Band RGB Waveform Canvas with Beatgrid & Dynamic Zoom (Responsive, No Right-Edge Cutoff)
function drawRGBWaveform(track) {
  if (!waveformCanvasContainer || !waveformCanvas) return;

  const containerRect = waveformCanvasContainer.getBoundingClientRect();
  const baseWidth = Math.round(containerRect.width || waveformCanvasContainer.clientWidth || 900);
  const width = Math.max(baseWidth, Math.round(baseWidth * (zoomLevel > 1.0 ? zoomLevel : 1.0)));
  const height = 100;

  if (waveformScrollWrapper) {
    waveformScrollWrapper.style.width = zoomLevel > 1.0 ? `${width}px` : "100%";
  }
  waveformCanvas.width = width;
  waveformCanvas.height = height;
  waveformCanvas.style.width = zoomLevel > 1.0 ? `${width}px` : "100%";
  waveformCanvas.style.height = `${height}px`;

  const ctx = waveformCanvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, width, height);

  // Center Line
  ctx.strokeStyle = "rgba(42, 50, 69, 0.4)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  // 1. Draw Beatgrid Lines if enabled
  if (showBeatgrid && track && (track.bpm || 0) > 0) {
    drawBeatgridLines(ctx, track, width, height);
  }

  const wf = track ? track.waveform_peaks : null;

  if (wf && wf.low && wf.low.length > 0) {
    const numPoints = wf.low.length;
    const barWidth = width / numPoints;

    for (let i = 0; i < numPoints; i++) {
      const x = i * barWidth;
      const lowVal = wf.low[i] || 0;
      const midVal = wf.mid ? wf.mid[i] || 0 : 0;
      const highVal = wf.high ? wf.high[i] || 0 : 0;

      const hLow = lowVal * (height * 0.92);
      const hMid = midVal * (height * 0.72);
      const hHigh = highVal * (height * 0.52);

      // Layer 1: Highs (Percussion & Cymbals) - Cyan
      if (hHigh > 0) {
        ctx.fillStyle = "rgba(64, 196, 255, 0.60)";
        ctx.fillRect(x, (height / 2) - (hHigh / 2), Math.max(1, barWidth - 0.4), hHigh);
      }

      // Layer 2: Mids (Vocals & Instruments) - Green
      if (hMid > 0) {
        ctx.fillStyle = "rgba(105, 240, 174, 0.70)";
        ctx.fillRect(x, (height / 2) - (hMid / 2), Math.max(1, barWidth - 0.4), hMid);
      }

      // Layer 3: Lows (Sub & Bass) - Red / Pink
      if (hLow > 0) {
        ctx.fillStyle = "rgba(255, 82, 82, 0.95)";
        ctx.fillRect(x, (height / 2) - (hLow / 2), Math.max(1, barWidth - 0.4), hLow);
      }
    }
  } else {
    ctx.fillStyle = "#8a96ab";
    ctx.font = "12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Select an analyzed track to render waveform", width / 2, height / 2 + 4);
  }
}

// Draw Beatgrid Lines with Bar Downbeats & Phrase Markers
function drawBeatgridLines(ctx, track, width, height) {
  const bpm = track.bpm || 120.0;
  const duration = track.duration_secs || 180.0;
  const secPerBeat = 60.0 / bpm;
  if (secPerBeat <= 0) return;

  let firstBeatOffset = 0.0;
  if (typeof track.first_beat_offset === "number" && !isNaN(track.first_beat_offset)) {
    firstBeatOffset = track.first_beat_offset;
  } else if (track.cues) {
    const fbCue = track.cues.find(c => (c.cue_type || '').includes('FIRST_BEAT') || (c.cue_type || '').includes('INTRO'));
    if (fbCue) firstBeatOffset = fbCue.position_secs || 0.0;
  }

  const totalBeats = Math.floor((duration - firstBeatOffset) / secPerBeat);

  ctx.save();
  ctx.textAlign = "center";

  for (let beat = 0; beat <= totalBeats; beat++) {
    const beatSec = firstBeatOffset + (beat * secPerBeat);
    if (beatSec < 0 || beatSec > duration) continue;

    const x = Math.round((beatSec / duration) * width);
    const barNum = Math.floor(beat / 4) + 1;
    const isPhraseDownbeat = (beat % 32 === 0);
    const isMajorBar = (beat % 16 === 0);
    const isBarStart = (beat % 4 === 0);

    if (isPhraseDownbeat) {
      // 32-beat phrase line (EDM major section change) - Purple neon
      ctx.strokeStyle = "rgba(213, 0, 249, 0.85)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      ctx.fillStyle = "#d500f9";
      ctx.font = "bold 9px monospace";
      ctx.fillText(`${barNum}.1`, x, 11);
    } else if (isMajorBar) {
      // 16-beat bar line - Cyan neon
      ctx.strokeStyle = "rgba(0, 229, 255, 0.75)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      ctx.fillStyle = "#00e5ff";
      ctx.font = "bold 9px monospace";
      ctx.fillText(`${barNum}.1`, x, 11);
    } else if (isBarStart) {
      // Regular 4-beat Bar line
      ctx.strokeStyle = "rgba(255, 255, 255, 0.40)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      if (zoomLevel >= 2.0) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
        ctx.font = "8px monospace";
        ctx.fillText(`${barNum}`, x, 10);
      }
    } else {
      // Sub-beat lines (visible at higher zoom levels)
      if (zoomLevel >= 1.5) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, 14);
        ctx.lineTo(x, height - 14);
        ctx.stroke();
      }
    }
  }

  ctx.restore();
}

// Snap timestamp to nearest beatgrid beat
function snapTimestampToBeat(sec, track) {
  if (!snapToGrid || !track || !track.bpm || track.bpm <= 0) return sec;
  const bpm = track.bpm;
  const secPerBeat = 60.0 / bpm;
  let firstBeatOffset = 0.0;
  if (typeof track.first_beat_offset === "number" && !isNaN(track.first_beat_offset)) {
    firstBeatOffset = track.first_beat_offset;
  } else if (track.cues) {
    const fbCue = track.cues.find(c => (c.cue_type || '').includes('FIRST_BEAT') || (c.cue_type || '').includes('INTRO'));
    if (fbCue) firstBeatOffset = fbCue.position_secs || 0.0;
  }
  const beatIndex = Math.round((sec - firstBeatOffset) / secPerBeat);
  const snapped = parseFloat((firstBeatOffset + (beatIndex * secPerBeat)).toFixed(3));
  return Math.max(0, Math.min(track.duration_secs || 180, snapped));
}

// Enhanced Zoom Management with Cursor & Needle Anchor Positioning
function setZoom(newZoom, anchorOptions = {}) {
  const oldZoom = zoomLevel;
  const targetZoom = Math.max(1.0, Math.min(16.0, parseFloat(newZoom.toFixed(1))));
  if (targetZoom === oldZoom && zoomLevel === targetZoom) return;

  const duration = (selectedTrack ? selectedTrack.duration_secs : 180) || 180;
  const container = waveformCanvasContainer;
  const viewportWidth = container ? container.clientWidth : 900;
  const oldTotalWidth = waveformCanvas ? waveformCanvas.width : viewportWidth;

  // Determine normalized anchor point [0, 1]
  let anchorPct = 0;
  let viewportOffsetPx = viewportWidth / 2;

  if (anchorOptions.mouseClientX && container) {
    // 1. Mouse-anchored Zoom (Ctrl + Wheel): keep the audio timestamp under the mouse stationary
    const rect = container.getBoundingClientRect();
    viewportOffsetPx = anchorOptions.mouseClientX - rect.left;
    const oldAbsoluteX = container.scrollLeft + viewportOffsetPx;
    anchorPct = oldTotalWidth > 0 ? (oldAbsoluteX / oldTotalWidth) : 0;
  } else {
    // 2. Needle/Playhead-anchored Zoom (Buttons + / - / 0 and Keyboard shortcuts): center the needle
    const curPlayheadSec = getCurrentPlayheadSec();
    anchorPct = Math.max(0, Math.min(1, curPlayheadSec / duration));
    viewportOffsetPx = viewportWidth / 2;
  }

  anchorPct = Math.max(0, Math.min(1, anchorPct));

  // Apply new zoom level
  zoomLevel = targetZoom;
  if (zoomLevelLabel) zoomLevelLabel.textContent = `${zoomLevel.toFixed(1)}x`;

  if (selectedTrack) {
    drawRGBWaveform(selectedTrack);
    renderCueMarkers(selectedTrack);
  }

  // Adjust scrollLeft so anchor point remains exactly in place
  if (container && waveformCanvas) {
    const newTotalWidth = waveformCanvas.width;
    const newAnchorAbsoluteX = anchorPct * newTotalWidth;
    const targetScrollLeft = newAnchorAbsoluteX - viewportOffsetPx;
    container.scrollLeft = Math.max(0, Math.min(newTotalWidth - viewportWidth, targetScrollLeft));
  }
}

// HotCue Studio & Active Learning DOM Elements
const hotcuePillsContainer = document.getElementById("hotcue-pills-container");
const btnAddCuePlayhead = document.getElementById("btn-add-cue-playhead");
const btnSaveTeachAi = document.getElementById("btn-save-teach-ai");

const CUE_TYPES = [
  "FIRST_BEAT", "INTRO", "BUILDUP", "DROP_1", "BREAK_1", 
  "DROP_2", "BREAK_2", "DROP_3", "VERSE", "OUTRO"
];

let activeDraggedCue = null;
let activeDraggedEl = null;

// Render Cue Points Markers Overlay with Interactive Drag & Drop
function renderCueMarkers(track) {
  cueOverlay.innerHTML = "";
  if (!track || !track.cues) return;

  const duration = track.duration_secs || 180;
  const validCues = track.cues.filter(c => typeof c.position_secs === 'number' && c.position_secs >= 0 && c.position_secs <= duration);

  validCues.forEach((c, idx) => {
    const pct = Math.min(100, Math.max(0, (c.position_secs / duration) * 100));
    const cueTypeClass = (c.cue_type || "cue").toLowerCase();

    const marker = document.createElement("div");
    marker.className = `cue-marker cue-${cueTypeClass}`;
    marker.style.left = `${pct}%`;
    marker.title = `HotCue ${idx + 1}: ${c.cue_type} (${formatTime(c.position_secs)}) - Drag to reposition`;

    const label = document.createElement("div");
    label.className = `cue-label cue-${cueTypeClass}`;
    label.style.left = `${pct}%`;
    label.textContent = `${c.cue_type || "CUE"} ${formatTime(c.position_secs)}`;
    label.title = "Drag to reposition";

    const startDrag = (e) => {
      e.stopPropagation();
      e.preventDefault();
      activeDraggedCue = c;
      activeDraggedEl = { marker, label };
      marker.classList.add("dragging");
      label.classList.add("dragging");

      const scrollWrapper = document.getElementById("waveform-scroll-wrapper");
      const container = document.getElementById("waveform-canvas-container");

      const onMouseMove = (moveEvent) => {
        if (!activeDraggedCue) return;
        const wrapperRect = scrollWrapper ? scrollWrapper.getBoundingClientRect() : (container ? container.getBoundingClientRect() : null);
        if (!wrapperRect || wrapperRect.width <= 0) return;

        const curX = moveEvent.clientX - wrapperRect.left;
        let newPct = Math.min(1, Math.max(0, curX / wrapperRect.width));
        let newSec = parseFloat((newPct * duration).toFixed(3));

        if (snapToGrid) {
          newSec = snapTimestampToBeat(newSec, track);
          newPct = Math.min(1, Math.max(0, newSec / duration));
        }

        activeDraggedCue.position_secs = newSec;
        marker.style.left = `${newPct * 100}%`;
        label.style.left = `${newPct * 100}%`;
        label.textContent = `${activeDraggedCue.cue_type} ${formatTime(newSec)}`;

        // Update corresponding pill time in real time if visible
        const cueIdx = track.cues.indexOf(activeDraggedCue);
        if (cueIdx >= 0 && hotcuePillsContainer) {
          const pills = hotcuePillsContainer.querySelectorAll('.cue-pill');
          if (pills[cueIdx]) {
            const timeEl = pills[cueIdx].querySelector('.cue-pill-time');
            if (timeEl) timeEl.textContent = formatTime(newSec);
          }
        }

        // Real-time audio seek preview while dragging (preserves playback state)
        seekToPosition(newPct);
      };

      const onMouseUp = () => {
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
        if (activeDraggedCue) {
          marker.classList.remove("dragging");
          label.classList.remove("dragging");
          activeDraggedCue = null;
          activeDraggedEl = null;

          // Re-sort cues by timestamp and re-render
          track.cues.sort((a, b) => (a.position_secs || 0) - (b.position_secs || 0));
          renderCueMarkers(track);
          renderCuePills(track);
          showToast(`HotCue repositioned to ${formatTime(c.position_secs)}. Click 'Save HotCues' to persist and adapt AI.`, "info", 3500);
        }
      };

      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
    };

    marker.addEventListener("mousedown", startDrag);
    label.addEventListener("mousedown", startDrag);

    cueOverlay.appendChild(marker);
    cueOverlay.appendChild(label);
  });
}

// Setup continuous hold-to-scroll on nudge buttons (-0.1s / +0.1s)
function setupContinuousNudge(buttonEl, delta, cue, track) {
  let holdTimeout = null;
  let repeatInterval = null;
  let holdStartTime = 0;

  const performStep = () => {
    const dur = track.duration_secs || 180;
    const elapsed = Date.now() - holdStartTime;
    // Progressive acceleration when holding for over 1.2s
    const stepMultiplier = elapsed > 2000 ? 3.0 : (elapsed > 1000 ? 1.8 : 1.0);
    const step = delta * stepMultiplier;

    cue.position_secs = Math.min(dur, Math.max(0, parseFloat((cue.position_secs + step).toFixed(3))));

    // Smooth real-time update of marker and time label
    const pct = (cue.position_secs / dur) * 100;
    const pillTime = buttonEl.closest('.cue-pill')?.querySelector('.cue-pill-time');
    if (pillTime) pillTime.textContent = formatTime(cue.position_secs);

    renderCueMarkers(track);
    seekToPosition(cue.position_secs / dur);
  };

  const startHold = (e) => {
    e.stopPropagation();
    e.preventDefault();
    holdStartTime = Date.now();

    performStep();

    holdTimeout = setTimeout(() => {
      repeatInterval = setInterval(() => {
        performStep();
      }, 50);
    }, 220);
  };

  const stopHold = () => {
    if (holdTimeout) {
      clearTimeout(holdTimeout);
      holdTimeout = null;
    }
    if (repeatInterval) {
      clearInterval(repeatInterval);
      repeatInterval = null;
    }
    track.cues.sort((a, b) => (a.position_secs || 0) - (b.position_secs || 0));
    renderCueMarkers(track);
    renderCuePills(track);
  };

  buttonEl.addEventListener("mousedown", startHold);
  buttonEl.addEventListener("mouseup", stopHold);
  buttonEl.addEventListener("mouseleave", stopHold);
  buttonEl.addEventListener("touchstart", startHold, { passive: false });
  buttonEl.addEventListener("touchend", stopHold);
  buttonEl.addEventListener("touchcancel", stopHold);
}

// Render Interactive HotCue Studio Pills with Quick Controls
function renderCuePills(track) {
  if (!hotcuePillsContainer) return;
  hotcuePillsContainer.innerHTML = "";

  if (!track || !track.cues || track.cues.length === 0) {
    hotcuePillsContainer.innerHTML = `<span class="no-cues-text">No HotCues set for this track. Click "Add Cue at Playhead" or press 'M' to create one.</span>`;
    return;
  }

  const duration = track.duration_secs || 180;
  const validCues = track.cues.filter(c => typeof c.position_secs === 'number' && c.position_secs >= 0 && c.position_secs <= duration);

  if (validCues.length === 0) {
    hotcuePillsContainer.innerHTML = `<span class="no-cues-text">No valid HotCues found. Click "Add Cue at Playhead" or press 'M' to create.</span>`;
    return;
  }

  validCues.forEach((c, idx) => {
    const cueTypeClass = (c.cue_type || "cue").toLowerCase();
    const pill = document.createElement("div");
    pill.className = `cue-pill`;
    
    pill.innerHTML = `
      <span class="cue-pill-badge cue-${cueTypeClass}">${idx + 1}</span>
      <select class="cue-type-select" style="background:transparent; border:none; color:var(--text-main); font-size:11px; font-weight:700; cursor:pointer;">
        ${CUE_TYPES.map(t => `<option value="${t}" ${c.cue_type === t ? 'selected' : ''} style="background:#131823; color:#fff;">${t}</option>`).join('')}
      </select>
      <span class="cue-pill-time" title="Click to trigger playback at this HotCue">${formatTime(c.position_secs)}</span>
      <div class="cue-pill-nudge">
        <button class="btn-nudge btn-nudge-left" title="Hold to scrub backward continuously">-0.1s</button>
        <button class="btn-nudge btn-nudge-right" title="Hold to scrub forward continuously">+0.1s</button>
      </div>
      <button class="btn-delete-cue" title="Remove HotCue">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    `;

    // Jump to cue position on clicking time or badge (maintains current play/pause state)
    const onCueJump = (e) => {
      e.stopPropagation();
      const dur = track.duration_secs || 180;
      seekToPosition(c.position_secs / dur);
    };

    pill.querySelector(".cue-pill-badge").addEventListener("click", onCueJump);
    pill.querySelector(".cue-pill-time").addEventListener("click", onCueJump);

    // Change cue type
    pill.querySelector(".cue-type-select").addEventListener("change", (e) => {
      c.cue_type = e.target.value;
      renderCueMarkers(track);
      renderCuePills(track);
    });

    // Continuous Hold Nudge Left (-0.1s)
    const btnLeft = pill.querySelector(".btn-nudge-left");
    setupContinuousNudge(btnLeft, -0.1, c, track);

    // Continuous Hold Nudge Right (+0.1s)
    const btnRight = pill.querySelector(".btn-nudge-right");
    setupContinuousNudge(btnRight, +0.1, c, track);

    // Delete cue
    pill.querySelector(".btn-delete-cue").addEventListener("click", (e) => {
      e.stopPropagation();
      const realIndex = track.cues.indexOf(c);
      if (realIndex !== -1) {
        track.cues.splice(realIndex, 1);
      }
      renderCueMarkers(track);
      renderCuePills(track);
      showToast(`HotCue removed. Click 'Save & Train AI' to persist changes.`, "info", 2500);
    });

    hotcuePillsContainer.appendChild(pill);
  });
}

// Helper to accurately resolve the current needle/playhead position in seconds
function getCurrentPlayheadSec() {
  if (!selectedTrack) return 0;
  const duration = selectedTrack.duration_secs || 180;

  // 1. If currently playing real audio and time is moving:
  if (currentAudio && !currentAudio.paused && !isNaN(currentAudio.currentTime) && currentAudio.currentTime > 0) {
    return currentAudio.currentTime;
  }

  // 2. If user seeked or clicked anywhere on the waveform:
  if (pendingSeekSec !== null && !isNaN(pendingSeekSec) && pendingSeekSec > 0) {
    return pendingSeekSec;
  }

  // 3. If simulated playback is actively running or was positioned:
  if (simulatedCurTime > 0 && !isNaN(simulatedCurTime)) {
    return simulatedCurTime;
  }

  // 4. If currentAudio exists and has a non-zero timestamp:
  if (currentAudio && !isNaN(currentAudio.currentTime) && currentAudio.currentTime > 0) {
    return currentAudio.currentTime;
  }

  // 5. Read visual playhead element position (% of container width):
  if (playhead && playhead.style.left && playhead.style.left !== "0%") {
    const pct = parseFloat(playhead.style.left) / 100.0;
    if (!isNaN(pct) && pct > 0) {
      return pct * duration;
    }
  }

  return 0;
}

// Add HotCue at current Playhead position
function addCueAtPlayhead() {
  if (!selectedTrack) {
    showToast("Select a track first to add HotCues.", "error");
    return;
  }

  const duration = selectedTrack.duration_secs || 180;
  let curSec = getCurrentPlayheadSec();

  // If snapToGrid is active, quantize to nearest beat on the beatgrid
  if (snapToGrid) {
    curSec = snapTimestampToBeat(curSec, selectedTrack);
  }

  curSec = Math.max(0, Math.min(duration, parseFloat(curSec.toFixed(3))));

  if (!selectedTrack.cues) selectedTrack.cues = [];

  // Suggest intelligent cue label based on position & existing cues
  const numExistingDrops = selectedTrack.cues.filter(c => (c.cue_type || "").includes("DROP")).length;
  const numExistingBreaks = selectedTrack.cues.filter(c => (c.cue_type || "").includes("BREAK")).length;

  let suggestedType = "DROP_1";
  if (selectedTrack.cues.length === 0 || curSec < 5.0) {
    suggestedType = "FIRST_BEAT";
  } else if (numExistingDrops === 1) {
    suggestedType = "BREAK_1";
  } else if (numExistingDrops > 1 && numExistingBreaks > 0) {
    suggestedType = `DROP_${numExistingDrops + 1}`;
  } else if (numExistingDrops > 0) {
    suggestedType = `DROP_${numExistingDrops + 1}`;
  }

  // Check if a cue already exists very close to this position (< 0.15s)
  const existingNear = selectedTrack.cues.find(c => Math.abs(c.position_secs - curSec) < 0.15);
  if (existingNear) {
    showToast(`A HotCue (${existingNear.cue_type}) already exists at ${formatTime(curSec)}.`, "info", 2500);
    return;
  }

  selectedTrack.cues.push({
    cue_type: suggestedType,
    position_secs: curSec,
    hotcue_num: selectedTrack.cues.length + 1
  });

  selectedTrack.cues.sort((a, b) => (a.position_secs || 0) - (b.position_secs || 0));

  renderCueMarkers(selectedTrack);
  renderCuePills(selectedTrack);

  showToast(`New HotCue ${suggestedType} added at ${formatTime(curSec)}.`, "info", 3000);
}

// Save Cues & Persist Grid Markers
async function saveAndTeachAI() {
  if (!selectedTrack) {
    showToast("Select a track to save changes.", "error");
    return;
  }

  const btn = document.getElementById("btn-save-teach-ai");
  const origHtml = btn.innerHTML;
  btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 10 10"/></svg> <span>Saving...</span>`;
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/save_user_cues`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        track_id: selectedTrack.id,
        cues: selectedTrack.cues || [],
        bpm: selectedTrack.bpm || 120.0
      })
    });

    const data = await res.json();
    if (data.status === "ok") {
      showToast("Beatgrid, HotCues and AI model calibrated successfully!", "success", 4000);
      if (data.cues) {
        selectedTrack.cues = data.cues;
        renderCueMarkers(selectedTrack);
        renderCuePills(selectedTrack);
      }
      if (data.bpm) {
        selectedTrack.bpm = parseFloat(data.bpm);
        if (inputBpm) inputBpm.value = selectedTrack.bpm.toFixed(2);
        drawRGBWaveform(selectedTrack);
      }
    } else {
      showToast(`Save error: ${data.error || 'Unknown'}`, "error");
    }
  } catch (err) {
    showToast(`Backend connection error: ${err.message}`, "error");
  } finally {
    btn.innerHTML = origHtml;
    btn.disabled = false;
  }
}

let isSeeking = false;
let seekingTimeout = null;

// Audio Seeking Helper (with Snap-to-Grid support)
function seekToPosition(pct) {
  if (!selectedTrack) return;
  const duration = selectedTrack.duration_secs || 180;
  let targetSec = pct * duration;

  if (snapToGrid) {
    targetSec = snapTimestampToBeat(targetSec, selectedTrack);
    pct = Math.max(0, Math.min(1, targetSec / duration));
  }

  isSeeking = true;
  if (seekingTimeout) clearTimeout(seekingTimeout);
  seekingTimeout = setTimeout(() => { isSeeking = false; }, 400);

  pendingSeekSec = targetSec;
  simulatedCurTime = targetSec;

  playhead.style.left = `${pct * 100}%`;
  timeDisplay.textContent = `${formatTime(targetSec)} / ${formatTime(duration)}`;

  if (currentAudio) {
    try {
      currentAudio.currentTime = targetSec;
    } catch (e) {
      console.log("Deferred seek until metadata loads...");
    }
  }
}

// Ultra-Smooth 60 FPS RequestAnimationFrame Playhead Animation Engine
let playheadAnimFrame = null;
let lastSimulatedTime = 0;

function startPlayheadAnimation() {
  stopPlayheadAnimation();
  lastSimulatedTime = performance.now();

  function animate(now) {
    if (!isPlaying) return;

    if (currentAudio && !currentAudio.paused && !isSeeking) {
      const curTime = currentAudio.currentTime;
      const duration = currentAudio.duration || (selectedTrack ? selectedTrack.duration_secs : 180) || 180;
      if (duration > 0 && !isNaN(curTime)) {
        const pct = Math.min(100, Math.max(0, (curTime / duration) * 100));
        playhead.style.left = `${pct}%`;
        timeDisplay.textContent = `${formatTime(curTime)} / ${formatTime(duration)}`;

        // Smooth continuous auto-scroll when zoomed in
        if (zoomLevel > 1.0 && waveformCanvasContainer && waveformCanvas) {
          const totalWidth = waveformCanvas.width;
          const playheadPx = (pct / 100) * totalWidth;
          const halfView = waveformCanvasContainer.clientWidth / 2;
          waveformCanvasContainer.scrollLeft = playheadPx - halfView;
        }
      }
    } else if (!isSeeking && isPlaying) {
      const dt = (now - lastSimulatedTime) / 1000.0;
      const dur = (selectedTrack ? selectedTrack.duration_secs : 180) || 180;
      simulatedCurTime += dt;
      if (simulatedCurTime >= dur) {
        simulatedCurTime = 0;
        isPlaying = false;
        btnPlay.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
        playhead.style.left = "0%";
        timeDisplay.textContent = `00:00 / ${formatTime(dur)}`;
        return;
      }
      const pct = Math.min(100, Math.max(0, (simulatedCurTime / dur) * 100));
      playhead.style.left = `${pct}%`;
      timeDisplay.textContent = `${formatTime(simulatedCurTime)} / ${formatTime(dur)}`;

      if (zoomLevel > 1.0 && waveformCanvasContainer && waveformCanvas) {
        const totalWidth = waveformCanvas.width;
        const playheadPx = (pct / 100) * totalWidth;
        const halfView = waveformCanvasContainer.clientWidth / 2;
        waveformCanvasContainer.scrollLeft = playheadPx - halfView;
      }
    }

    lastSimulatedTime = now;
    if (isPlaying) {
      playheadAnimFrame = requestAnimationFrame(animate);
    }
  }

  playheadAnimFrame = requestAnimationFrame(animate);
}

function stopPlayheadAnimation() {
  if (playheadAnimFrame) {
    cancelAnimationFrame(playheadAnimFrame);
    playheadAnimFrame = null;
  }
}

// Audio Playback
function togglePlay() {
  if (!selectedTrack) return;

  if (!currentAudio) {
    currentAudio = new Audio(`${API_BASE}/audio/?id=${selectedTrack.id}`);
    currentAudio.addEventListener("ended", () => {
      isPlaying = false;
      stopPlayheadAnimation();
      btnPlay.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
      playhead.style.left = "0%";
      pendingSeekSec = null;
      simulatedCurTime = 0;
    });

    currentAudio.addEventListener("loadedmetadata", () => {
      if (pendingSeekSec !== null && currentAudio) {
        try {
          currentAudio.currentTime = pendingSeekSec;
        } catch (e) {}
      }
    });
  }

  if (isPlaying) {
    if (currentAudio) currentAudio.pause();
    isPlaying = false;
    stopPlayheadAnimation();
    btnPlay.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
  } else {
    if (pendingSeekSec !== null) {
      try {
        currentAudio.currentTime = pendingSeekSec;
      } catch (e) {}
    }

    currentAudio.play().then(() => {
      isPlaying = true;
      btnPlay.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
      startPlayheadAnimation();
      if (pendingSeekSec !== null) {
        try {
          currentAudio.currentTime = pendingSeekSec;
        } catch (e) {}
      }
    }).catch(err => {
      console.log("Audio playback notice, using smooth client simulation:", err);
      isPlaying = true;
      btnPlay.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
      if (pendingSeekSec !== null) {
        simulatedCurTime = pendingSeekSec;
        pendingSeekSec = null;
      }
      startPlayheadAnimation();
    });
  }
}

function updatePlayhead() {
  // Maintained for seek event sync
  if (!isPlaying || !currentAudio || isSeeking) return;
  const duration = currentAudio.duration || (selectedTrack ? selectedTrack.duration_secs : 180) || 180;
  if (isNaN(currentAudio.currentTime) || duration <= 0) return;
  const pct = Math.min(100, Math.max(0, (currentAudio.currentTime / duration) * 100));
  playhead.style.left = `${pct}%`;
  timeDisplay.textContent = `${formatTime(currentAudio.currentTime)} / ${formatTime(duration)}`;
}

function formatTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// Single File Analysis API Trigger
async function triggerSingleAnalysis(filePath, fileName = null, fileBlob = null) {
  importAnalysisCard.classList.remove("hidden");
  importStatusTitle.textContent = `Analyzing ${fileName || filePath}...`;
  importStatusStep.textContent = "Decoding PCM Audio & Resampling to 22,050 Hz...";

  try {
    let res;
    if (fileBlob) {
      const base64Data = await fileToBase64(fileBlob);
      importStatusStep.textContent = "Running CQT Transform & Neural Key Detector...";
      const response = await fetch(`${API_BASE}/api/upload_and_analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_name: fileName, base64_data: base64Data })
      });
      res = await response.json();
    } else {
      importStatusStep.textContent = "Running CQT Transform & Neural Key Detector...";
      const response = await fetch(`${API_BASE}/api/analyze_file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: filePath })
      });
      res = await response.json();
    }

    importAnalysisCard.classList.add("hidden");

    if (res.status === "ok") {
      showToast(`Successfully analyzed '${res.result.file_name}'! Key: ${res.result.camelot_key} (${res.result.detected_key}) | BPM: ${res.result.bpm.toFixed(1)}`, "success", 5000);
      modalImport.classList.add("hidden");
      await loadTracks();
      const newTrack = tracksData.find(t => t.id === res.result.track_id);
      if (newTrack) selectTrack(newTrack);
    } else {
      showToast(`Analysis Error: ${res.error || 'Failed processing audio'}`, "error", 5000);
    }
  } catch (err) {
    importAnalysisCard.classList.add("hidden");
    showToast(`Error analyzing file: ${err.message}`, "error", 5000);
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });
}

// Scan & Render Discovered Local Files
async function scanDiscoveredFiles() {
  try {
    const res = await fetch(`${API_BASE}/api/scan_files`);
    const data = await res.json();
    if (data.status === "ok" && data.files) {
      discoveredFilesList.innerHTML = "";
      if (data.files.length === 0) {
        discoveredFilesList.innerHTML = `<div style="font-size:11px; color:#8a96ab; padding:4px;">No unanalyzed tracks found in workspace.</div>`;
        return;
      }

      data.files.forEach(fp => {
        const fname = fp.split(/[\\/]/).pop();
        const chip = document.createElement("div");
        chip.className = "file-item-chip";
        chip.innerHTML = `
          <span style="display:inline-flex; align-items:center; gap:6px;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--cyan-glow)" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg> <strong>${fname}</strong> <span style="font-size:10px; color:#8a96ab;">(${fp})</span></span>
          <button class="btn btn-secondary" style="padding:2px 8px; font-size:10px;">Analyze</button>
        `;
        chip.querySelector("button").addEventListener("click", (e) => {
          e.stopPropagation();
          triggerSingleAnalysis(fp, fname);
        });
        discoveredFilesList.appendChild(chip);
      });
    }
  } catch (err) {
    console.error("Scan files error:", err);
  }
}

// Camelot Modal Events
btnToggleCamelotWheel.addEventListener("click", () => {
  renderCamelotGrid();
  modalCamelotPicker.classList.remove("hidden");
});

btnCloseCamelotModal.addEventListener("click", () => {
  modalCamelotPicker.classList.add("hidden");
});

btnClearCamelotFilter.addEventListener("click", () => {
  filterCamelot.value = "";
  modalCamelotPicker.classList.add("hidden");
  loadTracks();
  showToast("Cleared Camelot key filter.", "info");
});

// Import Modal Handlers
btnImport.addEventListener("click", () => {
  modalImport.classList.remove("hidden");
  scanDiscoveredFiles();
});

btnCloseImportModal.addEventListener("click", () => {
  modalImport.classList.add("hidden");
  importAnalysisCard.classList.add("hidden");
});

dropzone.addEventListener("click", () => filePicker.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    const file = e.dataTransfer.files[0];
    triggerSingleAnalysis(file.name, file.name, file);
  }
});

filePicker.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    const file = e.target.files[0];
    triggerSingleAnalysis(file.name, file.name, file);
  }
});

btnRunSingleImport.addEventListener("click", () => {
  const pathVal = importPathInput.value.trim();
  if (pathVal) {
    triggerSingleAnalysis(pathVal, pathVal);
  } else {
    showToast("Please enter a valid file path or select a file.", "error");
  }
});

// Batch Analysis Modal Handlers
btnBatch.addEventListener("click", () => {
  modalBatch.classList.remove("hidden");
  batchModalFile.textContent = "Ready to start batch analysis...";
  batchModalPct.textContent = "0%";
  batchModalProgressBar.style.width = "0%";
  batchModalCount.textContent = "0 / 0 tracks";
  batchModalSpeed.textContent = "0.0 tracks/sec";
  batchModalEta.textContent = "ETA: --s";
});

btnCloseBatchModal.addEventListener("click", () => {
  modalBatch.classList.add("hidden");
  if (batchPollInterval) clearInterval(batchPollInterval);
});

btnStartBatch.addEventListener("click", async () => {
  try {
    const scanRes = await fetch(`${API_BASE}/api/scan_files`);
    const scanData = await scanRes.json();
    const filesToAnalyze = scanData.files || [];

    if (filesToAnalyze.length === 0) {
      showToast("No audio files found on disk to batch analyze.", "error");
      return;
    }

    const res = await fetch(`${API_BASE}/api/analyze_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_paths: filesToAnalyze })
    });
    const data = await res.json();

    if (data.status === "ok") {
      showToast(`Started Multi-Threaded Batch Analysis for ${data.total} tracks!`, "info");
      btnStartBatch.disabled = true;
      btnStartBatch.textContent = "Analyzing Batch...";
      startBatchPolling();
    } else {
      showToast(`Batch Error: ${data.error}`, "error");
    }
  } catch (err) {
    showToast(`Error initiating batch: ${err.message}`, "error");
  }
});

function startBatchPolling() {
  if (batchPollInterval) clearInterval(batchPollInterval);
  batchProgressContainer.classList.remove("hidden");

  batchPollInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/batch_status`);
      const data = await res.json();
      if (data.status === "ok" && data.batch) {
        const b = data.batch;
        const total = b.total_files || 1;
        const done = b.processed_files || 0;
        const pct = Math.round((done / total) * 100);

        batchModalFile.textContent = b.current_file ? `Analyzing: ${b.current_file}` : 'In progress...';
        batchModalPct.textContent = `${pct}%`;
        batchModalProgressBar.style.width = `${pct}%`;
        batchModalCount.textContent = `${done} / ${total} tracks`;
        batchModalSpeed.textContent = `${b.tracks_per_sec || 0} tracks/sec`;
        batchModalEta.textContent = `ETA: ${b.eta_seconds || 0}s`;

        batchProgressInner.style.width = `${pct}%`;
        batchSpeed.textContent = `${b.tracks_per_sec || 0} tracks/sec`;
        batchCount.textContent = `${done} / ${total}`;

        if (!b.is_running && done >= total && total > 0) {
          clearInterval(batchPollInterval);
          btnStartBatch.disabled = false;
          btnStartBatch.textContent = "Start Batch Analysis";
          showToast(`Batch Analysis Completed! Analyzed ${total} tracks successfully.`, "success", 5000);
          modalBatch.classList.add("hidden");
          batchProgressContainer.classList.add("hidden");
          await loadTracks();
        }
      }
    } catch (err) {
      console.error("Batch poll error:", err);
    }
  }, 400);
}

// Rekordbox XML Export Modal
btnExportXml.addEventListener("click", () => {
  modalExport.classList.remove("hidden");
});

btnCloseExportModal.addEventListener("click", () => {
  modalExport.classList.add("hidden");
});

btnConfirmExport.addEventListener("click", async () => {
  const outPath = exportPathInput.value.trim() || "rekordbox.xml";
  try {
    const res = await fetch(`${API_BASE}/api/export_rekordbox`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output_path: outPath })
    });
    const data = await res.json();
    if (data.status === "ok") {
      showToast(data.message, "success", 5000);
      modalExport.classList.add("hidden");
    } else {
      showToast(`Export Error: ${data.message}`, "error");
    }
  } catch (err) {
    showToast(`Export error: ${err.message}`, "error");
  }
});

// Waveform Canvas & Container Seek Event Registration
function initSeekListeners() {
  const container = document.getElementById("waveform-canvas-container");
  const wrapper = document.getElementById("waveform-scroll-wrapper");
  const canvas = document.getElementById("waveform-canvas");

  const onSeekClick = (e) => {
    if (e.target.closest(".cue-marker") || e.target.closest(".cue-label") || e.target.closest("button") || e.target.closest("select") || e.target.closest("input")) {
      return;
    }
    const targetEl = wrapper || canvas || container;
    if (!targetEl) return;
    const rect = targetEl.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.min(1, Math.max(0, clickX / rect.width));
    seekToPosition(pct);
  };

  if (wrapper) wrapper.onclick = onSeekClick;
  else if (canvas) canvas.onclick = onSeekClick;
  else if (container) container.onclick = onSeekClick;
}

// Window Resize Auto-Refit
window.addEventListener("resize", () => {
  if (selectedTrack) {
    drawRGBWaveform(selectedTrack);
    renderCueMarkers(selectedTrack);
  }
});

// Filter & Play Event Listeners
searchInput.addEventListener("input", loadTracks);
filterCamelot.addEventListener("change", loadTracks);
filterEnergy.addEventListener("change", loadTracks);

btnPlay.addEventListener("click", togglePlay);

if (btnAddCuePlayhead) {
  btnAddCuePlayhead.addEventListener("click", addCueAtPlayhead);
}

if (btnSaveTeachAi) {
  btnSaveTeachAi.addEventListener("click", saveAndTeachAI);
}

// BPM Editor Event Listeners
if (inputBpm) {
  inputBpm.addEventListener("change", () => {
    if (!selectedTrack) return;
    const newBpm = parseFloat(inputBpm.value);
    if (!isNaN(newBpm) && newBpm >= 40 && newBpm <= 260) {
      selectedTrack.bpm = newBpm;
      drawRGBWaveform(selectedTrack);
      showToast(`BPM calibrated to ${newBpm.toFixed(2)}. Beatgrid recalculated!`, "info", 2500);
    }
  });
}

if (btnBpmHalf) {
  btnBpmHalf.addEventListener("click", () => {
    if (!selectedTrack || !selectedTrack.bpm) return;
    selectedTrack.bpm = parseFloat((selectedTrack.bpm / 2).toFixed(2));
    if (inputBpm) inputBpm.value = selectedTrack.bpm.toFixed(2);
    drawRGBWaveform(selectedTrack);
    showToast(`BPM divided by 2 (/2): ${selectedTrack.bpm} BPM`, "info", 2500);
  });
}

if (btnBpmDouble) {
  btnBpmDouble.addEventListener("click", () => {
    if (!selectedTrack || !selectedTrack.bpm) return;
    selectedTrack.bpm = parseFloat((selectedTrack.bpm * 2).toFixed(2));
    if (inputBpm) inputBpm.value = selectedTrack.bpm.toFixed(2);
    drawRGBWaveform(selectedTrack);
    showToast(`BPM multiplied by 2 (x2): ${selectedTrack.bpm} BPM`, "info", 2500);
  });
}

if (btnBpmTap) {
  btnBpmTap.addEventListener("click", () => {
    const now = Date.now();
    tapTimes.push(now);
    tapTimes = tapTimes.filter(t => (now - t) <= 3000);
    if (tapTimes.length >= 3) {
      const intervals = [];
      for (let i = 1; i < tapTimes.length; i++) {
        intervals.push(tapTimes[i] - tapTimes[i - 1]);
      }
      const avgMs = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      if (avgMs > 0) {
        const calculatedBpm = parseFloat((60000 / avgMs).toFixed(2));
        if (calculatedBpm >= 40 && calculatedBpm <= 250) {
          if (selectedTrack) {
            selectedTrack.bpm = calculatedBpm;
            drawRGBWaveform(selectedTrack);
          }
          if (inputBpm) inputBpm.value = calculatedBpm.toFixed(2);
          showToast(`Tap Tempo: ${calculatedBpm} BPM detected!`, "info", 2000);
        }
      }
    }
  });
}

// Beatgrid & Snap Toggle Event Listeners
if (btnToggleBeatgrid) {
  btnToggleBeatgrid.addEventListener("click", () => {
    showBeatgrid = !showBeatgrid;
    btnToggleBeatgrid.classList.toggle("active", showBeatgrid);
    if (selectedTrack) drawRGBWaveform(selectedTrack);
    showToast(`Beatgrid ${showBeatgrid ? 'Enabled' : 'Disabled'}`, "info", 2000);
  });
}

if (btnToggleSnap) {
  btnToggleSnap.addEventListener("click", () => {
    snapToGrid = !snapToGrid;
    btnToggleSnap.classList.toggle("active", snapToGrid);
    showToast(`Magnetic Snap ${snapToGrid ? 'Enabled' : 'Disabled'}`, "info", 2000);
  });
}

// Zoom Controls
if (btnZoomIn) {
  btnZoomIn.addEventListener("click", () => setZoom(zoomLevel * 1.5));
}
if (btnZoomOut) {
  btnZoomOut.addEventListener("click", () => setZoom(zoomLevel / 1.5));
}
if (btnZoomReset) {
  btnZoomReset.addEventListener("click", () => setZoom(1.0));
}

if (waveformCanvasContainer) {
  waveformCanvasContainer.addEventListener("wheel", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.25 : (1 / 1.25);
      setZoom(zoomLevel * zoomFactor, { mouseClientX: e.clientX });
    }
  }, { passive: false });
}

// Keyboard Shortcuts Modal Event Listeners
if (btnShortcuts && modalShortcuts) {
  btnShortcuts.addEventListener("click", () => {
    modalShortcuts.classList.remove("hidden");
  });
}
if (btnCloseShortcutsModal && modalShortcuts) {
  btnCloseShortcutsModal.addEventListener("click", () => {
    modalShortcuts.classList.add("hidden");
  });
}
if (btnCloseShortcutsFooter && modalShortcuts) {
  btnCloseShortcutsFooter.addEventListener("click", () => {
    modalShortcuts.classList.add("hidden");
  });
}

// Waveform Panel Collapse / Expand Manager
function toggleWaveformPanel(forceState) {
  if (!waveformPanel) return;
  const isCurrentlyCollapsed = waveformPanel.classList.contains("collapsed");
  const willCollapse = (typeof forceState === "boolean") ? forceState : !isCurrentlyCollapsed;

  waveformPanel.classList.toggle("collapsed", willCollapse);
  try {
    localStorage.setItem("audioharmonix_waveform_collapsed", willCollapse ? "1" : "0");
  } catch (e) {}

  const drawerPillText = document.getElementById("drawer-pill-text");
  if (drawerPillText) {
    drawerPillText.textContent = willCollapse
      ? "Expand Waveform & HotCues (W)"
      : "Collapse Waveform (W)";
  }

  if (btnToggleWaveformCollapse) {
    btnToggleWaveformCollapse.title = willCollapse
      ? "Expand Waveform Panel (Key W)"
      : "Collapse Waveform to expand Playlist (Key W)";
    btnToggleWaveformCollapse.setAttribute(
      "aria-label",
      willCollapse ? "Expand Waveform Panel" : "Collapse Waveform Panel"
    );
  }

  // Redraw canvas upon expanding to guarantee crisp DPI rendering
  if (!willCollapse && selectedTrack) {
    setTimeout(() => {
      drawRGBWaveform(selectedTrack);
      renderCueMarkers(selectedTrack);
    }, 120);
  }
}

if (btnToggleWaveformCollapse) {
  btnToggleWaveformCollapse.addEventListener("click", () => toggleWaveformPanel());
}

// Set First Beat (1.1) of bar to current needle position
function setFirstBeatAtPlayhead() {
  if (!selectedTrack) {
    showToast("Select a track to calibrate First Beat (1.1).", "error");
    return;
  }

  const curSec = getCurrentPlayheadSec();
  const duration = selectedTrack.duration_secs || 180;
  const clampedSec = Math.max(0, Math.min(duration, parseFloat(curSec.toFixed(3))));

  if (!selectedTrack.cues) selectedTrack.cues = [];

  // Find existing FIRST_BEAT cue or update/create
  let fbCue = selectedTrack.cues.find(c => (c.cue_type || '').includes('FIRST_BEAT') || (c.cue_type || '').includes('INTRO'));
  if (fbCue) {
    fbCue.position_secs = clampedSec;
    fbCue.cue_type = "FIRST_BEAT";
  } else {
    fbCue = { cue_type: "FIRST_BEAT", position_secs: clampedSec, hotcue_num: 1 };
    selectedTrack.cues.unshift(fbCue);
  }

  selectedTrack.first_beat_offset = clampedSec;
  selectedTrack.cues.sort((a, b) => (a.position_secs || 0) - (b.position_secs || 0));

  drawRGBWaveform(selectedTrack);
  renderCueMarkers(selectedTrack);
  renderCuePills(selectedTrack);

  showToast(`Beatgrid 1.1 calibrated to ${formatTime(clampedSec)}. Click 'Save HotCues' to persist and adapt AI.`, "success", 4000);
}

// Micro-nudge the entire Beatgrid left or right by delta (e.g. ±5ms)
function nudgeBeatgrid(deltaSec) {
  if (!selectedTrack) return;
  const duration = selectedTrack.duration_secs || 180;
  if (!selectedTrack.cues) selectedTrack.cues = [];

  let fbCue = selectedTrack.cues.find(c => (c.cue_type || '').includes('FIRST_BEAT') || (c.cue_type || '').includes('INTRO'));
  let curOffset = (typeof selectedTrack.first_beat_offset === "number") ? selectedTrack.first_beat_offset : (fbCue ? fbCue.position_secs : 0.0);
  let newOffset = Math.max(0, Math.min(duration, parseFloat((curOffset + deltaSec).toFixed(4))));

  if (fbCue) {
    fbCue.position_secs = newOffset;
  } else {
    fbCue = { cue_type: "FIRST_BEAT", position_secs: newOffset, hotcue_num: 1 };
    selectedTrack.cues.unshift(fbCue);
  }

  selectedTrack.first_beat_offset = newOffset;
  drawRGBWaveform(selectedTrack);
  renderCueMarkers(selectedTrack);
  renderCuePills(selectedTrack);
}

// Setup hold-to-repeat for grid nudge buttons (5ms step)
function setupGridNudgeHold(buttonEl, deltaSec) {
  if (!buttonEl) return;
  let holdTimeout = null;
  let repeatInterval = null;

  const performNudge = (e) => {
    e.stopPropagation();
    e.preventDefault();
    nudgeBeatgrid(deltaSec);
  };

  const startHold = (e) => {
    performNudge(e);
    holdTimeout = setTimeout(() => {
      repeatInterval = setInterval(() => {
        nudgeBeatgrid(deltaSec);
      }, 50);
    }, 220);
  };

  const stopHold = () => {
    if (holdTimeout) clearTimeout(holdTimeout);
    if (repeatInterval) clearInterval(repeatInterval);
    holdTimeout = null;
    repeatInterval = null;
  };

  buttonEl.addEventListener("mousedown", startHold);
  buttonEl.addEventListener("mouseup", stopHold);
  buttonEl.addEventListener("mouseleave", stopHold);
  buttonEl.addEventListener("touchstart", startHold, { passive: false });
  buttonEl.addEventListener("touchend", stopHold);
  buttonEl.addEventListener("touchcancel", stopHold);
}

if (btnSetFirstBeat) {
  btnSetFirstBeat.addEventListener("click", () => setFirstBeatAtPlayhead());
}
setupGridNudgeHold(btnGridNudgeLeft, -0.005);
setupGridNudgeHold(btnGridNudgeRight, +0.005);

// Global DJ Keyboard Shortcuts Listener
window.addEventListener("keydown", (e) => {
  // If typing in form input, textarea, or select, bypass global DJ keys
  const activeEl = document.activeElement;
  if (activeEl && (activeEl.tagName === "INPUT" || activeEl.tagName === "TEXTAREA" || activeEl.tagName === "SELECT")) {
    if (e.key === "Escape") {
      activeEl.blur();
    }
    return;
  }

  // Dismiss any open modal on Escape
  if (e.key === "Escape") {
    document.querySelectorAll(".modal-backdrop").forEach(m => {
      m.classList.add("hidden");
    });
    return;
  }

  // Open / Close Shortcuts Modal on ? or F1
  if (e.key === "?" || e.key === "F1") {
    e.preventDefault();
    if (modalShortcuts) modalShortcuts.classList.toggle("hidden");
    return;
  }

  // Spacebar: Play / Pause Audio
  if (e.code === "Space" || e.key === " ") {
    e.preventDefault();
    togglePlay();
    return;
  }

  // Key W: Toggle Waveform Collapse / Maximize Playlist
  if (e.key === "w" || e.key === "W") {
    e.preventDefault();
    toggleWaveformPanel();
    return;
  }

  // Shift + M: Set First Beat (1.1) at current playhead
  if (e.shiftKey && (e.key === "M" || e.key === "m")) {
    e.preventDefault();
    setFirstBeatAtPlayhead();
    return;
  }

  // Arrow Left / Right: Beat & Bar Nudge / Scrub
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    if (!selectedTrack) return;
    const dur = selectedTrack.duration_secs || 180;
    const bpm = selectedTrack.bpm || 120.0;
    const beatSec = 60.0 / bpm;
    const stepSec = e.shiftKey ? (beatSec * 4) : beatSec;
    const curSec = (currentAudio && !isNaN(currentAudio.currentTime)) ? currentAudio.currentTime : (simulatedCurTime || 0);

    let targetSec = e.key === "ArrowLeft" ? (curSec - stepSec) : (curSec + stepSec);
    targetSec = Math.max(0, Math.min(dur, targetSec));
    seekToPosition(targetSec / dur);
    return;
  }

  // Numbers 1 to 8: Jump to HotCue 1 - 8 (preserves current play/pause state)
  if (/^[1-8]$/.test(e.key)) {
    const cueIndex = parseInt(e.key, 10) - 1;
    if (selectedTrack && selectedTrack.cues && selectedTrack.cues[cueIndex] && typeof selectedTrack.cues[cueIndex].position_secs === 'number') {
      e.preventDefault();
      const cue = selectedTrack.cues[cueIndex];
      const dur = selectedTrack.duration_secs || 180;
      seekToPosition(cue.position_secs / dur);
      showToast(`Jumped to HotCue ${cueIndex + 1} (${cue.cue_type || 'Cue'})`, "info", 1800);
      return;
    }
  }

  // Key M: Add HotCue at Playhead
  if (e.key === "m" || e.key === "M") {
    e.preventDefault();
    addCueAtPlayhead();
    return;
  }

  // Key B: Toggle Beatgrid Lines
  if (e.key === "b" || e.key === "B") {
    e.preventDefault();
    if (btnToggleBeatgrid) btnToggleBeatgrid.click();
    return;
  }

  // Key S: Toggle Snap to Beatgrid
  if (e.key === "s" || e.key === "S") {
    e.preventDefault();
    if (btnToggleSnap) btnToggleSnap.click();
    return;
  }

  // Zoom Hotkeys: + / = (Zoom In), - (Zoom Out), 0 (Reset)
  if (e.key === "+" || e.key === "=") {
    e.preventDefault();
    setZoom(zoomLevel * 1.5);
    return;
  }
  if (e.key === "-" || e.key === "_") {
    e.preventDefault();
    setZoom(zoomLevel / 1.5);
    return;
  }
  if (e.key === "0") {
    e.preventDefault();
    setZoom(1.0);
    return;
  }
});

// Initial Load
document.addEventListener("DOMContentLoaded", () => {
  initSeekListeners();
  loadTracks();

  // Restore saved waveform collapse state if preference was saved
  try {
    const savedCollapsed = localStorage.getItem("audioharmonix_waveform_collapsed");
    if (savedCollapsed === "1") {
      toggleWaveformPanel(true);
    }
  } catch (e) {}
});

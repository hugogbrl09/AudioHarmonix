/**
 * AudioHarmonix 2.0 Frontend Application Engine
 * Precise UX/UI Correction Pass
 * - HotCue: Single [Save & Teach AI] action button (no duplicate Save button)
 * - HotCue: Press-and-hold for +/-0.1s buttons
 * - HotCue: Interactive drag directly on the waveform with live feedback & snap
 * - Dividers: Double-click to restore default size on all resizers
 * - Minimal Notifications: Cleaned all redundant toasts for visual actions
 * - Table & Waveform: Independent column resizing, text ellipsis, and fluid canvas
 */

const API_BASE = window.location.origin;

let tracksData = [];
let selectedTrack = null;
let activeHarmonicFilter = '11A';
let currentAudio = null;
let isPlaying = false;
let isLooping = false;
let isShuffled = false;
let batchPollInterval = null;
let pendingDeleteTrack = null;
let pendingSeekSec = null;
let simulatedCurTime = 0;
let isDraggingScrubber = false;

// Waveform & Zoom & Beatgrid State
let zoomLevel = 1.0;
let showBeatgrid = true;
let snapToGrid = true;
let tapTimes = [];

// DOM Elements: Header & Navigation
const btnShortcuts = document.getElementById("btn-shortcuts");
const btnImport = document.getElementById("btn-import");
const btnBatch = document.getElementById("btn-batch");
const btnExportXml = document.getElementById("btn-export-xml");
const btnCollapseSidebar = document.getElementById("btn-collapse-sidebar");

const sidebarNav = document.getElementById("sidebar-nav");
const collectionColumn = document.getElementById("collection-column");
const resizerSidebar = document.getElementById("resizer-sidebar");
const resizerCollection = document.getElementById("resizer-collection");
const resizerWidgets = document.getElementById("resizer-widgets");
const hotcueWidget = document.getElementById("hotcue-widget");
const harmonicWidget = document.getElementById("harmonic-widget");

const navCollection = document.getElementById("nav-collection");
const navAnalyze = document.getElementById("nav-analyze");
const navPlaylists = document.getElementById("nav-playlists");
const navTags = document.getElementById("nav-tags");
const navSettings = document.getElementById("nav-settings");

// DOM Elements: Collection Column
const searchInput = document.getElementById("search-input");
const filterCamelot = document.getElementById("filter-camelot");
const filterEnergy = document.getElementById("filter-energy");
const trackTableBody = document.getElementById("track-table-body");
const trackCount = document.getElementById("track-count");
const collectionTable = document.getElementById("collection-table");
const collectionTableHeader = document.getElementById("collection-table-header");

// DOM Elements: Track Workspace Hero
const workspaceArtwork = document.getElementById("workspace-artwork");
const workspaceArtImg = document.getElementById("workspace-art-img");
const workspaceArtInitials = document.getElementById("workspace-art-initials");
const selectedTitle = document.getElementById("selected-title");
const workspaceArtistSubtitle = document.getElementById("workspace-artist-subtitle");
const heroKeyVal = document.getElementById("hero-key-val");
const heroBpmVal = document.getElementById("hero-bpm-val");
const heroEnergyVal = document.getElementById("hero-energy-val");

// BPM & Calibration Toolbar
const inputBpm = document.getElementById("input-bpm");
const btnBpmHalf = document.getElementById("btn-bpm-half");
const btnBpmDouble = document.getElementById("btn-bpm-double");
const btnBpmTap = document.getElementById("btn-bpm-tap");
const btnSetFirstBeat = document.getElementById("btn-set-first-beat");
const btnGridNudgeLeft = document.getElementById("btn-grid-nudge-left");
const btnGridNudgeRight = document.getElementById("btn-grid-nudge-right");
const btnToggleBeatgrid = document.getElementById("btn-toggle-beatgrid");
const btnToggleSnap = document.getElementById("btn-toggle-snap");

// Structure Segment Bar
const structureSegmentBar = document.getElementById("structure-segment-bar");

// Waveform Stage
const waveformCanvasContainer = document.getElementById("waveform-canvas-container");
const waveformScrollWrapper = document.getElementById("waveform-scroll-wrapper");
const waveformCanvas = document.getElementById("waveform-canvas");
const playhead = document.getElementById("playhead");
const cueOverlay = document.getElementById("cue-overlay");
const btnPlay = document.getElementById("btn-play");
const timeDisplay = document.getElementById("time-display");
const timeRemainingDisplay = document.getElementById("time-remaining-display");
const btnZoomIn = document.getElementById("btn-zoom-in");
const btnZoomOut = document.getElementById("btn-zoom-out");
const btnZoomReset = document.getElementById("btn-zoom-reset");
const zoomLevelLabel = document.getElementById("zoom-level-label");
const waveformMiniOverview = document.getElementById("waveform-mini-overview");
const miniOverviewProgress = document.getElementById("mini-overview-progress");
const miniOverviewHandle = document.getElementById("mini-overview-handle");

// Dual Widgets: HotCues & Harmonic Mixing
const hotcuePillsContainer = document.getElementById("hotcue-pills-container");
const btnAddCuePlayhead = document.getElementById("btn-add-cue-playhead");
const btnEditCues = document.getElementById("btn-edit-cues");
const btnSaveTeachAi = document.getElementById("btn-save-teach-ai");
const harmonicMatchCount = document.getElementById("harmonic-match-count");
const harmonicMatchesList = document.getElementById("harmonic-matches-list");
const activeFilterLabel = document.getElementById("active-filter-label");
const harmonicFilterPills = document.getElementById("harmonic-filter-pills");
const sidebarCamelotWheel = document.getElementById("sidebar-camelot-wheel");

// Bottom Persistent Player Bar
const playerArtDisc = document.getElementById("player-art-disc");
const playerArtImg = document.getElementById("player-art-img");
const playerArtText = document.getElementById("player-art-text");
const playerTrackTitle = document.getElementById("player-track-title");
const playerTrackArtist = document.getElementById("player-track-artist");
const btnPlayerPlay = document.getElementById("btn-player-play");
const btnPlayerPrev = document.getElementById("btn-player-prev");
const btnPlayerNext = document.getElementById("btn-player-next");
const btnPlayerShuffle = document.getElementById("btn-player-shuffle");
const btnPlayerRepeat = document.getElementById("btn-player-repeat");
const playerTimeCur = document.getElementById("player-time-cur");
const playerTimeRem = document.getElementById("player-time-rem");
const playerWaveformScrubber = document.getElementById("player-waveform-scrubber");
const playerScrubberProgress = document.getElementById("player-scrubber-progress");
const playerScrubberHandle = document.getElementById("player-scrubber-handle");
const playerBadgeBpm = document.getElementById("player-badge-bpm");
const playerBadgeKey = document.getElementById("player-badge-key");
const playerBadgeEnergy = document.getElementById("player-badge-energy");
const playerVolumeSlider = document.getElementById("player-volume-slider");

// Modals
const modalDeleteConfirm = document.getElementById("modal-delete-confirm");
const btnCloseDeleteModal = document.getElementById("btn-close-delete-modal");
const btnCancelDelete = document.getElementById("btn-cancel-delete");
const btnConfirmDelete = document.getElementById("btn-confirm-delete");
const deleteTrackTitleText = document.getElementById("delete-track-title-text");

const modalImport = document.getElementById("modal-import");
const btnCloseImportModal = document.getElementById("btn-close-import-modal");
const btnCloseImportFooter = document.getElementById("btn-close-import-footer");
const dropzone = document.getElementById("dropzone");
const filePicker = document.getElementById("file-picker");
const importPathInput = document.getElementById("import-path-input");
const btnRunSingleImport = document.getElementById("btn-run-single-import");
const discoveredFilesList = document.getElementById("discovered-files-list");
const importAnalysisCard = document.getElementById("import-analysis-card");
const importStatusTitle = document.getElementById("import-status-title");
const importStatusStep = document.getElementById("import-status-step");

const modalBatch = document.getElementById("modal-batch");
const btnCloseBatchModal = document.getElementById("btn-close-batch-modal");
const btnStartBatch = document.getElementById("btn-start-batch");
const btnCancelBatch = document.getElementById("btn-cancel-batch");
const batchModalTitle = document.getElementById("batch-modal-title");
const batchModalDesc = document.getElementById("batch-modal-desc");
const batchModalFile = document.getElementById("batch-modal-file");
const batchModalPct = document.getElementById("batch-modal-pct");
const batchModalProgressBar = document.getElementById("batch-modal-progress-bar");
const batchModalCount = document.getElementById("batch-modal-count");
const batchModalSpeed = document.getElementById("batch-modal-speed");
const batchModalEta = document.getElementById("batch-modal-eta");
const batchResultsSummary = document.getElementById("batch-results-summary");

const modalExport = document.getElementById("modal-export");
const btnCloseExportModal = document.getElementById("btn-close-export-modal");
const btnConfirmExport = document.getElementById("btn-confirm-export");
const exportPathInput = document.getElementById("export-path-input");

const modalShortcuts = document.getElementById("modal-shortcuts");
const btnCloseShortcutsModal = document.getElementById("btn-close-shortcuts-modal");
const btnCloseShortcutsFooter = document.getElementById("btn-close-shortcuts-footer");

const toastContainer = document.getElementById("toast-container");
const batchProgressContainer = document.getElementById("batch-progress-container");
const batchProgressInner = document.getElementById("batch-progress-inner");
const batchSpeed = document.getElementById("batch-speed");
const batchCount = document.getElementById("batch-count");

// Camelot 24-Position System Data
const CAMELOT_KEYS_DATA = [
  { num: 1,  minor: '1A',  minorName: 'G#m', major: '1B',  majorName: 'B',  color: '#06b6d4' },
  { num: 2,  minor: '2A',  minorName: 'D#m', major: '2B',  majorName: 'F#', color: '#0ea5e9' },
  { num: 3,  minor: '3A',  minorName: 'A#m', major: '3B',  majorName: 'C#', color: '#38bdf8' },
  { num: 4,  minor: '4A',  minorName: 'Fm',  major: '4B',  majorName: 'G#', color: '#3b82f6' },
  { num: 5,  minor: '5A',  minorName: 'Cm',  major: '5B',  majorName: 'D#', color: '#6366f1' },
  { num: 6,  minor: '6A',  minorName: 'Gm',  major: '6B',  majorName: 'A#', color: '#8b5cf6' },
  { num: 7,  minor: '7A',  minorName: 'Dm',  major: '7B',  majorName: 'F',  color: '#a855f7' },
  { num: 8,  minor: '8A',  minorName: 'Am',  major: '8B',  majorName: 'C',  color: '#ec4899' },
  { num: 9,  minor: '9A',  minorName: 'Em',  major: '9B',  majorName: 'G',  color: '#ef4444' },
  { num: 10, minor: '10A', minorName: 'Bm',  major: '10B', majorName: 'D',  color: '#f97316' },
  { num: 11, minor: '11A', minorName: 'F#m', major: '11B', majorName: 'A',  color: '#f59e0b' },
  { num: 12, minor: '12A', minorName: 'C#m', major: '12B', majorName: 'E',  color: '#10b981' }
];

const CUE_TYPES = [
  "INTRO", "VERSE", "BUILDUP", "DROP", "BREAK", "OUTRO"
];

function formatCleanCueType(rawType) {
  if (!rawType) return "CUE";
  const t = rawType.toString().toUpperCase().trim();
  if (t.includes("INTRO") || t.includes("FIRST")) return "INTRO";
  if (t.includes("BUILD")) return "BUILDUP";
  if (t.includes("DROP")) return "DROP";
  if (t.includes("BREAK")) return "BREAK";
  if (t.includes("OUTRO")) return "OUTRO";
  if (t.includes("VERSE")) return "VERSE";
  return t.replace(/_\d+/g, "").replace("_", " ");
}

// Helper: Toast Notifications (Reserved for Async Outcomes & System Errors)
function showToast(message, type = "info", duration = 4000) {
  if (!toastContainer) return;
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

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function getInitials(title) {
  if (!title) return 'AH';
  const clean = title.replace(/[^\w\s]/gi, '').trim();
  const parts = clean.split(/\s+/);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[1][0]).toUpperCase();
  }
  return clean.slice(0, 2).toUpperCase() || 'AH';
}

function formatTime(secs) {
  if (isNaN(secs) || secs < 0) secs = 0;
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// Press-and-Hold helper for continuous increment/decrement (Requirement 1)
function attachPressAndHold(buttonEl, stepFn) {
  let holdTimeout = null;
  let holdInterval = null;

  const start = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();

    // 1. Immediate single step on click
    stepFn();

    // 2. Start repeating interval after 250ms hold
    holdTimeout = setTimeout(() => {
      holdInterval = setInterval(() => {
        stepFn();
      }, 60);
    }, 250);

    const stop = () => {
      if (holdTimeout) { clearTimeout(holdTimeout); holdTimeout = null; }
      if (holdInterval) { clearInterval(holdInterval); holdInterval = null; }
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };

    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  };

  buttonEl.addEventListener("pointerdown", start);
}

// Fetch Tracks from Backend API
async function loadTracks() {
  const search = searchInput ? searchInput.value : "";
  const camelot = filterCamelot ? filterCamelot.value : "";
  const energy = filterEnergy ? filterEnergy.value : "1";

  try {
    const res = await fetch(`${API_BASE}/api/tracks?search=${encodeURIComponent(search)}&camelot=${camelot}&energy_min=${energy}`);
    const data = await res.json();
    if (data.status === "ok") {
      tracksData = data.tracks;
      renderTable(tracksData);
      if (selectedTrack) {
        const updated = tracksData.find(t => t.id === selectedTrack.id);
        if (updated) {
          selectedTrack = updated;
        }
      }
      updateHarmonicMatchesForFilter(activeHarmonicFilter);
      renderCircularCamelotWheel(sidebarCamelotWheel, selectedTrack, (k) => setHarmonicFilter(k));
    }
  } catch (err) {
    console.error("Error loading tracks:", err);
    showToast("Failed to connect to AudioHarmonix backend engine.", "error");
  }
}

// Render Track Table (Square Artwork, Independent Resizable Columns, Ellipsis Truncation)
function renderTable(tracks) {
  if (!trackTableBody) return;
  trackTableBody.innerHTML = "";
  if (trackCount) trackCount.textContent = `${tracks.length} Tracks`;

  if (tracks.length === 0) {
    trackTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding:30px; color:#64748b;">No tracks found. Import an audio file to begin.</td></tr>`;
    return;
  }

  tracks.forEach((t) => {
    const tr = document.createElement("tr");
    tr.className = `track-row ${selectedTrack && selectedTrack.id === t.id ? 'selected' : ''}`;

    const initials = getInitials(t.title || t.file_name);
    const keyClass = (t.camelot_key || '8a').toLowerCase();
    const energyScore = (t.energy_score || 5).toFixed(1);
    const artUrl = `${API_BASE}/api/artwork?id=${t.id}`;

    // Integer BPM presentation display
    const bpmDisplay = t.bpm ? Math.round(t.bpm) : '---';

    const artworkHtml = t.has_artwork
      ? `<div class="table-artwork-square"><img src="${artUrl}" class="artwork-img" alt="" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" /><span class="artwork-initials" style="display:none;">${initials}</span></div>`
      : `<div class="table-artwork-square"><span class="artwork-initials">${initials}</span></div>`;

    tr.innerHTML = `
      <td>
        <div class="cell-track-info">
          ${artworkHtml}
          <div class="table-meta-box">
            <div class="table-title" title="${escapeHtml(t.title || t.file_name)}">${escapeHtml(t.title || t.file_name)}</div>
          </div>
        </div>
      </td>
      <td>
        <div class="table-artist-text" title="${escapeHtml(t.artist || 'Unknown Artist')}">${escapeHtml(t.artist || 'Unknown Artist')}</div>
      </td>
      <td>
        <span class="key-badge key-${keyClass}">${t.camelot_key || '---'}</span>
      </td>
      <td>
        <span style="font-family:var(--font-mono); font-weight:700;" title="Precise: ${(t.bpm || 0).toFixed(2)} BPM">${bpmDisplay}</span>
      </td>
      <td>
        <span class="table-energy-num">${energyScore}</span>
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

// Select Track & Update Workspace
function selectTrack(track) {
  selectedTrack = track;
  renderTable(tracksData);

  const initials = getInitials(track.title || track.file_name);
  const artUrl = `${API_BASE}/api/artwork?id=${track.id}`;

  // Update Workspace Artwork
  if (workspaceArtImg && workspaceArtInitials) {
    if (track.has_artwork) {
      workspaceArtImg.src = artUrl;
      workspaceArtImg.classList.remove("hidden");
      workspaceArtInitials.style.display = "none";
      workspaceArtImg.onerror = () => {
        workspaceArtImg.classList.add("hidden");
        workspaceArtInitials.style.display = "flex";
      };
    } else {
      workspaceArtImg.classList.add("hidden");
      workspaceArtInitials.style.display = "flex";
      workspaceArtInitials.textContent = initials;
    }
  }

  // Update Player Artwork
  if (playerArtImg && playerArtText) {
    if (track.has_artwork) {
      playerArtImg.src = artUrl;
      playerArtImg.classList.remove("hidden");
      playerArtText.style.display = "none";
      playerArtImg.onerror = () => {
        playerArtImg.classList.add("hidden");
        playerArtText.style.display = "flex";
      };
    } else {
      playerArtImg.classList.add("hidden");
      playerArtText.style.display = "flex";
      playerArtText.textContent = initials;
    }
  }

  const fullTitle = track.title || track.file_name;
  if (selectedTitle) {
    selectedTitle.textContent = fullTitle;
    selectedTitle.title = fullTitle;
    
    // Check overflow for marquee hover animation
    requestAnimationFrame(() => {
      const isOverflow = selectedTitle.scrollWidth > selectedTitle.clientWidth;
      selectedTitle.classList.toggle("is-overflowing", isOverflow);
      if (isOverflow) {
        const diff = selectedTitle.scrollWidth - selectedTitle.clientWidth;
        selectedTitle.style.setProperty("--marquee-shift", `-${diff + 20}px`);
      }
    });
  }

  if (workspaceArtistSubtitle) workspaceArtistSubtitle.textContent = track.artist || 'Unknown Artist';
  
  if (playerTrackTitle) {
    playerTrackTitle.textContent = fullTitle;
    playerTrackTitle.title = fullTitle;
  }
  if (playerTrackArtist) playerTrackArtist.textContent = track.artist || 'Unknown Artist';

  const keyStr = `${track.detected_key || ''} | ${track.camelot_key || '---'}`;
  if (heroKeyVal) heroKeyVal.textContent = keyStr;
  if (playerBadgeKey) playerBadgeKey.textContent = track.camelot_key || '---';

  // Integer BPM presentation for player & hero
  const bpmRaw = track.bpm || 120.0;
  const bpmInt = Math.round(bpmRaw);
  if (heroBpmVal) heroBpmVal.textContent = bpmInt.toString();
  if (playerBadgeBpm) playerBadgeBpm.textContent = bpmInt.toString();
  if (inputBpm) inputBpm.value = bpmRaw.toFixed(2);

  const energyScore = (track.energy_score || 5).toFixed(1);
  if (heroEnergyVal) heroEnergyVal.textContent = energyScore;
  if (playerBadgeEnergy) playerBadgeEnergy.textContent = energyScore;

  // Reset Zoom
  zoomLevel = 1.0;
  if (zoomLevelLabel) zoomLevelLabel.textContent = "1.0x";
  if (btnZoomReset) btnZoomReset.textContent = "1.0x";

  // Draw Visual Components
  renderStructureSegments(track);
  drawRGBWaveform(track);
  renderCueMarkers(track);
  renderCueCards(track);
  
  // Set harmonic filter to selected track key initially
  activeHarmonicFilter = track.camelot_key || '11A';
  renderHarmonicFilterButtons(track);
  updateHarmonicMatchesForFilter(activeHarmonicFilter);
  renderCircularCamelotWheel(sidebarCamelotWheel, track, (k) => setHarmonicFilter(k));

  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  isPlaying = false;
  stopPlayheadAnimation();
  updatePlayButtonIcons(false);

  currentAudio = new Audio(`${API_BASE}/audio/?id=${track.id}`);
  currentAudio.volume = playerVolumeSlider ? parseFloat(playerVolumeSlider.value) : 0.9;
  
  currentAudio.addEventListener("ended", () => {
    if (isLooping) {
      currentAudio.currentTime = 0;
      currentAudio.play();
    } else {
      isPlaying = false;
      stopPlayheadAnimation();
      updatePlayButtonIcons(false);
      playhead.style.left = "0%";
      if (miniOverviewProgress) miniOverviewProgress.style.width = "0%";
      if (miniOverviewHandle) miniOverviewHandle.style.left = "0%";
      if (playerScrubberProgress) playerScrubberProgress.style.width = "0%";
      if (playerScrubberHandle) playerScrubberHandle.style.left = "0%";
      simulatedCurTime = 0;
    }
  });

  currentAudio.addEventListener("loadedmetadata", () => {
    if (pendingSeekSec !== null && currentAudio) {
      try { currentAudio.currentTime = pendingSeekSec; } catch (e) {}
    }
  });

  if (pendingSeekSec === null) {
    playhead.style.left = "0%";
    if (miniOverviewProgress) miniOverviewProgress.style.width = "0%";
    if (miniOverviewHandle) miniOverviewHandle.style.left = "0%";
    if (playerScrubberProgress) playerScrubberProgress.style.width = "0%";
    if (playerScrubberHandle) playerScrubberHandle.style.left = "0%";
    const totalDur = formatTime(track.duration_secs || 0);
    if (timeDisplay) timeDisplay.textContent = "0:00";
    if (timeRemainingDisplay) timeRemainingDisplay.textContent = `-${totalDur}`;
    if (playerTimeCur) playerTimeCur.textContent = "0:00";
    if (playerTimeRem) playerTimeRem.textContent = `-${totalDur}`;
  }
}

function updatePlayButtonIcons(playing) {
  const playSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
  const pauseSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;

  if (btnPlay) btnPlay.innerHTML = playing ? pauseSvg : playSvg;
  if (btnPlayerPlay) btnPlayerPlay.innerHTML = playing ? pauseSvg : `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
}

// Render Musical Structure Segments Bar Above Waveform
function renderStructureSegments(track) {
  if (!structureSegmentBar) return;
  structureSegmentBar.innerHTML = "";

  const duration = track.duration_secs || 180;
  const cues = track.cues || [];

  if (cues.length === 0) {
    const defaultSegments = [
      { name: "INTRO", cls: "seg-intro", pct: 15 },
      { name: "VERSE", cls: "seg-verse", pct: 20 },
      { name: "BUILD", cls: "seg-build", pct: 15 },
      { name: "DROP 1", cls: "seg-drop", pct: 20 },
      { name: "BREAKDOWN", cls: "seg-break", pct: 15 },
      { name: "DROP 2", cls: "seg-drop2", pct: 10 },
      { name: "OUTRO", cls: "seg-outro", pct: 5 }
    ];
    defaultSegments.forEach(s => {
      const seg = document.createElement("div");
      seg.className = `segment-pill ${s.cls}`;
      seg.style.flex = `${s.pct}`;
      seg.textContent = s.name;
      structureSegmentBar.appendChild(seg);
    });
    return;
  }

  const sorted = [...cues].sort((a, b) => (a.position_secs || 0) - (b.position_secs || 0));
  for (let i = 0; i < sorted.length; i++) {
    const cur = sorted[i];
    const nextSec = sorted[i + 1] ? sorted[i + 1].position_secs : duration;
    const durSec = Math.max(1, nextSec - cur.position_secs);
    const pct = Math.max(5, (durSec / duration) * 100);

    const seg = document.createElement("div");
    const cleanType = formatCleanCueType(cur.cue_type);
    let cls = "seg-verse";
    if (cleanType === "INTRO") cls = "seg-intro";
    else if (cleanType === "BUILDUP") cls = "seg-build";
    else if (cleanType === "DROP") cls = "seg-drop";
    else if (cleanType === "BREAK") cls = "seg-break";
    else if (cleanType === "OUTRO") cls = "seg-outro";

    seg.className = `segment-pill ${cls}`;
    seg.style.flex = `${pct}`;
    seg.textContent = cleanType;
    seg.title = `${cleanType} (${formatTime(cur.position_secs)}) - Click to jump`;
    seg.addEventListener("click", () => {
      seekToPosition(cur.position_secs / duration);
    });
    structureSegmentBar.appendChild(seg);
  }
}

// User-Controlled Horizontal Harmonic Filter (Section 7 Requirement)
function renderHarmonicFilterButtons(track) {
  if (!harmonicFilterPills) return;
  harmonicFilterPills.innerHTML = "";

  const baseKey = track ? (track.camelot_key || '11A') : '11A';
  const num = parseInt(baseKey) || 11;
  const letter = baseKey.slice(-1) || 'A';
  const otherLetter = letter === "A" ? "B" : "A";

  const subdom = `${((num - 2 + 12) % 12) + 1}${letter}`;
  const dom = `${((num % 12)) + 1}${letter}`;
  const relative = `${num}${otherLetter}`;

  const filterOptions = [
    { key: subdom, label: `-1 (${subdom})` },
    { key: baseKey, label: `Same (${baseKey})` },
    { key: dom, label: `+1 (${dom})` },
    { key: relative, label: `Relative (${relative})` }
  ];

  if (activeFilterLabel) activeFilterLabel.textContent = activeHarmonicFilter;

  filterOptions.forEach(opt => {
    const btn = document.createElement("button");
    btn.className = `btn-harmonic-pill ${activeHarmonicFilter === opt.key ? 'active' : 'compatible'}`;
    btn.textContent = opt.key;
    btn.title = `Harmonic Filter: ${opt.label}`;

    btn.addEventListener("click", () => {
      setHarmonicFilter(opt.key);
    });

    harmonicFilterPills.appendChild(btn);
  });
}

function setHarmonicFilter(key) {
  activeHarmonicFilter = key;
  if (activeFilterLabel) activeFilterLabel.textContent = key;
  renderHarmonicFilterButtons(selectedTrack);
  updateHarmonicMatchesForFilter(key);
  renderCircularCamelotWheel(sidebarCamelotWheel, selectedTrack, (k) => setHarmonicFilter(k));
}

function updateHarmonicMatchesForFilter(filterKey) {
  if (!harmonicMatchesList) return;
  harmonicMatchesList.innerHTML = "";

  const matches = tracksData.filter(t => (!selectedTrack || t.id !== selectedTrack.id) && t.camelot_key === filterKey);
  if (harmonicMatchCount) harmonicMatchCount.textContent = `${matches.length} Matches Found (${filterKey})`;

  if (matches.length === 0) {
    harmonicMatchesList.innerHTML = `<span class="no-matches-text">No compatible tracks found in library for key ${filterKey}.</span>`;
    return;
  }

  matches.forEach(m => {
    const item = document.createElement("div");
    item.className = "harmonic-match-item";
    const initials = getInitials(m.title || m.file_name);
    const artUrl = `${API_BASE}/api/artwork?id=${m.id}`;

    const artworkHtml = m.has_artwork
      ? `<div class="match-item-artwork-square"><img src="${artUrl}" class="artwork-img" alt="" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" /><span class="artwork-initials" style="display:none;">${initials}</span></div>`
      : `<div class="match-item-artwork-square"><span class="artwork-initials">${initials}</span></div>`;

    item.innerHTML = `
      <div class="match-item-left">
        ${artworkHtml}
        <div class="match-item-text">
          <div class="match-item-title" title="${escapeHtml(m.title || m.file_name)}">${escapeHtml(m.title || m.file_name)}</div>
          <div class="match-item-artist">${escapeHtml(m.artist || 'Unknown Artist')}</div>
        </div>
      </div>
      <div class="match-item-key">${m.camelot_key}</div>
    `;

    item.addEventListener("click", () => {
      selectTrack(m);
    });

    harmonicMatchesList.appendChild(item);
  });
}

// Single Source of Truth for Circular Camelot Wheel SVG (Sidebar Integration)
function renderCircularCamelotWheel(svgElement, track, onKeyClick) {
  if (!svgElement) return;
  svgElement.innerHTML = "";

  const curKey = activeHarmonicFilter || (track ? (track.camelot_key || '11A') : '11A');
  const num = parseInt(curKey) || 11;
  const letter = curKey.slice(-1) || 'A';
  const otherLetter = letter === "A" ? "B" : "A";

  const compatibleKeys = new Set([
    curKey,
    `${num}${otherLetter}`,
    `${((num - 2 + 12) % 12) + 1}${letter}`,
    `${((num % 12)) + 1}${letter}`
  ]);

  const cx = 100;
  const cy = 100;
  const rOuterA = 96;
  const rInnerA = 68;
  const rOuterB = 67;
  const rInnerB = 40;
  const rHub = 38;

  function createSectorPath(startAngleDeg, endAngleDeg, rIn, rOut) {
    const startRad = (startAngleDeg - 90) * (Math.PI / 180);
    const endRad = (endAngleDeg - 90) * (Math.PI / 180);

    const x1 = cx + rOut * Math.cos(startRad);
    const y1 = cy + rOut * Math.sin(startRad);
    const x2 = cx + rOut * Math.cos(endRad);
    const y2 = cy + rOut * Math.sin(endRad);
    const x3 = cx + rIn * Math.cos(endRad);
    const y3 = cy + rIn * Math.sin(endRad);
    const x4 = cx + rIn * Math.cos(startRad);
    const y4 = cy + rIn * Math.sin(startRad);

    const largeArc = (endAngleDeg - startAngleDeg) > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${rOut} ${rOut} 0 ${largeArc} 1 ${x2} ${y2} L ${x3} ${y3} A ${rIn} ${rIn} 0 ${largeArc} 0 ${x4} ${y4} Z`;
  }

  // Draw 12 Sectors
  for (let i = 0; i < 12; i++) {
    const keyData = CAMELOT_KEYS_DATA[i];
    const keyNum = keyData.num;
    const centerDeg = (keyNum % 12) * 30;
    const startDeg = centerDeg - 14.5;
    const endDeg = centerDeg + 14.5;

    // --- Outer Ring: Minor Key (A) ---
    const isCurA = (curKey === keyData.minor);
    const isCompatA = compatibleKeys.has(keyData.minor);
    const pathA = document.createElementNS("http://www.w3.org/2000/svg", "path");
    pathA.setAttribute("d", createSectorPath(startDeg, endDeg, rInnerA, rOuterA));
    pathA.setAttribute("class", "camelot-sector");
    pathA.setAttribute("fill", keyData.color);
    pathA.setAttribute("fill-opacity", isCurA ? "1.0" : (isCompatA ? "0.80" : "0.22"));
    pathA.setAttribute("stroke", isCurA ? "#ffffff" : (isCompatA ? "#38bdf8" : "rgba(255,255,255,0.1)"));
    pathA.setAttribute("stroke-width", isCurA ? "2" : (isCompatA ? "1.5" : "0.5"));

    const titleA = document.createElementNS("http://www.w3.org/2000/svg", "title");
    titleA.textContent = `${keyData.minor} (${keyData.minorName}) - ${isCurA ? 'Active Filter Key' : (isCompatA ? 'Harmonic Match' : 'Incompatible')}`;
    pathA.appendChild(titleA);

    pathA.addEventListener("click", () => {
      if (onKeyClick) onKeyClick(keyData.minor);
    });

    svgElement.appendChild(pathA);

    // Text Label A
    const midRadA = (centerDeg - 90) * (Math.PI / 180);
    const textRA = (rOuterA + rInnerA) / 2;
    const txA = cx + textRA * Math.cos(midRadA);
    const tyA = cy + textRA * Math.sin(midRadA);

    const textA = document.createElementNS("http://www.w3.org/2000/svg", "text");
    textA.setAttribute("x", txA.toString());
    textA.setAttribute("y", (tyA + 3.5).toString());
    textA.setAttribute("text-anchor", "middle");
    textA.setAttribute("fill", isCurA ? "#000000" : "#ffffff");
    textA.setAttribute("font-size", "9");
    textA.setAttribute("font-weight", isCurA || isCompatA ? "800" : "600");
    textA.setAttribute("font-family", "JetBrains Mono, monospace");
    textA.setAttribute("pointer-events", "none");
    textA.textContent = keyData.minor;
    svgElement.appendChild(textA);

    // --- Inner Ring: Major Key (B) ---
    const isCurB = (curKey === keyData.major);
    const isCompatB = compatibleKeys.has(keyData.major);
    const pathB = document.createElementNS("http://www.w3.org/2000/svg", "path");
    pathB.setAttribute("d", createSectorPath(startDeg, endDeg, rInnerB, rOuterB));
    pathB.setAttribute("class", "camelot-sector");
    pathB.setAttribute("fill", keyData.color);
    pathB.setAttribute("fill-opacity", isCurB ? "1.0" : (isCompatB ? "0.80" : "0.22"));
    pathB.setAttribute("stroke", isCurB ? "#ffffff" : (isCompatB ? "#38bdf8" : "rgba(255,255,255,0.1)"));
    pathB.setAttribute("stroke-width", isCurB ? "2" : (isCompatB ? "1.5" : "0.5"));

    const titleB = document.createElementNS("http://www.w3.org/2000/svg", "title");
    titleB.textContent = `${keyData.major} (${keyData.majorName}) - ${isCurB ? 'Active Filter Key' : (isCompatB ? 'Harmonic Match' : 'Incompatible')}`;
    pathB.appendChild(titleB);

    pathB.addEventListener("click", () => {
      if (onKeyClick) onKeyClick(keyData.major);
    });

    svgElement.appendChild(pathB);

    // Text Label B
    const textRB = (rOuterB + rInnerB) / 2;
    const txB = cx + textRB * Math.cos(midRadA);
    const tyB = cy + textRB * Math.sin(midRadA);

    const textB = document.createElementNS("http://www.w3.org/2000/svg", "text");
    textB.setAttribute("x", txB.toString());
    textB.setAttribute("y", (tyB + 3).toString());
    textB.setAttribute("text-anchor", "middle");
    textB.setAttribute("fill", isCurB ? "#000000" : "#cbd5e1");
    textB.setAttribute("font-size", "8");
    textB.setAttribute("font-weight", isCurB || isCompatB ? "800" : "600");
    textB.setAttribute("font-family", "JetBrains Mono, monospace");
    textB.setAttribute("pointer-events", "none");
    textB.textContent = keyData.major;
    svgElement.appendChild(textB);
  }

  // Center Hub Circle
  const hub = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  hub.setAttribute("cx", cx.toString());
  hub.setAttribute("cy", cy.toString());
  hub.setAttribute("r", rHub.toString());
  hub.setAttribute("fill", "#11151e");
  hub.setAttribute("stroke", "rgba(255,255,255,0.15)");
  hub.setAttribute("stroke-width", "1.5");
  svgElement.appendChild(hub);

  // Hub Center Text
  const hubTextKey = document.createElementNS("http://www.w3.org/2000/svg", "text");
  hubTextKey.setAttribute("x", cx.toString());
  hubTextKey.setAttribute("y", (cy - 3).toString());
  hubTextKey.setAttribute("text-anchor", "middle");
  hubTextKey.setAttribute("fill", "#38bdf8");
  hubTextKey.setAttribute("font-size", "13");
  hubTextKey.setAttribute("font-weight", "800");
  hubTextKey.setAttribute("font-family", "JetBrains Mono, monospace");
  hubTextKey.textContent = curKey;
  svgElement.appendChild(hubTextKey);

  const hubTextSub = document.createElementNS("http://www.w3.org/2000/svg", "text");
  hubTextSub.setAttribute("x", cx.toString());
  hubTextSub.setAttribute("y", (cy + 12).toString());
  hubTextSub.setAttribute("text-anchor", "middle");
  hubTextSub.setAttribute("fill", "#94a3b8");
  hubTextSub.setAttribute("font-size", "8.5");
  hubTextSub.setAttribute("font-weight", "600");
  hubTextSub.setAttribute("font-family", "Inter, sans-serif");
  hubTextSub.textContent = track ? (track.detected_key || "KEY") : "KEY";
  svgElement.appendChild(hubTextSub);
}

// Draw 3-Band RGB Waveform Canvas with Scalable Zoom & Fluid 100% Width
function drawRGBWaveform(track) {
  if (!waveformCanvasContainer || !waveformCanvas) return;

  const containerRect = waveformCanvasContainer.getBoundingClientRect();
  const baseWidth = Math.round(containerRect.width || waveformCanvasContainer.clientWidth || 900);
  const totalWidth = Math.max(baseWidth, Math.round(baseWidth * zoomLevel));
  const height = 125;

  if (waveformScrollWrapper) {
    waveformScrollWrapper.style.width = `${totalWidth}px`;
  }
  waveformCanvas.width = totalWidth;
  waveformCanvas.height = height;
  waveformCanvas.style.width = `${totalWidth}px`;
  waveformCanvas.style.height = `${height}px`;

  if (cueOverlay) {
    cueOverlay.style.width = `${totalWidth}px`;
  }

  const ctx = waveformCanvas.getContext("2d");
  ctx.clearRect(0, 0, totalWidth, height);

  ctx.fillStyle = "#090c13";
  ctx.fillRect(0, 0, totalWidth, height);

  // Center Zero-Crossing Line
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(totalWidth, height / 2);
  ctx.stroke();

  // Draw Beatgrid Lines if enabled
  if (showBeatgrid && track && (track.bpm || 0) > 0) {
    drawBeatgridLines(ctx, track, totalWidth, height);
  }

  const wf = track ? track.waveform_peaks : null;

  if (wf && wf.low && wf.low.length > 0) {
    const numPoints = wf.low.length;
    const barWidth = totalWidth / numPoints;

    for (let i = 0; i < numPoints; i++) {
      const x = i * barWidth;
      const lowVal = wf.low[i] || 0;
      const midVal = wf.mid ? wf.mid[i] || 0 : 0;
      const highVal = wf.high ? wf.high[i] || 0 : 0;

      const hLow = lowVal * (height * 0.90);
      const hMid = midVal * (height * 0.70);
      const hHigh = highVal * (height * 0.50);

      // Layer 1: Highs (Percussion & Cymbals) - Indigo / Purple
      if (hHigh > 0) {
        ctx.fillStyle = "rgba(139, 92, 246, 0.70)";
        ctx.fillRect(x, (height / 2) - (hHigh / 2), Math.max(1, barWidth - 0.3), hHigh);
      }

      // Layer 2: Mids (Vocals & Leads) - Cyan / Green
      if (hMid > 0) {
        ctx.fillStyle = "rgba(6, 182, 212, 0.80)";
        ctx.fillRect(x, (height / 2) - (hMid / 2), Math.max(1, barWidth - 0.3), hMid);
      }

      // Layer 3: Lows (Sub & Kick) - Red / Coral
      if (hLow > 0) {
        ctx.fillStyle = "rgba(239, 68, 68, 0.95)";
        ctx.fillRect(x, (height / 2) - (hLow / 2), Math.max(1, barWidth - 0.3), hLow);
      }
    }
  } else {
    ctx.fillStyle = "#64748b";
    ctx.font = "12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Select an analyzed track to render waveform", totalWidth / 2, height / 2 + 4);
  }
}

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
      ctx.strokeStyle = "rgba(168, 85, 247, 0.85)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      ctx.fillStyle = "#c084fc";
      ctx.font = "bold 9px monospace";
      ctx.fillText(`${barNum}.1`, x, 11);
    } else if (isMajorBar) {
      ctx.strokeStyle = "rgba(6, 182, 212, 0.75)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      ctx.fillStyle = "#22d3ee";
      ctx.font = "bold 9px monospace";
      ctx.fillText(`${barNum}.1`, x, 11);
    } else if (isBarStart) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    } else if (zoomLevel >= 1.5) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 14);
      ctx.lineTo(x, height - 14);
      ctx.stroke();
    }
  }
  ctx.restore();
}

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

// Waveform Zoom & Shift+Scroll Pan
function handleWaveformWheel(e) {
  if (!selectedTrack) return;
  e.preventDefault();

  // Shift + Scroll -> Pan horizontally
  if (e.shiftKey) {
    if (waveformCanvasContainer) {
      waveformCanvasContainer.scrollLeft += e.deltaY;
    }
    return;
  }

  // Normal Scroll -> Zoom anchored around mouse position
  const container = waveformCanvasContainer;
  const containerRect = container.getBoundingClientRect();
  const mouseX = Math.max(0, Math.min(containerRect.width, e.clientX - containerRect.left));
  const scrollLeft = container.scrollLeft;
  const currentTotalWidth = waveformCanvas.width || containerRect.width;

  const anchorRatio = (scrollLeft + mouseX) / currentTotalWidth;

  const zoomFactor = e.deltaY < 0 ? 1.25 : (1 / 1.25);
  const newZoom = Math.max(1.0, Math.min(16.0, parseFloat((zoomLevel * zoomFactor).toFixed(2))));
  if (newZoom === zoomLevel) return;

  zoomLevel = newZoom;
  if (zoomLevelLabel) zoomLevelLabel.textContent = `${zoomLevel.toFixed(1)}x`;
  if (btnZoomReset) btnZoomReset.textContent = `${zoomLevel.toFixed(1)}x`;

  drawRGBWaveform(selectedTrack);
  renderCueMarkers(selectedTrack);

  const newTotalWidth = waveformCanvas.width;
  const targetScrollLeft = (anchorRatio * newTotalWidth) - mouseX;
  container.scrollLeft = Math.max(0, Math.min(newTotalWidth - container.clientWidth, targetScrollLeft));
}

function setZoom(newZoom) {
  zoomLevel = Math.max(1.0, Math.min(16.0, parseFloat(newZoom.toFixed(1))));
  
  if (zoomLevelLabel) zoomLevelLabel.textContent = `${zoomLevel.toFixed(1)}x`;
  if (btnZoomReset) btnZoomReset.textContent = `${zoomLevel.toFixed(1)}x`;

  if (selectedTrack) {
    const curSec = getCurrentPlayheadSec();
    const duration = selectedTrack.duration_secs || 180;
    
    drawRGBWaveform(selectedTrack);
    renderCueMarkers(selectedTrack);

    if (waveformCanvasContainer && waveformCanvas) {
      const containerWidth = waveformCanvasContainer.clientWidth;
      const totalWidth = waveformCanvas.width;
      const playheadPx = (curSec / duration) * totalWidth;
      waveformCanvasContainer.scrollLeft = playheadPx - (containerWidth / 2);
    }
  }
}

// Render Cue Markers Overlay with Interactive Dragging (Requirement 3 & 4)
function renderCueMarkers(track) {
  if (!cueOverlay) return;
  cueOverlay.innerHTML = "";
  if (!track || !track.cues) return;

  const duration = track.duration_secs || 180;
  const validCues = track.cues.filter(c => typeof c.position_secs === 'number' && c.position_secs >= 0 && c.position_secs <= duration);

  validCues.forEach((c, idx) => {
    const letter = String.fromCharCode(65 + idx);
    const padNum = (idx % 8) + 1;
    const pct = Math.min(100, Math.max(0, (c.position_secs / duration) * 100));
    const cleanType = formatCleanCueType(c.cue_type);

    const cueMarkerWrap = document.createElement("div");
    cueMarkerWrap.className = "waveform-cue-marker-wrap";
    cueMarkerWrap.style.left = `${pct}%`;
    cueMarkerWrap.title = `HotCue ${letter}: ${cleanType} (${formatTime(c.position_secs)}) - Click & drag to move`;

    cueMarkerWrap.innerHTML = `
      <div class="waveform-cue-flag pad-${padNum}">
        <span class="flag-letter">${letter}</span>
        <span class="flag-type">${cleanType}</span>
      </div>
      <div class="waveform-cue-line pad-border-${padNum}"></div>
    `;

    // Make cue marker draggable directly on the waveform (Requirement 3 & 4)
    cueMarkerWrap.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();

      cueMarkerWrap.classList.add("is-dragging");
      document.querySelectorAll(".waveform-cue-marker-wrap").forEach(m => m.classList.remove("is-selected"));
      cueMarkerWrap.classList.add("is-selected");

      // Highlight corresponding row in HotCue Studio
      document.querySelectorAll(".hotcue-pill-row").forEach((r, rIdx) => {
        r.classList.toggle("active", rIdx === idx);
      });

      const container = waveformCanvasContainer;
      const scrollWrapper = waveformScrollWrapper || container;

      const onMove = (moveEvt) => {
        const rect = scrollWrapper.getBoundingClientRect();
        const mouseX = Math.max(0, Math.min(rect.width, moveEvt.clientX - rect.left));
        let rawSec = (mouseX / rect.width) * duration;

        if (snapToGrid) {
          rawSec = snapTimestampToBeat(rawSec, track);
        } else {
          rawSec = parseFloat(rawSec.toFixed(3));
        }
        rawSec = Math.max(0, Math.min(duration, rawSec));

        c.position_secs = rawSec;
        const newPct = (rawSec / duration) * 100;
        cueMarkerWrap.style.left = `${newPct}%`;

        // Update live time readout in HotCue Studio
        const rows = document.querySelectorAll(".hotcue-pill-row");
        if (rows[idx]) {
          const timeEl = rows[idx].querySelector(".hotcue-pill-time");
          if (timeEl) timeEl.textContent = formatTime(rawSec);
        }

        // Live flag tooltip
        const typeEl = cueMarkerWrap.querySelector(".flag-type");
        if (typeEl) typeEl.textContent = `${formatCleanCueType(c.cue_type)} ${formatTime(rawSec)}`;
      };

      const onUp = () => {
        cueMarkerWrap.classList.remove("is-dragging");
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        window.removeEventListener("pointercancel", onUp);

        // Re-sort cues after drag
        track.cues.sort((a, b) => (a.position_secs || 0) - (b.position_secs || 0));
        renderCueMarkers(track);
        renderCueCards(track);
        renderStructureSegments(track);
      };

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
    });

    cueOverlay.appendChild(cueMarkerWrap);
  });
}

// Render HotCue Studio Compact Items with Press-and-Hold (Requirement 1)
function renderCueCards(track) {
  if (!hotcuePillsContainer) return;
  hotcuePillsContainer.innerHTML = "";

  if (!track || !track.cues || track.cues.length === 0) {
    hotcuePillsContainer.innerHTML = `<span class="no-cues-text">No HotCues set. Click "Add Cue" or press 'M' at needle.</span>`;
    return;
  }

  const duration = track.duration_secs || 180;
  const validCues = track.cues.filter(c => typeof c.position_secs === 'number' && c.position_secs >= 0 && c.position_secs <= duration);

  validCues.forEach((c, idx) => {
    const letter = String.fromCharCode(65 + idx);
    const padNum = (idx % 8) + 1;
    const cleanType = formatCleanCueType(c.cue_type);
    const row = document.createElement("div");
    row.className = "hotcue-pill-row";
    row.title = `Click to jump to HotCue ${letter} (${formatTime(c.position_secs)})`;

    row.innerHTML = `
      <div class="hotcue-row-left">
        <div class="pad-box pad-${padNum}">${letter}</div>
        <select class="cue-row-select">
          ${CUE_TYPES.map(t => `<option value="${t}" ${cleanType === t ? 'selected' : ''}>${t}</option>`).join('')}
        </select>
      </div>
      <div class="hotcue-row-right">
        <span class="hotcue-pill-time">${formatTime(c.position_secs)}</span>
        <div class="hotcue-row-nudges">
          <button class="btn-pill-nudge btn-nudge-left" title="Click or hold to decrement (-0.1s)">-0.1s</button>
          <button class="btn-pill-nudge btn-nudge-right" title="Click or hold to increment (+0.1s)">+0.1s</button>
        </div>
        <button class="btn-pill-del" title="Delete Cue">✕</button>
      </div>
    `;

    // Click row to jump
    row.addEventListener("click", () => {
      seekToPosition(c.position_secs / duration);
      document.querySelectorAll(".hotcue-pill-row").forEach(r => r.classList.remove("active"));
      row.classList.add("active");
    });

    const selectEl = row.querySelector(".cue-row-select");
    selectEl.addEventListener("click", (e) => e.stopPropagation());
    selectEl.addEventListener("change", (e) => {
      e.stopPropagation();
      c.cue_type = e.target.value;
      renderCueMarkers(track);
      renderStructureSegments(track);
    });

    // Press and Hold continuous fine adjustment (Requirement 1)
    const btnNudgeLeft = row.querySelector(".btn-pill-nudge.btn-nudge-left");
    const btnNudgeRight = row.querySelector(".btn-pill-nudge.btn-nudge-right");
    const timeSpan = row.querySelector(".hotcue-pill-time");

    attachPressAndHold(btnNudgeLeft, () => {
      c.position_secs = Math.max(0, parseFloat((c.position_secs - 0.1).toFixed(3)));
      timeSpan.textContent = formatTime(c.position_secs);
      renderCueMarkers(track);
      renderStructureSegments(track);
    });

    attachPressAndHold(btnNudgeRight, () => {
      c.position_secs = Math.min(duration, parseFloat((c.position_secs + 0.1).toFixed(3)));
      timeSpan.textContent = formatTime(c.position_secs);
      renderCueMarkers(track);
      renderStructureSegments(track);
    });

    // Direct Time Editing on double click
    timeSpan.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      editHotCueRow(row, c, track);
    });

    row.querySelector(".btn-pill-del").addEventListener("click", (e) => {
      e.stopPropagation();
      const realIdx = track.cues.indexOf(c);
      if (realIdx !== -1) track.cues.splice(realIdx, 1);
      renderCueMarkers(track);
      renderCueCards(track);
      renderStructureSegments(track);
    });

    hotcuePillsContainer.appendChild(row);
  });
}

function editHotCueRow(row, cue, track) {
  if (!row || !cue || !track) return;
  const timeSpan = row.querySelector(".hotcue-pill-time");
  const selectEl = row.querySelector(".cue-row-select");
  if (!timeSpan || !selectEl) return;

  selectEl.focus();

  if (row.querySelector(".input-cue-time-edit")) return;

  const duration = track.duration_secs || 180;
  const currentFormatted = formatTime(cue.position_secs);

  const inputEl = document.createElement("input");
  inputEl.className = "input-cue-time-edit";
  inputEl.type = "text";
  inputEl.value = currentFormatted;

  timeSpan.style.display = "none";
  timeSpan.parentNode.insertBefore(inputEl, timeSpan);
  inputEl.focus();
  inputEl.select();

  const commitTime = () => {
    const val = inputEl.value.trim();
    let parsedSec = cue.position_secs;
    if (val.includes(":")) {
      const parts = val.split(":");
      const mins = parseFloat(parts[0]) || 0;
      const secs = parseFloat(parts[1]) || 0;
      parsedSec = (mins * 60) + secs;
    } else {
      parsedSec = parseFloat(val) || cue.position_secs;
    }

    if (!isNaN(parsedSec)) {
      cue.position_secs = Math.max(0, Math.min(duration, parseFloat(parsedSec.toFixed(3))));
    }

    timeSpan.textContent = formatTime(cue.position_secs);
    inputEl.remove();
    timeSpan.style.display = "";

    track.cues.sort((a, b) => (a.position_secs || 0) - (b.position_secs || 0));
    renderCueMarkers(track);
    renderStructureSegments(track);
    seekToPosition(cue.position_secs / duration);
  };

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitTime();
    } else if (e.key === "Escape") {
      inputEl.remove();
      timeSpan.style.display = "";
    }
  });

  inputEl.addEventListener("blur", commitTime);
}

function getCurrentPlayheadSec() {
  if (!selectedTrack) return 0;
  const duration = selectedTrack.duration_secs || 180;
  if (currentAudio && !currentAudio.paused && !isNaN(currentAudio.currentTime) && currentAudio.currentTime > 0) {
    return currentAudio.currentTime;
  }
  if (pendingSeekSec !== null && !isNaN(pendingSeekSec)) return pendingSeekSec;
  if (simulatedCurTime > 0) return simulatedCurTime;
  return 0;
}

function addCueAtPlayhead() {
  if (!selectedTrack) {
    showToast("Select a track first to add HotCues.", "error");
    return;
  }

  const duration = selectedTrack.duration_secs || 180;
  let curSec = getCurrentPlayheadSec();

  if (snapToGrid) {
    curSec = snapTimestampToBeat(curSec, selectedTrack);
  }
  curSec = Math.max(0, Math.min(duration, parseFloat(curSec.toFixed(3))));

  if (!selectedTrack.cues) selectedTrack.cues = [];

  const existingNear = selectedTrack.cues.find(c => Math.abs(c.position_secs - curSec) < 0.15);
  if (existingNear) return;

  selectedTrack.cues.push({
    cue_type: selectedTrack.cues.length === 0 ? "INTRO" : "DROP",
    position_secs: curSec,
    hotcue_num: selectedTrack.cues.length + 1
  });

  selectedTrack.cues.sort((a, b) => (a.position_secs || 0) - (b.position_secs || 0));

  renderCueMarkers(selectedTrack);
  renderCueCards(selectedTrack);
  renderStructureSegments(selectedTrack);
}

// Single Save & Teach AI Function: Saves cues & adapts neural models (Requirement 2)
async function saveAndTeachAI() {
  if (!selectedTrack) {
    showToast("Select a track to save changes.", "error");
    return;
  }

  const btn = btnSaveTeachAi;
  const origText = btn ? btn.textContent : "Save & Teach AI";
  if (btn) {
    btn.textContent = "Training...";
    btn.disabled = true;
  }

  try {
    const res = await fetch(`${API_BASE}/api/save_user_cues`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        track_id: selectedTrack.id,
        cues: selectedTrack.cues || [],
        bpm: selectedTrack.bpm || 120.0,
        teach_ai: true
      })
    });

    const data = await res.json();
    if (data.status === "ok") {
      showToast("HotCues saved and AI model calibrated successfully!", "success", 3000);
      if (data.cues) {
        selectedTrack.cues = data.cues;
        renderCueMarkers(selectedTrack);
        renderCueCards(selectedTrack);
        renderStructureSegments(selectedTrack);
      }
    } else {
      showToast(`Save error: ${data.error || 'Unknown'}`, "error");
    }
  } catch (err) {
    showToast(`Backend connection error: ${err.message}`, "error");
  } finally {
    if (btn) {
      btn.textContent = origText;
      btn.disabled = false;
    }
  }
}

// Audio Seeking & Continuous Dragging Playback
function seekToPosition(pct, updateAudio = true) {
  if (!selectedTrack) return;
  const duration = selectedTrack.duration_secs || 180;
  let targetSec = pct * duration;

  if (snapToGrid) {
    targetSec = snapTimestampToBeat(targetSec, selectedTrack);
    pct = Math.max(0, Math.min(1, targetSec / duration));
  }

  pendingSeekSec = targetSec;
  simulatedCurTime = targetSec;

  playhead.style.left = `${pct * 100}%`;
  if (miniOverviewProgress) miniOverviewProgress.style.width = `${pct * 100}%`;
  if (miniOverviewHandle) miniOverviewHandle.style.left = `${pct * 100}%`;
  if (playerScrubberProgress) playerScrubberProgress.style.width = `${pct * 100}%`;
  if (playerScrubberHandle) playerScrubberHandle.style.left = `${pct * 100}%`;

  const curFormatted = formatTime(targetSec);
  const remFormatted = `-${formatTime(Math.max(0, duration - targetSec))}`;

  if (timeDisplay) timeDisplay.textContent = curFormatted;
  if (timeRemainingDisplay) timeRemainingDisplay.textContent = remFormatted;
  if (playerTimeCur) playerTimeCur.textContent = curFormatted;
  if (playerTimeRem) playerTimeRem.textContent = remFormatted;

  if (updateAudio && currentAudio) {
    try { currentAudio.currentTime = targetSec; } catch (e) {}
  }
}

let playheadAnimFrame = null;
let lastSimulatedTime = 0;

function startPlayheadAnimation() {
  stopPlayheadAnimation();
  lastSimulatedTime = performance.now();

  function animate(now) {
    if (!isPlaying) return;

    const duration = (selectedTrack ? selectedTrack.duration_secs : 180) || 180;
    let curTime = 0;

    if (currentAudio && !currentAudio.paused && !isNaN(currentAudio.currentTime)) {
      curTime = currentAudio.currentTime;
    } else {
      const dt = (now - lastSimulatedTime) / 1000.0;
      simulatedCurTime += dt;
      if (simulatedCurTime >= duration) {
        if (isLooping) {
          simulatedCurTime = 0;
        } else {
          simulatedCurTime = 0;
          isPlaying = false;
          updatePlayButtonIcons(false);
          playhead.style.left = "0%";
          if (miniOverviewProgress) miniOverviewProgress.style.width = "0%";
          if (playerScrubberProgress) playerScrubberProgress.style.width = "0%";
          return;
        }
      }
      curTime = simulatedCurTime;
    }

    if (!isDraggingScrubber) {
      const pct = Math.min(100, Math.max(0, (curTime / duration) * 100));
      playhead.style.left = `${pct}%`;
      if (miniOverviewProgress) miniOverviewProgress.style.width = `${pct}%`;
      if (miniOverviewHandle) miniOverviewHandle.style.left = `${pct}%`;
      if (playerScrubberProgress) playerScrubberProgress.style.width = `${pct}%`;
      if (playerScrubberHandle) playerScrubberHandle.style.left = `${pct}%`;

      const curFormatted = formatTime(curTime);
      const remFormatted = `-${formatTime(Math.max(0, duration - curTime))}`;

      if (timeDisplay) timeDisplay.textContent = curFormatted;
      if (timeRemainingDisplay) timeRemainingDisplay.textContent = remFormatted;
      if (playerTimeCur) playerTimeCur.textContent = curFormatted;
      if (playerTimeRem) playerTimeRem.textContent = remFormatted;

      // Smooth auto-scroll when zoomed in
      if (zoomLevel > 1.0 && waveformCanvasContainer && waveformCanvas) {
        const containerWidth = waveformCanvasContainer.clientWidth;
        const totalWidth = waveformCanvas.width;
        const playheadPx = (pct / 100) * totalWidth;
        waveformCanvasContainer.scrollLeft = playheadPx - (containerWidth / 2);
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

function togglePlay() {
  if (!selectedTrack) {
    if (tracksData.length > 0) selectTrack(tracksData[0]);
    else return;
  }

  if (!currentAudio) {
    currentAudio = new Audio(`${API_BASE}/audio/?id=${selectedTrack.id}`);
    currentAudio.volume = playerVolumeSlider ? parseFloat(playerVolumeSlider.value) : 0.9;
  }

  if (isPlaying) {
    if (currentAudio) currentAudio.pause();
    isPlaying = false;
    stopPlayheadAnimation();
    updatePlayButtonIcons(false);
  } else {
    if (pendingSeekSec !== null) {
      try { currentAudio.currentTime = pendingSeekSec; } catch (e) {}
    }

    currentAudio.play().then(() => {
      isPlaying = true;
      updatePlayButtonIcons(true);
      startPlayheadAnimation();
    }).catch(() => {
      isPlaying = true;
      updatePlayButtonIcons(true);
      if (pendingSeekSec !== null) {
        simulatedCurTime = pendingSeekSec;
        pendingSeekSec = null;
      }
      startPlayheadAnimation();
    });
  }
}

// Delete Track Modal
function openDeleteModal(id, title) {
  pendingDeleteTrack = { id, title };
  if (deleteTrackTitleText) deleteTrackTitleText.textContent = `"${title}"`;
  if (modalDeleteConfirm) modalDeleteConfirm.classList.remove("hidden");
}

if (btnCloseDeleteModal) btnCloseDeleteModal.addEventListener("click", () => modalDeleteConfirm.classList.add("hidden"));
if (btnCancelDelete) btnCancelDelete.addEventListener("click", () => modalDeleteConfirm.classList.add("hidden"));
if (btnConfirmDelete) {
  btnConfirmDelete.addEventListener("click", async () => {
    if (!pendingDeleteTrack) return;
    const { id, title } = pendingDeleteTrack;
    modalDeleteConfirm.classList.add("hidden");
    try {
      const res = await fetch(`${API_BASE}/api/delete_track`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: id })
      });
      const data = await res.json();
      if (data.status === "ok") {
        if (selectedTrack && selectedTrack.id === id) selectedTrack = null;
        loadTracks();
      }
    } catch (err) {
      showToast(`Error deleting track: ${err.message}`, "error");
    }
  });
}

// Sidebar Collapse Toggle
if (btnCollapseSidebar) {
  btnCollapseSidebar.addEventListener("click", () => {
    if (sidebarNav) {
      sidebarNav.classList.toggle("collapsed");
      const isCol = sidebarNav.classList.contains("collapsed");
      localStorage.setItem("ah_sidebar_collapsed", isCol ? "1" : "0");
      requestAnimationFrame(() => {
        if (selectedTrack) {
          drawRGBWaveform(selectedTrack);
          renderCueMarkers(selectedTrack);
        }
      });
    }
  });
}

// Single Import & Batch Analysis
if (btnImport) {
  btnImport.addEventListener("click", async () => {
    modalImport.classList.remove("hidden");
    if (importAnalysisCard) importAnalysisCard.classList.add("hidden");
    loadDiscoveredFiles();
  });
}
if (btnCloseImportModal) btnCloseImportModal.addEventListener("click", () => modalImport.classList.add("hidden"));
if (btnCloseImportFooter) btnCloseImportFooter.addEventListener("click", () => modalImport.classList.add("hidden"));

if (dropzone) {
  dropzone.addEventListener("click", () => {
    if (filePicker) filePicker.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#3b82f6";
    dropzone.style.background = "rgba(59, 130, 246, 0.1)";
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
    dropzone.style.background = "rgba(0, 0, 0, 0.2)";
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
    dropzone.style.background = "rgba(0, 0, 0, 0.2)";
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      triggerSingleAnalysis(e.dataTransfer.files[0].name, e.dataTransfer.files[0]);
    }
  });
}

if (filePicker) {
  filePicker.addEventListener("change", (e) => {
    if (e.target.files.length > 0) triggerSingleAnalysis(e.target.files[0].name, e.target.files[0]);
  });
}
if (btnRunSingleImport) {
  btnRunSingleImport.addEventListener("click", () => {
    const val = importPathInput.value.trim();
    if (val) triggerSingleAnalysis(val);
  });
}

async function loadDiscoveredFiles() {
  if (!discoveredFilesList) return;
  discoveredFilesList.innerHTML = `<span style="font-size:11px; color:#64748b;">Scanning workspace...</span>`;

  try {
    const res = await fetch(`${API_BASE}/api/scan_files`);
    const data = await res.json();
    const files = data.files || [];

    if (files.length === 0) {
      discoveredFilesList.innerHTML = `<span style="font-size:11px; color:#64748b;">No new unanalyzed audio files found in workspace.</span>`;
      return;
    }

    discoveredFilesList.innerHTML = "";
    files.slice(0, 8).forEach(f => {
      const chip = document.createElement("div");
      chip.className = "discovered-file-chip";
      const name = f.split(/[\/\\]/).pop();
      chip.innerHTML = `
        <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:380px;">${escapeHtml(name)}</span>
        <button class="btn-chip-analyze">Analyze</button>
      `;
      chip.querySelector("button").addEventListener("click", () => {
        triggerSingleAnalysis(f);
      });
      discoveredFilesList.appendChild(chip);
    });
  } catch (e) {
    discoveredFilesList.innerHTML = `<span style="font-size:11px; color:#ef4444;">Could not scan workspace files.</span>`;
  }
}

async function triggerSingleAnalysis(filePath, fileBlob = null) {
  if (importAnalysisCard) {
    importAnalysisCard.classList.remove("hidden");
    if (importStatusTitle) importStatusTitle.textContent = "Analyzing audio file...";
    if (importStatusStep) importStatusStep.textContent = "Extracting ID3 tags, artwork & AI neural features...";
  }

  try {
    let res;
    if (fileBlob) {
      const base64Data = await fileToBase64(fileBlob);
      const response = await fetch(`${API_BASE}/api/upload_and_analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_name: filePath, base64_data: base64Data })
      });
      res = await response.json();
    } else {
      const response = await fetch(`${API_BASE}/api/analyze_file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: filePath })
      });
      res = await response.json();
    }

    if (res.status === "ok") {
      if (importStatusTitle) importStatusTitle.textContent = "Analysis complete!";
      if (importStatusStep) importStatusStep.textContent = `Key: ${res.result.camelot_key} | BPM: ${Math.round(res.result.bpm)} | Artist: ${res.result.artist || 'Unknown'}`;
      
      setTimeout(async () => {
        if (importAnalysisCard) importAnalysisCard.classList.add("hidden");
        modalImport.classList.add("hidden");
        await loadTracks();
        const newTrack = tracksData.find(t => t.id === res.result.track_id);
        if (newTrack) selectTrack(newTrack);
      }, 700);
    } else {
      if (importStatusTitle) importStatusTitle.textContent = "Analysis failed";
      if (importStatusStep) importStatusStep.textContent = res.error || "Unknown error";
      showToast(`Error: ${res.error}`, "error");
      setTimeout(() => {
        if (importAnalysisCard) importAnalysisCard.classList.add("hidden");
      }, 3000);
    }
  } catch (err) {
    if (importStatusTitle) importStatusTitle.textContent = "Analysis failed";
    if (importStatusStep) importStatusStep.textContent = err.message;
    showToast(`Error: ${err.message}`, "error");
    setTimeout(() => {
      if (importAnalysisCard) importAnalysisCard.classList.add("hidden");
    }, 3000);
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

// Batch Analysis with Real Interactive Feedback
if (btnBatch) {
  btnBatch.addEventListener("click", () => {
    modalBatch.classList.remove("hidden");
    if (batchResultsSummary) batchResultsSummary.classList.add("hidden");
    if (btnStartBatch) {
      btnStartBatch.textContent = "Start Batch Analysis";
      btnStartBatch.disabled = false;
    }
  });
}
if (btnCloseBatchModal) btnCloseBatchModal.addEventListener("click", () => modalBatch.classList.add("hidden"));

if (btnStartBatch) {
  btnStartBatch.addEventListener("click", async () => {
    if (btnStartBatch.textContent === "Done") {
      modalBatch.classList.add("hidden");
      return;
    }

    try {
      const scanRes = await fetch(`${API_BASE}/api/scan_files`);
      const scanData = await scanRes.json();
      const files = scanData.files || [];
      if (files.length === 0) {
        showToast("No audio files found on disk.", "error");
        return;
      }

      btnStartBatch.textContent = "Analyzing...";
      btnStartBatch.disabled = true;
      if (batchModalDesc) batchModalDesc.textContent = `Batch processing ${files.length} audio tracks in background...`;
      if (batchResultsSummary) batchResultsSummary.classList.add("hidden");

      const res = await fetch(`${API_BASE}/api/analyze_batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_paths: files })
      });
      const data = await res.json();
      if (data.status === "ok") {
        startBatchPolling();
      }
    } catch (err) {
      showToast(`Batch Error: ${err.message}`, "error");
      btnStartBatch.textContent = "Start Batch Analysis";
      btnStartBatch.disabled = false;
    }
  });
}

function startBatchPolling() {
  if (batchPollInterval) clearInterval(batchPollInterval);
  if (batchProgressContainer) batchProgressContainer.classList.remove("hidden");

  batchPollInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/batch_status`);
      const data = await res.json();
      if (data.status === "ok" && data.batch) {
        const b = data.batch;
        const total = b.total_files || 1;
        const done = b.processed_files || 0;
        const pct = Math.round((done / total) * 100);

        // Update Modal Progress
        if (batchModalProgressBar) batchModalProgressBar.style.width = `${pct}%`;
        if (batchModalPct) batchModalPct.textContent = `${pct}%`;
        if (batchModalFile) batchModalFile.textContent = b.current_file ? `Analyzing: ${b.current_file}` : "Processing...";
        if (batchModalCount) batchModalCount.textContent = `${done} of ${total} tracks`;
        if (batchModalSpeed) batchModalSpeed.textContent = `${b.tracks_per_sec || 0} tracks/sec`;
        if (batchModalEta) batchModalEta.textContent = `ETA: ${b.eta_seconds || 0}s`;

        // Update Sub-strip
        if (batchProgressInner) batchProgressInner.style.width = `${pct}%`;
        if (batchSpeed) batchSpeed.textContent = `${b.tracks_per_sec || 0} tracks/sec`;
        if (batchCount) batchCount.textContent = `${done} / ${total}`;

        if (!b.is_running && done >= total && total > 0) {
          clearInterval(batchPollInterval);
          if (batchProgressContainer) batchProgressContainer.classList.add("hidden");
          
          if (batchResultsSummary) {
            batchResultsSummary.innerHTML = `
              <strong>Batch Processing Complete!</strong><br>
              Analyzed ${total} tracks in ${(done / Math.max(0.1, b.tracks_per_sec || 1)).toFixed(1)}s.
            `;
            batchResultsSummary.classList.remove("hidden");
          }

          if (btnStartBatch) {
            btnStartBatch.textContent = "Done";
            btnStartBatch.disabled = false;
          }

          await loadTracks();
        }
      }
    } catch (err) {}
  }, 400);
}

// Rekordbox Export
if (btnExportXml) btnExportXml.addEventListener("click", () => modalExport.classList.remove("hidden"));
if (btnCloseExportModal) btnCloseExportModal.addEventListener("click", () => modalExport.classList.add("hidden"));
if (btnConfirmExport) {
  btnConfirmExport.addEventListener("click", async () => {
    const outPath = exportPathInput ? exportPathInput.value.trim() : "rekordbox.xml";
    try {
      const res = await fetch(`${API_BASE}/api/export_rekordbox`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ output_path: outPath })
      });
      const data = await res.json();
      if (data.status === "ok") {
        showToast(data.message, "success");
        modalExport.classList.add("hidden");
      }
    } catch (err) {
      showToast(`Export error: ${err.message}`, "error");
    }
  });
}

// Shortcuts Modal
if (btnShortcuts) btnShortcuts.addEventListener("click", () => modalShortcuts.classList.remove("hidden"));
if (btnCloseShortcutsModal) btnCloseShortcutsModal.addEventListener("click", () => modalShortcuts.classList.add("hidden"));
if (btnCloseShortcutsFooter) btnCloseShortcutsFooter.addEventListener("click", () => modalShortcuts.classList.add("hidden"));

// Continuous Draggable Scrubber & Waveform Seek
function initSeekListeners() {
  const onWaveformClick = (e) => {
    if (!selectedTrack) return;
    const target = waveformScrollWrapper || waveformCanvas || waveformCanvasContainer;
    if (!target) return;
    const rect = target.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.min(1, Math.max(0, clickX / rect.width));
    seekToPosition(pct);
  };

  if (waveformCanvasContainer) {
    waveformCanvasContainer.onclick = onWaveformClick;
    waveformCanvasContainer.addEventListener("wheel", handleWaveformWheel, { passive: false });
  }
  if (waveformScrollWrapper) waveformScrollWrapper.onclick = onWaveformClick;
  if (waveformCanvas) waveformCanvas.onclick = onWaveformClick;

  if (waveformMiniOverview) {
    waveformMiniOverview.onclick = (e) => {
      const rect = waveformMiniOverview.getBoundingClientRect();
      const pct = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      seekToPosition(pct);
    };
  }

  // Bottom Player Scrubber: Continuous Click & Drag
  if (playerWaveformScrubber) {
    const handleScrubberPointer = (e) => {
      if (!selectedTrack) return;
      const rect = playerWaveformScrubber.getBoundingClientRect();
      const pct = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      seekToPosition(pct, isPlaying);
    };

    playerWaveformScrubber.addEventListener("pointerdown", (e) => {
      isDraggingScrubber = true;
      if (playerScrubberHandle) playerScrubberHandle.classList.add("is-dragging");
      handleScrubberPointer(e);

      const onPointerMove = (moveEvt) => {
        if (isDraggingScrubber) {
          handleScrubberPointer(moveEvt);
        }
      };

      const onPointerUp = (upEvt) => {
        if (isDraggingScrubber) {
          isDraggingScrubber = false;
          if (playerScrubberHandle) playerScrubberHandle.classList.remove("is-dragging");
          handleScrubberPointer(upEvt);
          if (currentAudio && pendingSeekSec !== null) {
            try { currentAudio.currentTime = pendingSeekSec; } catch (err) {}
          }
        }
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };

      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
    });
  }
}

// Resizable Workspace Dividers & Independent Column Resizing (Section 3, 4, 5, 8)
function initResizers() {
  const DEFAULT_SIDEBAR_WIDTH = 180;
  const DEFAULT_COLLECTION_WIDTH = 410;
  const defaultColWidths = { track: 145, artist: 105, key: 55, bpm: 50, energy: 55 };

  // 1. Sidebar Resizer
  if (resizerSidebar && sidebarNav) {
    let isResizing = false;
    resizerSidebar.addEventListener("pointerdown", (e) => {
      isResizing = true;
      resizerSidebar.classList.add("resizing");
      const startX = e.clientX;
      const startW = sidebarNav.offsetWidth;

      const onMove = (mv) => {
        if (!isResizing) return;
        const newW = Math.max(60, Math.min(250, startW + (mv.clientX - startX)));
        sidebarNav.style.width = `${newW}px`;
      };
      const onUp = () => {
        if (isResizing) {
          isResizing = false;
          resizerSidebar.classList.remove("resizing");
          localStorage.setItem("ah_sidebar_width", sidebarNav.offsetWidth);
        }
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });

    // Double-click to restore default size (Silent reset without notification toast)
    resizerSidebar.addEventListener("dblclick", (e) => {
      e.preventDefault();
      e.stopPropagation();
      sidebarNav.style.width = `${DEFAULT_SIDEBAR_WIDTH}px`;
      localStorage.setItem("ah_sidebar_width", DEFAULT_SIDEBAR_WIDTH.toString());
      if (selectedTrack) {
        requestAnimationFrame(() => {
          drawRGBWaveform(selectedTrack);
          renderCueMarkers(selectedTrack);
        });
      }
    });

    const savedSidebarW = localStorage.getItem("ah_sidebar_width");
    if (savedSidebarW) sidebarNav.style.width = `${savedSidebarW}px`;
  }

  // 2. Collection Column Resizer
  if (resizerCollection && collectionColumn) {
    let isResizing = false;
    resizerCollection.addEventListener("pointerdown", (e) => {
      isResizing = true;
      resizerCollection.classList.add("resizing");
      const startX = e.clientX;
      const startW = collectionColumn.offsetWidth;

      const onMove = (mv) => {
        if (!isResizing) return;
        const newW = Math.max(280, Math.min(650, startW + (mv.clientX - startX)));
        collectionColumn.style.width = `${newW}px`;
      };
      const onUp = () => {
        if (isResizing) {
          isResizing = false;
          resizerCollection.classList.remove("resizing");
          localStorage.setItem("ah_collection_width", collectionColumn.offsetWidth);
        }
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });

    // Double-click to restore default size (Silent reset without notification toast)
    resizerCollection.addEventListener("dblclick", (e) => {
      e.preventDefault();
      e.stopPropagation();
      collectionColumn.style.width = `${DEFAULT_COLLECTION_WIDTH}px`;
      localStorage.setItem("ah_collection_width", DEFAULT_COLLECTION_WIDTH.toString());
      if (selectedTrack) {
        requestAnimationFrame(() => {
          drawRGBWaveform(selectedTrack);
          renderCueMarkers(selectedTrack);
        });
      }
    });

    const savedColW = localStorage.getItem("ah_collection_width");
    if (savedColW) collectionColumn.style.width = `${savedColW}px`;
  }

  // 3. HotCue & Harmonic Mixing Resizer (True Two-Way Splitter - Section 5 & 8)
  if (resizerWidgets && hotcueWidget && harmonicWidget) {
    let isResizing = false;
    resizerWidgets.addEventListener("pointerdown", (e) => {
      isResizing = true;
      resizerWidgets.classList.add("resizing");
      const container = document.getElementById("workstation-dual-widgets");
      const containerRect = container.getBoundingClientRect();
      const totalW = containerRect.width;

      const onMove = (mv) => {
        if (!isResizing) return;
        const pointerX = mv.clientX - containerRect.left;
        const newHotcueW = Math.max(160, Math.min(totalW - 160, pointerX));
        hotcueWidget.style.flex = `0 0 ${newHotcueW}px`;
        harmonicWidget.style.flex = `1 1 0`;
      };

      const onUp = () => {
        if (isResizing) {
          isResizing = false;
          resizerWidgets.classList.remove("resizing");
        }
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });

    // Double-click to restore default size (Silent reset without notification toast)
    resizerWidgets.addEventListener("dblclick", (e) => {
      e.preventDefault();
      e.stopPropagation();
      hotcueWidget.style.flex = `1 1 0`;
      harmonicWidget.style.flex = `1 1 0`;
    });
  }

  // 4. Independent Table Column Resizers (Section 3, 4, 8)
  if (collectionTableHeader) {
    const minWidths = { track: 80, artist: 70, key: 45, bpm: 40, energy: 45 };
    const maxWidths = { track: 320, artist: 220, key: 100, bpm: 90, energy: 90 };
    const savedWidths = JSON.parse(localStorage.getItem("ah_col_widths_v3") || "{}");

    const ths = collectionTableHeader.querySelectorAll(".th-resizable");
    ths.forEach(th => {
      const colName = th.dataset.col;
      if (savedWidths[colName]) {
        th.style.width = `${savedWidths[colName]}px`;
      }

      const resizer = th.querySelector(".col-resizer");
      if (resizer) {
        resizer.addEventListener("pointerdown", (e) => {
          e.stopPropagation();
          e.preventDefault();
          const startX = e.clientX;
          const startW = th.offsetWidth;
          resizer.classList.add("resizing");

          const onMove = (mv) => {
            const minW = minWidths[colName] || 40;
            const maxW = maxWidths[colName] || 400;
            const newW = Math.max(minW, Math.min(maxW, startW + (mv.clientX - startX)));
            th.style.width = `${newW}px`;
          };

          const onUp = () => {
            resizer.classList.remove("resizing");
            const currentWidths = {};
            ths.forEach(t => {
              currentWidths[t.dataset.col] = t.offsetWidth;
            });
            localStorage.setItem("ah_col_widths_v3", JSON.stringify(currentWidths));
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
          };

          window.addEventListener("pointermove", onMove);
          window.addEventListener("pointerup", onUp);
        });

        // Double-click on column resizer resets ONLY that column to default (Silent reset)
        resizer.addEventListener("dblclick", (e) => {
          e.preventDefault();
          e.stopPropagation();
          const defaultW = defaultColWidths[colName] || 100;
          th.style.width = `${defaultW}px`;
          const currentWidths = JSON.parse(localStorage.getItem("ah_col_widths_v3") || "{}");
          currentWidths[colName] = defaultW;
          localStorage.setItem("ah_col_widths_v3", JSON.stringify(currentWidths));
        });
      }
    });
  }

  // 5. Waveform Auto-Resize Observer (Prevents black dead space - Section 6)
  if (window.ResizeObserver && waveformCanvasContainer) {
    const ro = new ResizeObserver(() => {
      if (selectedTrack) {
        drawRGBWaveform(selectedTrack);
        renderCueMarkers(selectedTrack);
      }
    });
    ro.observe(waveformCanvasContainer);
  }
}

// Persistent Player Controls
if (btnPlayerPlay) btnPlayerPlay.addEventListener("click", togglePlay);
if (btnPlay) btnPlay.addEventListener("click", togglePlay);

if (btnPlayerPrev) {
  btnPlayerPrev.addEventListener("click", () => {
    if (tracksData.length === 0) return;
    const curIdx = selectedTrack ? tracksData.findIndex(t => t.id === selectedTrack.id) : 0;
    const prevIdx = (curIdx - 1 + tracksData.length) % tracksData.length;
    selectTrack(tracksData[prevIdx]);
    togglePlay();
  });
}

if (btnPlayerNext) {
  btnPlayerNext.addEventListener("click", () => {
    if (tracksData.length === 0) return;
    const curIdx = selectedTrack ? tracksData.findIndex(t => t.id === selectedTrack.id) : 0;
    const nextIdx = (curIdx + 1) % tracksData.length;
    selectTrack(tracksData[nextIdx]);
    togglePlay();
  });
}

if (btnPlayerRepeat) {
  btnPlayerRepeat.addEventListener("click", () => {
    isLooping = !isLooping;
    btnPlayerRepeat.style.color = isLooping ? "#3b82f6" : "var(--text-secondary)";
  });
}

if (btnPlayerShuffle) {
  btnPlayerShuffle.addEventListener("click", () => {
    isShuffled = !isShuffled;
    btnPlayerShuffle.style.color = isShuffled ? "#3b82f6" : "var(--text-secondary)";
  });
}

if (playerVolumeSlider) {
  playerVolumeSlider.addEventListener("input", (e) => {
    const vol = parseFloat(e.target.value);
    if (currentAudio) currentAudio.volume = vol;
  });
}

// Sidebar Navigation
if (navCollection) {
  navCollection.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    navCollection.classList.add("active");
  });
}
if (navAnalyze) navAnalyze.addEventListener("click", () => btnImport.click());
if (navSettings) navSettings.addEventListener("click", () => modalShortcuts.classList.remove("hidden"));

// BPM Calibration
if (inputBpm) {
  inputBpm.addEventListener("change", () => {
    if (!selectedTrack) return;
    const newBpm = parseFloat(inputBpm.value);
    if (!isNaN(newBpm) && newBpm >= 40 && newBpm <= 260) {
      selectedTrack.bpm = newBpm;
      drawRGBWaveform(selectedTrack);
      if (heroBpmVal) heroBpmVal.textContent = Math.round(newBpm).toString();
      if (playerBadgeBpm) playerBadgeBpm.textContent = Math.round(newBpm).toString();
    }
  });
}

if (btnBpmHalf) {
  btnBpmHalf.addEventListener("click", () => {
    if (!selectedTrack || !selectedTrack.bpm) return;
    selectedTrack.bpm = parseFloat((selectedTrack.bpm / 2).toFixed(2));
    if (inputBpm) inputBpm.value = selectedTrack.bpm.toFixed(2);
    if (heroBpmVal) heroBpmVal.textContent = Math.round(selectedTrack.bpm).toString();
    if (playerBadgeBpm) playerBadgeBpm.textContent = Math.round(selectedTrack.bpm).toString();
    drawRGBWaveform(selectedTrack);
  });
}

if (btnBpmDouble) {
  btnBpmDouble.addEventListener("click", () => {
    if (!selectedTrack || !selectedTrack.bpm) return;
    selectedTrack.bpm = parseFloat((selectedTrack.bpm * 2).toFixed(2));
    if (inputBpm) inputBpm.value = selectedTrack.bpm.toFixed(2);
    if (heroBpmVal) heroBpmVal.textContent = Math.round(selectedTrack.bpm).toString();
    if (playerBadgeBpm) playerBadgeBpm.textContent = Math.round(selectedTrack.bpm).toString();
    drawRGBWaveform(selectedTrack);
  });
}

if (btnBpmTap) {
  btnBpmTap.addEventListener("click", () => {
    const now = Date.now();
    tapTimes.push(now);
    tapTimes = tapTimes.filter(t => (now - t) <= 3000);
    if (tapTimes.length >= 3) {
      const intervals = [];
      for (let i = 1; i < tapTimes.length; i++) intervals.push(tapTimes[i] - tapTimes[i - 1]);
      const avgMs = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      if (avgMs > 0) {
        const calculatedBpm = parseFloat((60000 / avgMs).toFixed(2));
        if (calculatedBpm >= 40 && calculatedBpm <= 250) {
          if (selectedTrack) {
            selectedTrack.bpm = calculatedBpm;
            drawRGBWaveform(selectedTrack);
            if (heroBpmVal) heroBpmVal.textContent = Math.round(calculatedBpm).toString();
            if (playerBadgeBpm) playerBadgeBpm.textContent = Math.round(calculatedBpm).toString();
          }
          if (inputBpm) inputBpm.value = calculatedBpm.toFixed(2);
        }
      }
    }
  });
}

if (btnSetFirstBeat) {
  btnSetFirstBeat.addEventListener("click", () => {
    if (!selectedTrack) return;
    const curSec = getCurrentPlayheadSec();
    selectedTrack.first_beat_offset = curSec;
    drawRGBWaveform(selectedTrack);
    renderCueMarkers(selectedTrack);
  });
}

if (btnGridNudgeLeft) {
  btnGridNudgeLeft.addEventListener("click", () => {
    if (!selectedTrack) return;
    selectedTrack.first_beat_offset = (selectedTrack.first_beat_offset || 0) - 0.005;
    drawRGBWaveform(selectedTrack);
  });
}

if (btnGridNudgeRight) {
  btnGridNudgeRight.addEventListener("click", () => {
    if (!selectedTrack) return;
    selectedTrack.first_beat_offset = (selectedTrack.first_beat_offset || 0) + 0.005;
    drawRGBWaveform(selectedTrack);
  });
}

if (btnToggleBeatgrid) {
  btnToggleBeatgrid.addEventListener("click", () => {
    showBeatgrid = !showBeatgrid;
    btnToggleBeatgrid.classList.toggle("active", showBeatgrid);
    if (selectedTrack) drawRGBWaveform(selectedTrack);
  });
}

if (btnToggleSnap) {
  btnToggleSnap.addEventListener("click", () => {
    snapToGrid = !snapToGrid;
    btnToggleSnap.classList.toggle("active", snapToGrid);
  });
}

// Zoom Button Handlers
if (btnZoomIn) btnZoomIn.addEventListener("click", () => setZoom(zoomLevel * 1.5));
if (btnZoomOut) btnZoomOut.addEventListener("click", () => setZoom(zoomLevel / 1.5));
if (btnZoomReset) btnZoomReset.addEventListener("click", () => setZoom(1.0));

if (btnAddCuePlayhead) btnAddCuePlayhead.addEventListener("click", addCueAtPlayhead);

let isEditModeActive = false;
if (btnEditCues) {
  btnEditCues.addEventListener("click", () => {
    if (!selectedTrack || !selectedTrack.cues || selectedTrack.cues.length === 0) return;

    isEditModeActive = !isEditModeActive;
    btnEditCues.classList.toggle("active", isEditModeActive);

    const rows = document.querySelectorAll(".hotcue-pill-row");
    let targetRow = document.querySelector(".hotcue-pill-row.active") || rows[0];
    if (targetRow) {
      const rowIndex = Array.from(rows).indexOf(targetRow);
      if (rowIndex !== -1 && selectedTrack.cues[rowIndex]) {
        editHotCueRow(targetRow, selectedTrack.cues[rowIndex], selectedTrack);
      }
    }
  });
}

if (btnSaveTeachAi) btnSaveTeachAi.addEventListener("click", saveAndTeachAI);

if (searchInput) searchInput.addEventListener("input", loadTracks);
if (filterCamelot) filterCamelot.addEventListener("change", loadTracks);
if (filterEnergy) filterEnergy.addEventListener("change", loadTracks);

// Global DJ Keyboard Shortcuts
window.addEventListener("keydown", (e) => {
  const activeEl = document.activeElement;
  if (activeEl && (activeEl.tagName === "INPUT" || activeEl.tagName === "TEXTAREA" || activeEl.tagName === "SELECT")) {
    if (e.key === "Escape") activeEl.blur();
    return;
  }

  if (e.key === "Escape") {
    document.querySelectorAll(".modal-backdrop").forEach(m => m.classList.add("hidden"));
    return;
  }

  if ((e.ctrlKey || e.metaKey) && (e.key === "b" || e.key === "B")) {
    e.preventDefault();
    if (btnCollapseSidebar) btnCollapseSidebar.click();
    return;
  }

  if (e.key === "?" || e.key === "F1") {
    e.preventDefault();
    if (modalShortcuts) modalShortcuts.classList.toggle("hidden");
    return;
  }

  if (e.code === "Space" || e.key === " ") {
    e.preventDefault();
    togglePlay();
    return;
  }

  if (e.shiftKey && (e.key === "M" || e.key === "m")) {
    e.preventDefault();
    if (btnSetFirstBeat) btnSetFirstBeat.click();
    return;
  }

  if (e.key === "m" || e.key === "M") {
    e.preventDefault();
    addCueAtPlayhead();
    return;
  }

  if (e.key === "b" || e.key === "B") {
    e.preventDefault();
    if (btnToggleBeatgrid) btnToggleBeatgrid.click();
    return;
  }

  if (e.key === "s" || e.key === "S") {
    e.preventDefault();
    if (btnToggleSnap) btnToggleSnap.click();
    return;
  }

  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    if (!selectedTrack) return;
    const dur = selectedTrack.duration_secs || 180;
    const bpm = selectedTrack.bpm || 120.0;
    const beatSec = 60.0 / bpm;
    const stepSec = e.shiftKey ? (beatSec * 4) : beatSec;
    const curSec = (currentAudio && !isNaN(currentAudio.currentTime)) ? currentAudio.currentTime : (simulatedCurTime || 0);
    const targetSec = Math.max(0, Math.min(dur, e.key === "ArrowLeft" ? (curSec - stepSec) : (curSec + stepSec)));
    seekToPosition(targetSec / dur);
    return;
  }

  if (/^[1-8]$/.test(e.key)) {
    const cueIndex = parseInt(e.key, 10) - 1;
    if (selectedTrack && selectedTrack.cues && selectedTrack.cues[cueIndex]) {
      e.preventDefault();
      const cue = selectedTrack.cues[cueIndex];
      const dur = selectedTrack.duration_secs || 180;
      seekToPosition(cue.position_secs / dur);
      return;
    }
  }

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
  }
});

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  initSeekListeners();
  initResizers();
  
  const savedColState = localStorage.getItem("ah_sidebar_collapsed");
  if (savedColState === "1" && sidebarNav) {
    sidebarNav.classList.add("collapsed");
  }

  loadTracks();
});

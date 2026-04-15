/**
 * live.js — NeuroGuard Live Detection
 * ══════════════════════════════════════
 * Captures webcam frames, sends to Flask backend (/api/predict),
 * and renders real-time drowsiness / distraction results.
 * Falls back to a client-side simulation if backend is offline.
 */

const API_BASE       = "http://localhost:5050";
const CAPTURE_FPS    = 5;   // frames per second sent to API
const CAPTURE_ITV    = 1000 / CAPTURE_FPS;
const ALERT_THRESH   = "DROWSY";
const MAX_LOG        = 80;

// ─── State ───────────────────────────────────────────
let stream          = null;
let running         = false;
let apiOnline       = false;
let sessionStart    = null;
let captureInterval = null;
let clockInterval   = null;
let totalFrames     = 0;
let alertCount      = 0;
let confHistory     = [];
let lastState       = null;
let alertDismissed  = false;
let spikeData       = new Array(60).fill(0);
let simMode         = false;   // true if backend offline

// Performance tracking
let fpsCounter      = 0;
let fpsTimer        = Date.now();

// ─── DOM refs ────────────────────────────────────────
const startBtn       = document.getElementById("startBtn");
const stopBtn        = document.getElementById("stopBtn");
const resetBtn       = document.getElementById("resetBtn");
const video          = document.getElementById("webcamVideo");
const overlayCanvas  = document.getElementById("overlayCanvas");
const overlayCtx     = overlayCanvas.getContext("2d");
const camIdle        = document.getElementById("camIdle");
const hudOverlay     = document.getElementById("hudOverlay");
const hudStateBadge  = document.getElementById("hudStateBadge");
const hudStateText   = document.getElementById("hudStateText");
const hsbDot         = document.getElementById("hsbDot");
const modeLabel      = document.getElementById("modeLabel");
const sessionTimeEl  = document.getElementById("sessionTime");
const fpsDisplay     = document.getElementById("fpsDisplay");
const camDot         = document.getElementById("camDot");

const apiStatusEl    = document.getElementById("apiStatus");
const apiDot         = apiStatusEl.querySelector(".api-dot");
const apiText        = apiStatusEl.querySelector(".api-text");

const drowsyAlert    = document.getElementById("drowsyAlert");
const resultCard     = document.getElementById("resultCard");
const rclEmoji       = document.getElementById("rclEmoji");
const rclLabel       = document.getElementById("rclLabel");
const rclConf        = document.getElementById("rclConf");
const rclBar         = document.getElementById("rclBar");
const logBody        = document.getElementById("logBody");
const evRate         = document.getElementById("evRate");

const streamCanvas   = document.getElementById("eventStreamCanvas");
const streamCtx      = streamCanvas.getContext("2d");
const spikeCanvas    = document.getElementById("spikeCanvas");
const spikeCtx       = spikeCanvas.getContext("2d");

// ─── Capture canvas (hidden) ─────────────────────────
const capCanvas  = document.createElement("canvas");
const capCtx     = capCanvas.getContext("2d");
capCanvas.width  = 320;
capCanvas.height = 240;

// ─────────────────────────────────────────────────────
// API Status Check
// ─────────────────────────────────────────────────────
async function checkAPI() {
  try {
    const r = await fetch(`${API_BASE}/api/status`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) {
      const d = await r.json();
      setAPIOnline(true, d.mode);
      return true;
    }
  } catch (_) {}
  setAPIOnline(false);
  return false;
}

function setAPIOnline(online, mode = "") {
  apiOnline = online;
  if (online) {
    apiDot.className = "api-dot online";
    apiText.textContent = mode.includes("SNN") ? "SNN Model Active" : "API Online (Heuristic)";
    apiStatusEl.classList.add("online");
    modeLabel.textContent = mode.includes("SNN") ? "SNN" : "HEURISTIC";
    document.getElementById("setupCard").style.display = "none";
  } else {
    apiDot.className = "api-dot offline";
    apiText.textContent = "API Offline";
    apiStatusEl.classList.remove("online");
    modeLabel.textContent = "SIMULATED";
    document.getElementById("setupCard").style.display = "flex";
  }
}

// ─────────────────────────────────────────────────────
// Webcam
// ─────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    video.play();
    video.classList.add("active");
    overlayCanvas.classList.add("active");
    overlayCanvas.width  = video.videoWidth  || 640;
    overlayCanvas.height = video.videoHeight || 480;
    return true;
  } catch (err) {
    addLog(`Camera error: ${err.message}`, "alert");
    return false;
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.classList.remove("active");
  overlayCanvas.classList.remove("active");
}

// ─────────────────────────────────────────────────────
// Frame capture → API
// ─────────────────────────────────────────────────────
async function captureAndPredict() {
  if (!running) return;

  // Draw current video frame to hidden canvas
  capCtx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
  const b64 = capCanvas.toDataURL("image/jpeg", 0.75);

  // FPS counting
  fpsCounter++;
  const now = Date.now();
  if (now - fpsTimer >= 1000) {
    fpsDisplay.textContent = `${fpsCounter} FPS`;
    fpsCounter = 0;
    fpsTimer = now;
  }

  totalFrames++;
  document.getElementById("totalFrames").textContent = totalFrames;

  let result;
  if (apiOnline) {
    result = await callAPI(b64);
  }
  if (!result) {
    // Simulation mode
    result = simulateDetection();
    simMode = true;
  } else {
    simMode = false;
  }

  renderResult(result);
  drawOverlay(result);
  updateEventStream(result);
  updateSpikes(result);
}

async function callAPI(b64frame) {
  try {
    const resp = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: b64frame }),
      signal: AbortSignal.timeout(3000),
    });
    if (resp.ok) return await resp.json();
  } catch (_) {
    setAPIOnline(false);
  }
  return null;
}

// ─────────────────────────────────────────────────────
// Render Results
// ─────────────────────────────────────────────────────
const STATE_CONFIG = {
  ALERT:  { emoji:"😊", color:"var(--accent-green)", cls:"state-alert",  dot:"alert",  hudColor:"#10b981" },
  TIRED:  { emoji:"😪", color:"var(--accent-amber)", cls:"state-tired",  dot:"tired",  hudColor:"#f59e0b" },
  DROWSY: { emoji:"😴", color:"var(--accent-red)",   cls:"state-drowsy", dot:"drowsy", hudColor:"#ef4444" },
  NO_FACE:{ emoji:"🔍", color:"var(--text-muted)",   cls:"",             dot:"noface", hudColor:"#4a6080" },
};

function renderResult(r) {
  const state  = r.state || "NO_FACE";
  const cfg    = STATE_CONFIG[state] || STATE_CONFIG["NO_FACE"];
  const conf   = r.confidence || 0;
  const perc   = r.perclos    || 0;
  const blink  = r.blink_rate || 0;
  const droop  = r.head_droop || 0;

  // Result card
  rclEmoji.textContent = cfg.emoji;
  rclLabel.textContent = r.label || state;
  rclLabel.className   = `rcl-label ${cfg.cls}`;
  rclConf.textContent  = `${conf.toFixed(1)}%`;
  rclBar.style.width   = `${conf}%`;
  rclBar.style.background = conf > 80
    ? state === "ALERT" ? "linear-gradient(90deg,#10b981,#34d399)"
    : state === "DROWSY" ? "linear-gradient(90deg,#ef4444,#f87171)"
    : "linear-gradient(90deg,#f59e0b,#fbbf24)"
    : "linear-gradient(90deg,#00d4ff,#7c3aed)";

  resultCard.className = `result-card-live ${cfg.cls}`;

  // HUD
  hudStateText.textContent = r.label || state;
  hsbDot.className = `hsb-dot ${cfg.dot}`;
  hudStateBadge.style.color = cfg.hudColor;

  // Metrics
  setMetricBar("perclosBar", perc, 100);
  document.getElementById("perclosVal").textContent = `${perc.toFixed(1)}%`;
  setMetricBar("blinkBar", blink, 60, true);
  document.getElementById("blinkVal").textContent = `${blink.toFixed(1)}/min`;
  setMetricBar("droopBar", droop, 45);
  document.getElementById("droopVal").textContent = `${droop.toFixed(1)}°`;

  // Average confidence
  confHistory.push(conf);
  if (confHistory.length > 100) confHistory.shift();
  const avg = confHistory.reduce((a,b)=>a+b,0)/confHistory.length;
  document.getElementById("avgConf").textContent = `${avg.toFixed(1)}%`;

  // Alert banner
  if (state === "DROWSY" && !alertDismissed) {
    drowsyAlert.classList.remove("hidden");
    if (lastState !== "DROWSY") {
      alertCount++;
      document.getElementById("alertCount").textContent = alertCount;
      addLog(`🚨 DROWSINESS DETECTED — confidence ${conf.toFixed(1)}%`, "alert");
    }
  } else if (state === "ALERT") {
    drowsyAlert.classList.add("hidden");
    alertDismissed = false;
    if (lastState === "DROWSY") addLog("✅ Driver alert again.", "ok");
  } else if (state === "TIRED" && lastState !== "TIRED") {
    addLog(`⚠️ Fatigue detected — PERCLOS ${perc.toFixed(0)}%`, "warn");
  }

  lastState = state;
}

function setMetricBar(id, val, max, isBlink = false) {
  const bar = document.getElementById(id);
  if (!bar) return;
  const pct = Math.min((val / max) * 100, 100);
  bar.style.width = pct + "%";
  // Color
  if (isBlink) {
    bar.className = `mi-bar ${val < 10 ? "danger" : val > 30 ? "warn" : "safe"}`;
  } else {
    bar.className = `mi-bar ${pct > 60 ? "danger" : pct > 30 ? "warn" : ""}`;
  }
}

// ─────────────────────────────────────────────────────
// Canvas face/eye overlay
// ─────────────────────────────────────────────────────
function drawOverlay(r) {
  overlayCanvas.width  = video.videoWidth  || 640;
  overlayCanvas.height = video.videoHeight || 480;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (!r.face_bbox && !r.face_det) return;

  const bbox = r.face_bbox;
  if (!bbox) return;

  // Mirror x coordinate for mirrored video
  const W  = overlayCanvas.width;
  const mx = W - bbox[0] - bbox[2];

  const state = r.state || "ALERT";
  const color = state === "DROWSY" ? "#ef4444"
              : state === "TIRED"  ? "#f59e0b"
              : "#00d4ff";

  // Face bounding box
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth   = 2;
  overlayCtx.setLineDash([6, 4]);
  overlayCtx.strokeRect(mx, bbox[1], bbox[2], bbox[3]);
  overlayCtx.setLineDash([]);

  // Corner accents
  const cs = 14;
  overlayCtx.lineWidth = 3;
  [[mx,bbox[1]],[mx+bbox[2],bbox[1]],[mx,bbox[1]+bbox[3]],[mx+bbox[2],bbox[1]+bbox[3]]].forEach(([cx,cy],i)=>{
    const sx = i<2 ? 1 : 1;
    const ex = i<2 ? (i===0?cs:-cs) : (i===2?cs:-cs);
    const ey = i%2===0 || i>1 ? (i<2?cs:-cs) : (i<2?cs:-cs);
    overlayCtx.beginPath();
    overlayCtx.moveTo(cx, cy + (i<2?cs:-(cs)));
    overlayCtx.lineTo(cx, cy);
    overlayCtx.lineTo(cx + (i===0||i===2?cs:-cs), cy);
    overlayCtx.stroke();
  });

  // State label above face
  overlayCtx.fillStyle = color;
  overlayCtx.font = "bold 13px Inter, sans-serif";
  overlayCtx.textAlign = "center";
  overlayCtx.fillText(state, mx + bbox[2] / 2, bbox[1] - 8);

  // Eyes indicator
  const eyeStr = r.eyes_open === 2 ? "EYES OPEN" : r.eyes_open === 1 ? "ONE EYE" : "EYES CLOSED";
  overlayCtx.font = "11px JetBrains Mono, monospace";
  overlayCtx.fillStyle = r.eyes_open > 0 ? "#10b981" : "#ef4444";
  overlayCtx.fillText(eyeStr, mx + bbox[2] / 2, bbox[1] + bbox[3] + 16);
}

// ─────────────────────────────────────────────────────
// Event Stream Visualizer
// ─────────────────────────────────────────────────────
let eventCount = 0;
function updateEventStream(r) {
  const W = streamCanvas.offsetWidth || 600;
  streamCanvas.width  = W * window.devicePixelRatio;
  streamCanvas.height = 60 * window.devicePixelRatio;
  streamCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
  const H = 60;

  streamCtx.fillStyle = "rgba(10,16,28,0.3)";
  streamCtx.fillRect(0, 0, W, H);

  const state = r.state || "ALERT";
  const density = state === "ALERT" ? 18 : state === "TIRED" ? 10 : 4;
  eventCount = state === "ALERT" ? 4200 + Math.floor(Math.random()*400)
             : state === "TIRED" ? 2100 + Math.floor(Math.random()*300)
             : 500  + Math.floor(Math.random()*200);

  evRate.textContent = eventCount.toLocaleString() + " ev/s";

  for (let i = 0; i < density; i++) {
    const x = Math.random() * W;
    const y = Math.random() * H;
    const isPos = Math.random() > 0.5;
    streamCtx.fillStyle = isPos
      ? `rgba(0,212,255,${Math.random() * 0.8 + 0.2})`
      : `rgba(249,115,22,${Math.random() * 0.8 + 0.2})`;
    streamCtx.beginPath();
    streamCtx.arc(x, y, 1.5, 0, Math.PI * 2);
    streamCtx.fill();
  }
}

// ─────────────────────────────────────────────────────
// Spike Activity Canvas
// ─────────────────────────────────────────────────────
function updateSpikes(r) {
  const state = r.state || "ALERT";
  const baseAct = state === "ALERT" ? 0.75 : state === "TIRED" ? 0.45 : 0.15;
  spikeData.push(Math.random() < baseAct ? Math.random() * 0.8 + 0.2 : 0);
  if (spikeData.length > 60) spikeData.shift();

  const W = spikeCanvas.offsetWidth || 320;
  spikeCanvas.width  = W * window.devicePixelRatio;
  spikeCanvas.height = 80 * window.devicePixelRatio;
  spikeCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
  const H = 80;

  spikeCtx.clearRect(0, 0, W, H);

  spikeData.forEach((val, i) => {
    if (val === 0) return;
    const x = (i / 59) * W;
    const sh = val * (H * 0.7);
    const g = spikeCtx.createLinearGradient(0, H/2 - sh, 0, H/2);
    g.addColorStop(0, "rgba(0,212,255,0.9)");
    g.addColorStop(1, "rgba(0,212,255,0.05)");
    spikeCtx.fillStyle = g;
    spikeCtx.fillRect(x - 1, H/2 - sh, 2, sh);
    spikeCtx.fillRect(x - 1, H/2, 2, sh * 0.4);
  });

  spikeCtx.strokeStyle = "rgba(0,212,255,0.12)";
  spikeCtx.lineWidth = 1;
  spikeCtx.beginPath();
  spikeCtx.moveTo(0, H/2); spikeCtx.lineTo(W, H/2);
  spikeCtx.stroke();
}

// ─────────────────────────────────────────────────────
// Session clock
// ─────────────────────────────────────────────────────
function startClock() {
  sessionStart = Date.now();
  clockInterval = setInterval(() => {
    const s = Math.floor((Date.now() - sessionStart) / 1000);
    const m = Math.floor(s / 60);
    sessionTimeEl.textContent = `${String(m).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;
  }, 1000);
}
function stopClock() {
  clearInterval(clockInterval);
  sessionTimeEl.textContent = "00:00";
}

// ─────────────────────────────────────────────────────
// Simulation (offline mode)
// ─────────────────────────────────────────────────────
let simStep = 0;
function simulateDetection() {
  simStep++;
  const cycle = Math.floor(simStep / 30) % 3;
  if (cycle === 0) return {
    state:"ALERT", label:"DRIVER ALERT 😊", confidence:95+Math.random()*4,
    perclos:5+Math.random()*10, blink_rate:16+Math.random()*4,
    head_droop:1+Math.random()*3, eyes_open:2, face_det:true, face_bbox:null
  };
  if (cycle === 1) return {
    state:"TIRED", label:"FATIGUE DETECTED 😪", confidence:80+Math.random()*12,
    perclos:35+Math.random()*20, blink_rate:24+Math.random()*8,
    head_droop:10+Math.random()*10, eyes_open:1, face_det:true, face_bbox:null
  };
  return {
    state:"DROWSY", label:"DROWSY 😴", confidence:90+Math.random()*8,
    perclos:75+Math.random()*15, blink_rate:6+Math.random()*4,
    head_droop:25+Math.random()*15, eyes_open:0, face_det:true, face_bbox:null
  };
}

// ─────────────────────────────────────────────────────
// Log
// ─────────────────────────────────────────────────────
function addLog(msg, type = "info") {
  const now = new Date();
  const ts  = `${String(now.getHours()).padStart(2,"0")}:${String(now.getMinutes()).padStart(2,"0")}:${String(now.getSeconds()).padStart(2,"0")}`;
  const el  = document.createElement("div");
  el.className = `log-entry ${type}`;
  el.textContent = `[${ts}] ${msg}`;
  logBody.appendChild(el);
  logBody.scrollTop = logBody.scrollHeight;
  // Trim
  while (logBody.children.length > MAX_LOG) logBody.removeChild(logBody.firstChild);
}

// ─────────────────────────────────────────────────────
// Start / Stop
// ─────────────────────────────────────────────────────
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Starting...";

  const apiOk = await checkAPI();
  addLog(apiOk ? `✅ Backend connected (${apiOnline ? "API" : "Heuristic"} mode)` : "⚠️ Backend offline — simulation mode", apiOk ? "ok" : "warn");

  const camOk = await startCamera();
  if (!camOk) {
    startBtn.disabled = false;
    startBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16"/></svg> Start Detection`;
    return;
  }

  running = true;
  camIdle.classList.add("hidden");
  hudOverlay.style.display = "block";
  camDot.classList.add("recording");
  startClock();

  captureInterval = setInterval(captureAndPredict, CAPTURE_ITV);

  startBtn.disabled = true;
  startBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16"/></svg> Running...`;
  stopBtn.disabled = false;
  addLog("🎥 Detection started.", "ok");
});

stopBtn.addEventListener("click", stopDetection);

function stopDetection() {
  running = false;
  clearInterval(captureInterval);
  stopCamera();
  stopClock();
  camIdle.classList.remove("hidden");
  hudOverlay.style.display = "none";
  camDot.classList.remove("recording");
  drowsyAlert.classList.add("hidden");
  alertDismissed = false;

  startBtn.disabled  = false;
  startBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16"/></svg> Start Detection`;
  stopBtn.disabled   = true;
  addLog("⏹ Detection stopped.", "info");

  // Reset API to check status on next start
  fetch(`${API_BASE}/api/reset`, { method: "POST" }).catch(() => {});
}

resetBtn.addEventListener("click", () => {
  totalFrames = 0; alertCount = 0; confHistory = []; lastState = null;
  document.getElementById("totalFrames").textContent = "0";
  document.getElementById("alertCount").textContent  = "0";
  document.getElementById("avgConf").textContent     = "—";
  logBody.innerHTML = `<div class="log-entry info">Session reset.</div>`;
  spikeData.fill(0);
  simStep = 0;
  addLog("🔄 Stats reset.", "info");
  fetch(`${API_BASE}/api/reset`, { method: "POST" }).catch(() => {});
});

// ─────────────────────────────────────────────────────
window.dismissAlert = function() {
  alertDismissed = true;
  drowsyAlert.classList.add("hidden");
};

// ─────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────
checkAPI();

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const recordBtn = document.getElementById("record-btn");
const themeToggle = document.getElementById("theme-toggle");
const themeToggleText = document.getElementById("theme-toggle-text");
const connectionBadge = document.getElementById("connection-badge");
const connectionText = document.getElementById("connection-text");
const cursorDot = document.getElementById("cursor-dot");
const cursorRing = document.getElementById("cursor-ring");
const recordText = document.getElementById("record-text");
const recordIcon = document.getElementById("record-icon");
const recordingIndicator = document.getElementById("recording-indicator");
const recordingTimer = document.getElementById("recording-timer");
const selectedFile = document.getElementById("selected-file");
const statusMessage = document.getElementById("status-message");
const mainPanel = document.querySelector(".main-panel");
const analyzingPanel = document.getElementById("analyzing-panel");
const resultsPanel = document.getElementById("results-panel");
const primaryEmotion = document.getElementById("primary-emotion");
const primaryConfidenceBar = document.getElementById("primary-confidence-bar");
const primaryConfidenceText = document.getElementById("primary-confidence-text");
const secondaryEmotions = document.getElementById("secondary-emotions");
const resetBtn = document.getElementById("reset-btn");
const resultSummary = document.getElementById("result-summary");

let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let recordedChunks = [];
let recordingStartTime = 0;
let recordingTimerId = null;
let recordingAutoStopId = null;
let isRecording = false;

const THEME_STORAGE_KEY = "neuvocal-theme";
const AUTO_STOP_RECORDING_MS = 5000;
const API_BASE_STORAGE_KEY = "neuvocal-api-base";
let backendHealth = null;
let resolvedApiBase = null;

const emotionColors = {
  happy: "#f59e0b",
  sad: "#3b82f6",
  angry: "#ef4444",
  neutral: "#10b981",
  fear: "#8b5cf6",
  disgust: "#22c55e",
  surprise: "#f97316",
  calm: "#06b6d4",
};

function initializeCursorAnimation() {
  if (!cursorDot || !cursorRing) {
    return;
  }

  const isCoarsePointer = window.matchMedia("(pointer: coarse)").matches;
  if (isCoarsePointer) {
    return;
  }

  document.body.classList.add("cursor-enabled");

  let dotX = window.innerWidth / 2;
  let dotY = window.innerHeight / 2;
  let ringX = dotX;
  let ringY = dotY;
  let targetX = dotX;
  let targetY = dotY;

  const animate = () => {
    ringX += (targetX - ringX) * 0.18;
    ringY += (targetY - ringY) * 0.18;
    cursorDot.style.transform = `translate(${dotX - 4}px, ${dotY - 4}px)`;
    cursorRing.style.transform = `translate(${ringX - 17}px, ${ringY - 17}px)`;
    window.requestAnimationFrame(animate);
  };

  window.addEventListener("mousemove", (event) => {
    dotX = event.clientX;
    dotY = event.clientY;
    targetX = event.clientX;
    targetY = event.clientY;
    document.body.classList.add("cursor-visible");
  });

  window.addEventListener("mouseout", () => {
    document.body.classList.remove("cursor-visible");
  });

  const interactiveSelector = "button, a, input, select, textarea, .upload-section, .emotion-card";
  document.querySelectorAll(interactiveSelector).forEach((element) => {
    element.addEventListener("mouseenter", () => {
      document.body.classList.add("cursor-hover");
    });
    element.addEventListener("mouseleave", () => {
      document.body.classList.remove("cursor-hover");
    });
  });

  animate();
}

function getApiBaseUrl() {
  const savedApiBase = window.localStorage.getItem(API_BASE_STORAGE_KEY);
  if (savedApiBase) {
    return savedApiBase.replace(/\/$/, "");
  }

  const host = window.location.hostname;
  const port = window.location.port;
  const runningOnBackendPort = port === "3000";
  if (runningOnBackendPort) {
    return "";
  }
  if (host === "127.0.0.1" || host === "localhost") {
    return `${window.location.protocol}//${host}:3000`;
  }
  return "";
}

const apiBaseUrl = getApiBaseUrl();

function getApiUrl(pathname) {
  return `${(resolvedApiBase ?? apiBaseUrl)}${pathname}`;
}

function getApiCandidates() {
  const candidates = [];
  const current = (resolvedApiBase ?? apiBaseUrl).replace(/\/$/, "");
  const host = window.location.hostname;

  if (current || window.location.protocol.startsWith("http")) {
    candidates.push(current);
  }
  if (host !== "127.0.0.1") {
    candidates.push("http://127.0.0.1:3000");
  }
  if (host !== "localhost") {
    candidates.push("http://localhost:3000");
  }

  return [...new Set(candidates)];
}

function summarizeHealthIssue(payload) {
  if (!payload) {
    return "Backend health check returned no data.";
  }
  if (!payload.groqConfigured) {
    return "GROQ_API_KEY is missing. Add it in .env and restart the backend.";
  }
  return null;
}

function setConnectionStatus(connected, detail = "") {
  if (!connectionBadge || !connectionText) {
    return;
  }
  connectionBadge.classList.toggle("connected", connected);
  connectionBadge.classList.toggle("disconnected", !connected);
  connectionText.textContent = connected ? "Connected" : "Not Connected";
  if (detail) {
    connectionBadge.title = detail;
  }
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  themeToggleText.textContent = theme === "light" ? "Dark Theme" : "Light Theme";
}

function initializeTheme() {
  const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  const preferredTheme = savedTheme || "dark";
  applyTheme(preferredTheme);
}

function toggleTheme() {
  const nextTheme = document.body.dataset.theme === "light" ? "dark" : "light";
  applyTheme(nextTheme);
  window.localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
}

function showMessage(message, type = "info") {
  statusMessage.textContent = message;
  statusMessage.dataset.type = type;
  statusMessage.classList.remove("hidden");
}

function clearMessage() {
  statusMessage.textContent = "";
  statusMessage.dataset.type = "";
  statusMessage.classList.add("hidden");
}

function showSelectedFile(file) {
  selectedFile.textContent = `Selected: ${file.name}`;
  selectedFile.classList.remove("hidden");
}

function resetUploadState() {
  fileInput.value = "";
  selectedFile.textContent = "";
  selectedFile.classList.add("hidden");
  clearMessage();
}

function updateRecordingTimer() {
  const elapsedSeconds = Math.floor((Date.now() - recordingStartTime) / 1000);
  const minutes = String(Math.floor(elapsedSeconds / 60)).padStart(2, "0");
  const seconds = String(elapsedSeconds % 60).padStart(2, "0");
  recordingTimer.textContent = `${minutes}:${seconds}`;
}

function setRecordingUI(active) {
  isRecording = active;
  recordBtn.classList.toggle("recording", active);
  recordingIndicator.classList.toggle("hidden", !active);
  recordText.textContent = active ? "Stop Recording" : "Start Recording";
  recordIcon.setAttribute("data-lucide", active ? "square" : "mic");
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function setView(state) {
  mainPanel.classList.toggle("hidden", state !== "main");
  analyzingPanel.classList.toggle("hidden", state !== "analyzing");
  resultsPanel.classList.toggle("hidden", state !== "results");
}

function isAllowedAudio(file) {
  if (!file) {
    return false;
  }
  const name = file.name.toLowerCase();
  return name.endsWith(".mp3") || name.endsWith(".wav");
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderResults(payload) {
  const dominant = (payload.predicted_label || "unknown").toLowerCase();
  const dominantColor = emotionColors[dominant] || "#6366f1";
  const confidence = Number(payload.confidence || 0);
  const sortedEntries = Object.entries(payload.probabilities || {}).sort((a, b) => b[1] - a[1]);

  primaryEmotion.textContent = payload.predicted_label || "Unknown";
  primaryEmotion.style.color = dominantColor;
  primaryEmotion.style.textShadow = `0 0 30px ${dominantColor}66`;
  primaryConfidenceBar.style.width = formatPercent(confidence);
  primaryConfidenceBar.style.background = dominantColor;
  primaryConfidenceText.textContent = `${formatPercent(confidence)} confidence`;
  resultSummary.textContent = payload.transcript
    ? "Real prediction from audio + transcript"
    : "Real prediction from audio";

  secondaryEmotions.innerHTML = "";
  for (const [label, score] of sortedEntries.slice(1)) {
    const normalized = label.toLowerCase();
    const color = emotionColors[normalized] || "#94a3b8";
    const card = document.createElement("article");
    card.className = "emotion-card";
    card.style.animationDelay = `${secondaryEmotions.children.length * 90}ms`;
    card.innerHTML = `
      <div class="emotion-card-header">
        <span>${label}</span>
        <span>${formatPercent(score)}</span>
      </div>
      <div class="emotion-card-bar-container">
        <div class="emotion-card-bar" style="width:${formatPercent(score)}; background:${color};"></div>
      </div>
    `;
    secondaryEmotions.appendChild(card);
  }

  setView("results");
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

async function ensureBackendReady() {
  const candidates = getApiCandidates();
  let lastError = "Backend is not reachable.";

  for (const base of candidates) {
    const healthUrl = `${base}/api/health`;
    try {
      const response = await fetch(healthUrl);
      const payload = await response.json();

      if (!response.ok) {
        lastError = payload.error || `Backend health check failed at ${healthUrl}.`;
        continue;
      }

      const healthIssue = summarizeHealthIssue(payload);
      if (healthIssue) {
        throw new Error(healthIssue);
      }

      backendHealth = payload;
      resolvedApiBase = base;
      window.localStorage.setItem(API_BASE_STORAGE_KEY, base);
      const supabaseState = payload.supabaseConnected ? "Supabase connected" : "Supabase not connected";
      setConnectionStatus(true, `Backend: ${base || "same-origin"} | ${supabaseState}`);
      return;
    } catch (error) {
      lastError = error.message || `Cannot connect to ${healthUrl}`;
    }
  }

  setConnectionStatus(false, lastError);
  throw new Error(`${lastError} Start backend with npm.cmd run dev and open http://127.0.0.1:3000`);
}

async function detectEmotion(file) {
  const formData = new FormData();
  formData.append("audio", file);

  try {
    await ensureBackendReady();
    setView("analyzing");
    const response = await fetch(getApiUrl("/api/predict"), {
      method: "POST",
      body: formData,
    });
    const rawText = await response.text();
    let payload = null;

    if (rawText.trim()) {
      try {
        payload = JSON.parse(rawText);
      } catch (_error) {
        throw new Error(`Server returned an invalid response: ${rawText.slice(0, 160)}`);
      }
    }

    if (!response.ok) {
      throw new Error(
        (payload && (payload.details || payload.error)) ||
        `Prediction failed with status ${response.status}. Make sure the backend is running on port 3000.`
      );
    }

    if (!payload) {
      throw new Error("Server returned an empty response.");
    }

    if (payload.warning) {
      showMessage(payload.warning, "info");
    } else if (payload.dbWarning) {
      showMessage(`Prediction done, DB save failed: ${payload.dbWarning}`, "info");
    } else {
      clearMessage();
    }

    renderResults(payload);
  } catch (error) {
    setView("main");
    showMessage(error.message || "Prediction failed.", "error");
  }
}

function mergeBuffers(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function clampSample(sample) {
  return Math.max(-1, Math.min(1, sample));
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, value) {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = clampSample(samples[index]);
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showMessage("Microphone recording is not supported in this browser.", "error");
    return;
  }

  clearMessage();

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);
    recordedChunks = [];

    processorNode.onaudioprocess = (event) => {
      const channelData = event.inputBuffer.getChannelData(0);
      recordedChunks.push(new Float32Array(channelData));
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    recordingStartTime = Date.now();
    updateRecordingTimer();
    recordingTimerId = window.setInterval(updateRecordingTimer, 1000);
    recordingAutoStopId = window.setTimeout(() => {
      stopRecording();
    }, AUTO_STOP_RECORDING_MS);
    setRecordingUI(true);
    showMessage("Recording started. Emotion will be analyzed automatically after 5 seconds.", "info");
  } catch (error) {
    showMessage(`Microphone access failed: ${error.message}`, "error");
    await cleanupRecording();
  }
}

async function cleanupRecording() {
  if (recordingTimerId) {
    window.clearInterval(recordingTimerId);
    recordingTimerId = null;
  }

  if (recordingAutoStopId) {
    window.clearTimeout(recordingAutoStopId);
    recordingAutoStopId = null;
  }

  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }

  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }

  if (mediaStream) {
    for (const track of mediaStream.getTracks()) {
      track.stop();
    }
    mediaStream = null;
  }

  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }

  setRecordingUI(false);
}

async function stopRecording() {
  if (!audioContext || recordedChunks.length === 0) {
    await cleanupRecording();
    showMessage("No audio was recorded. Please try again.", "error");
    return;
  }

  const sampleRate = audioContext.sampleRate;
  const samples = mergeBuffers(recordedChunks);
  await cleanupRecording();

  if (samples.length === 0) {
    showMessage("No audio was captured. Please try again.", "error");
    return;
  }

  const wavBlob = encodeWav(samples, sampleRate);
  const wavFile = new File([wavBlob], `recording-${Date.now()}.wav`, { type: "audio/wav" });
  showSelectedFile(wavFile);
  showMessage("Recording complete. Detecting emotion directly...", "info");
  detectEmotion(wavFile);
}

function handleFileSelection(file) {
  if (!isAllowedAudio(file)) {
    resetUploadState();
    showMessage("Please choose only .mp3 or .wav audio files.", "error");
    return;
  }

  showSelectedFile(file);
  showMessage("Uploading audio for emotion detection...", "info");
  detectEmotion(file);
}

browseBtn.addEventListener("click", () => {
  fileInput.click();
});

dropZone.addEventListener("click", (event) => {
  if (event.target === browseBtn) {
    return;
  }
  fileInput.click();
});

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files || [];
  if (file) {
    handleFileSelection(file);
  }
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const [file] = event.dataTransfer.files || [];
  if (file) {
    handleFileSelection(file);
  }
});

recordBtn.addEventListener("click", async () => {
  if (isRecording) {
    await stopRecording();
    return;
  }
  await startRecording();
});

resetBtn.addEventListener("click", () => {
  resetUploadState();
  setView("main");
});

themeToggle.addEventListener("click", () => {
  toggleTheme();
});

initializeTheme();
initializeCursorAnimation();
ensureBackendReady().catch(() => {});

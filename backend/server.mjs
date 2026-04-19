import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import multer from "multer";
import { createClient } from "@supabase/supabase-js";
import { HfInference } from "@huggingface/inference";
import ffmpeg from "fluent-ffmpeg";
import ffmpegInstaller from "@ffmpeg-installer/ffmpeg";
ffmpeg.setFfmpegPath(ffmpegInstaller.path);

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const rootDir = path.resolve(__dirname, "..");
const uploadsDir = path.join(rootDir, "uploads");
const outputsDir = path.join(rootDir, "outputs");
const frontendFiles = ["index.html", "styles.css", "script.js"];
const port = Number(process.env.PORT || 3000);
const EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise", "calm", "disgust"];
const supabaseUrl = process.env.SUPABASE_URL || "";
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const supabaseTable = process.env.SUPABASE_TABLE || "voice_predictions";
const supabaseEnabled = Boolean(supabaseUrl && supabaseServiceKey);
const supabase = supabaseEnabled ? createClient(supabaseUrl, supabaseServiceKey) : null;
const HF_API_KEY = process.env.HF_API_KEY || "";
const HF_MODEL = "superb/wav2vec2-base-superb-er"; // Ultra-reliable model that never gets unallocated from the free-tier

fs.mkdirSync(uploadsDir, { recursive: true });
fs.mkdirSync(outputsDir, { recursive: true });

// CORS: in production set FRONTEND_URL to your Vercel domain (e.g. https://your-app.vercel.app)
// In dev it allows localhost:3001 as well as any origin without credentials.
const allowedOrigins = process.env.FRONTEND_URL
  ? [process.env.FRONTEND_URL, "http://localhost:3001"]
  : true; // true = allow all origins (open in dev)

app.use(cors({
  origin: allowedOrigins,
  credentials: true,
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use("/outputs", express.static(outputsDir));
app.use(express.static(rootDir, {
  extensions: ["html"],
  setHeaders: (res, filePath) => {
    if (frontendFiles.includes(path.basename(filePath))) {
      res.setHeader("Cache-Control", "no-store");
    }
  },
}));

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (_req, file, cb) => {
    const stamp = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
    cb(null, `${stamp}${path.extname(file.originalname || ".wav")}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024,
  },
});

const jobs = new Map();

function createJobRecord(config) {
  const jobId = `job_${Date.now()}_${Math.round(Math.random() * 1e6)}`;
  const job = {
    id: jobId,
    status: "queued",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    logs: [],
    config,
  };
  jobs.set(jobId, job);
  return job;
}

function appendJobLog(job, source, message) {
  job.logs.push({
    source,
    message,
    at: new Date().toISOString(),
  });
  job.updatedAt = new Date().toISOString();
  if (job.logs.length > 200) {
    job.logs = job.logs.slice(-200);
  }
}

function scoresFromLabel(predictedLabel, confidence = 0.74) {
  const remaining = Math.max(0.01, 1 - confidence);
  const perOther = remaining / Math.max(1, EMOTIONS.length - 1);
  return Object.fromEntries(
    EMOTIONS.map((emotion) => [
      emotion,
      emotion === predictedLabel ? confidence : perOther,
    ])
  );
}

async function transcribeWithGroq(audioPath) {
  const fileBuffer = fs.readFileSync(audioPath);
  const formData = new FormData();
  formData.append(
    "file",
    new Blob([fileBuffer], { type: "audio/wav" }),
    path.basename(audioPath)
  );
  formData.append("model", "whisper-large-v3-turbo");

  const response = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
    },
    body: formData,
  });

  const raw = await response.text();
  let payload = {};
  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch (_error) {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.error?.message || "Groq transcription failed.");
  }
  return String(payload.text || "");
}

async function classifyEmotionWithGroq(transcript) {
  if (!transcript.trim()) {
    return null;
  }

  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "llama-3.3-70b-versatile",
      temperature: 0.2,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `You are a multilingual emotion classifier. The user will provide a speech transcript in ANY language (English, Hindi, Telugu, Spanish, French, Arabic, Chinese, etc.). Regardless of the language, classify the dominant emotion into exactly one of: ${EMOTIONS.join(", ")}. Return a JSON object with keys "label" (string) and "confidence" (number between 0 and 1). Do not translate, just classify the emotion.`,
        },
        {
          role: "user",
          content: transcript,
        },
      ],
    }),
  });

  const raw = await response.text();
  let payload = {};
  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch (_error) {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.error?.message || "Groq emotion classification failed.");
  }

  const content = payload.choices?.[0]?.message?.content || "{}";
  const parsed = JSON.parse(content);
  const label = String(parsed.label || "neutral").toLowerCase();
  const confidence = Math.min(0.95, Math.max(0.35, Number(parsed.confidence || 0.72)));

  if (!EMOTIONS.includes(label)) {
    return null;
  }

  return { label, confidence };
}

// ── Convert any audio format → 16kHz mono WAV (required by wav2vec2) ──
function convertToWav(inputPath) {
  return new Promise((resolve, reject) => {
    const outputPath = inputPath.replace(/\.[^.]+$/, "_converted.wav");
    ffmpeg(inputPath)
      .audioChannels(1)
      .audioFrequency(16000)
      .audioCodec("pcm_s16le")
      .format("wav")
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error("FFmpeg conversion failed: " + err.message)))
      .save(outputPath);
  });
}

// ── Send raw audio bytes to HuggingFace wav2vec2 SER model ──────────────────
// This analyses acoustic features (pitch, energy, tone) — NOT text/words.
async function classifyEmotionFromAudio(audioPath) {
  if (!HF_API_KEY) {
    throw new Error("HF_API_KEY is missing. Add it to .env to use voice modulation detection.");
  }

  // Convert to proper 16kHz mono WAV first
  const wavPath = await convertToWav(audioPath);
  const audioBuffer = fs.readFileSync(wavPath);
  
  // Cleanup converted file after reading
  try { fs.unlinkSync(wavPath); } catch (_) {}

  // Instantiate the official HuggingFace Inference Client
  const hf = new HfInference(HF_API_KEY);

  let payload;
  let attempts = 0;
  let lastError = null;

  while (attempts < 3) {
    attempts++;
    try {
      // The official SDK reliably handles the `Loading` states and sockets
      payload = await hf.audioClassification({
        model: HF_MODEL,
        data: audioBuffer,
      });
      break; // Success!
    } catch (error) {
      lastError = error;
      console.log(`HF SDK Attempt ${attempts} failed: ${error.message}. Retrying in 2 seconds...`);
      await new Promise(r => setTimeout(r, 2000));
    }
  }

  if (!payload && lastError) {
    throw new Error(`HuggingFace API failed after 3 attempts. Raw error: ${lastError.message}`);
  }

  // Model still loading — HF returns {error: "Loading..."}
  if (payload.error) {
    throw new Error("HuggingFace model error: " + payload.error);
  }

  if (!Array.isArray(payload) || payload.length === 0) {
    throw new Error("Unexpected HuggingFace response format.");
  }

  // Normalise labels to lowercase and map to our EMOTIONS list
  const LABEL_MAP = {
    neu: "neutral",
    hap: "happy",
    ang: "angry",
    sad: "sad",
    anger: "angry",
    angry: "angry",
    happiness: "happy",
    happy: "happy",
    sadness: "sad",
  };

  // Build probabilities dict — use all returned scores
  const probabilities = {};
  for (const item of payload) {
    const raw_label = String(item.label || "").toLowerCase();
    const label = LABEL_MAP[raw_label] || raw_label;
    probabilities[label] = Number(item.score || 0);
  }

  // Fill missing EMOTIONS with 0
  for (const e of EMOTIONS) {
    if (!(e in probabilities)) probabilities[e] = 0;
  }

  // Find dominant emotion
  const predicted_label = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0];
  const confidence = probabilities[predicted_label];

  return { predicted_label, confidence, probabilities };
}

async function analyzeAudioInNode(audioPath) {
  const transcript = await transcribeWithGroq(audioPath);
  if (!transcript || !transcript.trim()) {
    throw new Error("Transcription returned empty text. Please speak clearly into the microphone.");
  }

  const classified = await classifyEmotionWithGroq(transcript);
  if (!classified) {
    throw new Error("Emotion classification did not return a valid label.");
  }

  return {
    audio: audioPath,
    predicted_label: classified.label,
    confidence: classified.confidence,
    probabilities: scoresFromLabel(classified.label, classified.confidence), // Generates synthetic array for UI 
    transcript,
    mode: "groq-nlp",
    summary: "Prediction from Groq Whisper transcription + Llama emotion classification.",
  };
}

function getHealthSnapshot() {
  return {
    ok: true,
    service: "voice-emotion-detection-backend",
    timestamp: new Date().toISOString(),
    runtime: "npm-only",
    groqConfigured: Boolean(process.env.GROQ_API_KEY),
    mode: "groq-nlp",
    supabaseConfigured: supabaseEnabled,
    supabaseTable,
  };
}

async function getSupabaseConnectionStatus() {
  if (!supabaseEnabled || !supabase) {
    return {
      configured: false,
      connected: false,
      details: "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing.",
    };
  }
  try {
    const { error } = await supabase.from(supabaseTable).select("*", { count: "exact", head: true });
    if (error) {
      return {
        configured: true,
        connected: false,
        details: error.message,
      };
    }
    return {
      configured: true,
      connected: true,
      details: "Supabase connection OK.",
    };
  } catch (error) {
    return {
      configured: true,
      connected: false,
      details: error.message,
    };
  }
}

async function savePredictionToSupabase(payload) {
  if (!supabaseEnabled || !supabase) {
    return {
      ok: false,
      skipped: true,
      message: "Supabase is not configured.",
    };
  }
  const record = {
    audio: payload.audio || "",
    predicted_label: payload.predicted_label || "",
    confidence: payload.confidence || 0,
    probabilities: payload.probabilities || {},
    transcript: payload.transcript || "",
    mode: payload.mode || "openai",
    summary: payload.summary || "",
  };
  const { error } = await supabase.from(supabaseTable).insert([record]);
  if (error) {
    return {
      ok: false,
      skipped: false,
      message: error.message,
    };
  }
  return {
    ok: true,
    skipped: false,
    message: "Saved to Supabase.",
  };
}

app.get("/api/health", async (_req, res) => {
  const base = getHealthSnapshot();
  const supabaseStatus = await getSupabaseConnectionStatus();
  res.json({
    ...base,
    supabaseConnected: supabaseStatus.connected,
    supabaseDetails: supabaseStatus.details,
  });
});

app.get("/api/jobs", (_req, res) => {
  res.json({
    jobs: Array.from(jobs.values()).sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
  });
});

app.get("/api/jobs/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    res.status(404).json({ error: "Job not found." });
    return;
  }
  res.json(job);
});

app.get("/api/predictions", async (_req, res) => {
  if (!supabaseEnabled || !supabase) {
    res.status(400).json({
      error: "Supabase is not configured.",
      details: "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env.",
    });
    return;
  }
  try {
    const { data, error } = await supabase
      .from(supabaseTable)
      .select("*")
      .order("created_at", { ascending: false })
      .limit(20);
    if (error) {
      res.status(500).json({
        error: "Failed to read predictions from Supabase.",
        details: error.message,
      });
      return;
    }
    res.json({
      predictions: data || [],
    });
  } catch (error) {
    res.status(500).json({
      error: "Supabase request failed.",
      details: error.message,
    });
  }
});

app.post("/api/train", async (_req, res) => {
  const job = createJobRecord({ mode: "npm-only" });
  job.status = "failed";
  appendJobLog(job, "info", "Training is disabled in npm-only mode.");
  res.status(501).json({
    error: "Training is not available in npm-only mode.",
    details: "This backend no longer uses Python. Use an external API-based pipeline or add a JavaScript training stack.",
  });
});

app.post("/api/prepare-dataset", async (_req, res) => {
  res.status(501).json({
    error: "Dataset preparation is not available in npm-only mode.",
    details: "This route previously depended on Python preprocessing.",
  });
});

app.post("/api/predict", upload.single("audio"), async (req, res) => {
  const audioPath = req.file ? req.file.path : req.body.audioPath;

  if (!audioPath) {
    res.status(400).json({ error: "Provide an uploaded `audio` file or `audioPath`." });
    return;
  }
  if (!process.env.HF_API_KEY) {
    res.status(400).json({
      error: "HF_API_KEY is missing.",
      details: "Add HF_API_KEY to your .env file. Get a free token at https://huggingface.co/settings/tokens",
    });
    return;
  }

  try {
    const payload = await analyzeAudioInNode(audioPath);
    const dbResult = await savePredictionToSupabase(payload);
    res.json({
      ...payload,
      checkpoint: null,
      supabaseSaved: dbResult.ok,
      dbWarning: dbResult.ok || dbResult.skipped ? null : dbResult.message,
    });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(500).json({
      error: "Prediction failed.",
      details: error.message,
    });
  }
});

app.use("/api", (_req, res) => {
  res.status(404).json({
    error: "API route not found.",
  });
});

app.use((error, _req, res, _next) => {
  if (error instanceof multer.MulterError) {
    res.status(400).json({
      error: "File upload failed.",
      details: error.message,
    });
    return;
  }

  res.status(500).json({
    error: "Unexpected server error.",
    details: error.message,
  });
});

app.get("/", (_req, res) => {
  res.sendFile(path.join(rootDir, "index.html"));
});

app.listen(port, "0.0.0.0", () => {
  console.log(`Voice emotion backend listening on http://0.0.0.0:${port}`);
});

"use client"

import { useState, useRef, useEffect } from "react"
import { WaveAnimation } from "@/components/ui/wave-animation-1"
import { Mic, Square, Loader2, Upload, Activity, Clock, ChevronDown } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface PredictionResult {
  predicted_label: string
  confidence: number
  probabilities: Record<string, number>
  transcript?: string
  timestamp: number
}

const EMOTION_COLORS: Record<string, string> = {
  happy:    "from-yellow-400 to-orange-400",
  sad:      "from-blue-400 to-indigo-500",
  angry:    "from-red-500 to-rose-600",
  fear:     "from-purple-500 to-violet-600",
  disgust:  "from-green-600 to-emerald-500",
  surprise: "from-pink-400 to-fuchsia-500",
  neutral:  "from-zinc-400 to-slate-500",
  calm:     "from-teal-400 to-cyan-500",
}

const emotionColor = (label: string) =>
  EMOTION_COLORS[label?.toLowerCase()] ?? "from-indigo-400 to-purple-500"

export default function Home() {
  const [isRecording, setIsRecording] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [history, setHistory] = useState<PredictionResult[]>([])
  const [error, setError] = useState<string | null>(null)
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null)
  const [recordingTime, setRecordingTime] = useState(0)
  const [historyOpen, setHistoryOpen] = useState(false)

  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (recordingTime >= 5 && isRecording) stopRecording()
  }, [recordingTime, isRecording])

  // ── Emoji cursor based on detected emotion ──
  const EMOTION_EMOJIS: Record<string, string> = {
    happy:    "😄",
    sad:      "😢",
    angry:    "😠",
    fear:     "😨",
    disgust:  "🤢",
    surprise: "😲",
    neutral:  "😐",
    calm:     "😌",
  }

  useEffect(() => {
    const emoji = result ? (EMOTION_EMOJIS[result.predicted_label?.toLowerCase()] ?? "🎙️") : null

    if (!emoji) {
      document.body.style.cursor = "default"
      return
    }

    // Draw emoji onto a canvas and use it as a CSS cursor data-URL
    const size = 40
    const canvas = document.createElement("canvas")
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.font = `${size - 4}px serif`
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.clearRect(0, 0, size, size)
      ctx.fillText(emoji, size / 2, size / 2)
    }
    const dataURL = canvas.toDataURL()
    // hotspot = centre of emoji
    document.body.style.cursor = `url("${dataURL}") ${size / 2} ${size / 2}, auto`

    return () => {
      document.body.style.cursor = "default"
    }
  }, [result])

  const startRecording = async () => {
    try {
      setError(null)
      setResult(null)
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaStreamRef.current = stream

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
      audioContextRef.current = audioCtx
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      setAnalyserNode(analyser)

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }
      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" })
        const file = new File([blob], `rec-${Date.now()}.wav`, { type: "audio/wav" })
        await analyzeAudio(file)
      }

      mediaRecorder.start(200)
      setIsRecording(true)
      setRecordingTime(0)
      timerRef.current = setInterval(() => setRecordingTime((p) => p + 1), 1000)
    } catch (err: any) {
      setError("Microphone access denied: " + err.message)
    }
  }

  const stopRecording = () => {
    if (!mediaRecorderRef.current || !isRecording) return
    mediaRecorderRef.current.stop()
    setIsRecording(false)
    if (timerRef.current) clearInterval(timerRef.current)
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    setTimeout(() => setAnalyserNode(null), 800)
  }

  const analyzeAudio = async (file: File) => {
    setIsAnalyzing(true)
    setError(null)
    const formData = new FormData()
    formData.append("audio", file)
    try {
      // Use explicit backend URL if set, otherwise fall back to the
      // Next.js rewrite proxy (/api/* → Railway) defined in next.config.ts
      const apiBase = process.env.NEXT_PUBLIC_BACKEND_URL ?? ""
      const res = await fetch(`${apiBase}/api/predict`, {
        method: "POST",
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.details || data.error || "Prediction failed")
      const entry: PredictionResult = { ...data, timestamp: Date.now() }
      setResult(entry)
      setHistory((prev) => [entry, ...prev].slice(0, 20))
    } catch (err: any) {
      setError(err.message || "Failed to analyze audio")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) analyzeAudio(file)
  }

  // Sort probabilities descending
  const sortedProbs = result?.probabilities
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <main className="relative flex min-h-screen w-full flex-col items-center justify-start overflow-hidden bg-black text-white px-4 py-14"
      style={{ fontFamily: "var(--font-sans)" }}>

      {/* ── Full-page Wave ── */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <WaveAnimation
          waveSpeed={3}
          baseWaveIntensity={50}
          particleColor="#fff200"
          pointSize={2}
          gridDistance={2}
          analyserNode={analyserNode}
          className="w-full h-full"
        />
      </div>
      {/* Darkening gradient */}
      <div className="absolute inset-0 z-[1] pointer-events-none bg-gradient-to-b from-black/75 via-black/30 to-black/80" />

      {/* ── Header ── */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative z-10 text-center mb-10"
      >
        <div className="inline-flex items-center gap-2 px-4 py-1.5 mb-3 rounded-full border border-yellow-400/30 bg-yellow-400/10 text-yellow-300 text-xs font-semibold tracking-widest uppercase backdrop-blur-md">
          <Activity className="w-3 h-3" />
          NeuVocal AI &nbsp;·&nbsp; Llama 3.3 on Groq
        </div>
        <h1 className="text-6xl md:text-7xl font-bold tracking-tight text-white leading-none mb-3">
          Voice{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-yellow-300 to-orange-400">
            Emotion
          </span>
        </h1>
        <p className="text-white/50 text-base max-w-md mx-auto">
          Speak — the wave reacts to your voice, and AI reads your emotion.
        </p>
      </motion.header>

      <div className="relative z-10 w-full max-w-xl flex flex-col gap-4">

        {/* ── Main Card ── */}
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="w-full rounded-2xl border border-white/10 bg-black/50 backdrop-blur-2xl p-8 flex flex-col items-center gap-6 shadow-2xl"
        >
          <AnimatePresence mode="wait">

            {/* ── Idle / Recording ── */}
            {!isAnalyzing && !result && (
              <motion.div key="idle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex flex-col items-center gap-6 w-full"
              >
                {/* Mic Button */}
                <motion.button
                  whileHover={{ scale: 1.06 }}
                  whileTap={{ scale: 0.94 }}
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`relative flex items-center justify-center w-24 h-24 rounded-full border-2 backdrop-blur-xl transition-all duration-300 shadow-xl
                    ${isRecording
                      ? "bg-yellow-400/20 border-yellow-400/70 text-yellow-300 shadow-yellow-500/20"
                      : "bg-white/10 border-white/20 text-white shadow-black/40 hover:bg-white/15"
                    }`}
                >
                  {isRecording && (
                    <div className="absolute inset-0 rounded-full border-2 border-yellow-400/50 animate-ping" />
                  )}
                  {isRecording
                    ? <Square className="w-9 h-9 relative z-10" />
                    : <Mic className="w-9 h-9 relative z-10" />
                  }
                </motion.button>

                {/* Timer / Hint */}
                {isRecording ? (
                  <div className="flex flex-col items-center gap-1">
                    <span className="font-mono text-3xl font-bold text-yellow-300 tabular-nums">
                      00:0{recordingTime}
                    </span>
                    <span className="text-white/40 text-xs">auto-stops at 5s</span>
                  </div>
                ) : (
                  <p className="text-white/40 text-sm">Click the mic to record · or upload a file</p>
                )}

                {/* Upload */}
                {!isRecording && (
                  <label className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 cursor-pointer transition-colors text-sm font-medium">
                    <Upload className="w-4 h-4 text-white/60" />
                    Upload Audio
                    <input type="file" accept=".wav,.mp3,.webm" className="hidden" onChange={handleFileUpload} />
                  </label>
                )}
              </motion.div>
            )}

            {/* ── Analyzing ── */}
            {isAnalyzing && (
              <motion.div key="analyzing"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex flex-col items-center gap-4 py-8"
              >
                <Loader2 className="w-12 h-12 text-yellow-400 animate-spin" />
                <p className="text-yellow-300 font-semibold animate-pulse text-lg">Analyzing emotion…</p>
              </motion.div>
            )}

            {/* ── Result ── */}
            {result && !isAnalyzing && (
              <motion.div key="result"
                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.4 }}
                className="flex flex-col items-center gap-5 w-full"
              >
                {/* Dominant label */}
                <div className="text-center">
                  <p className="text-white/40 text-xs font-semibold uppercase tracking-widest mb-1">Dominant Emotion</p>
                  <h2
                    className={`text-5xl font-bold capitalize text-transparent bg-clip-text bg-gradient-to-r ${emotionColor(result.predicted_label)}`}
                    style={{ textShadow: "none" }}
                  >
                    {result.predicted_label}
                  </h2>
                </div>

                {/* All emotion bars */}
                <div className="w-full flex flex-col gap-2.5">
                  {sortedProbs.map(([label, score]) => {
                    const pct = Math.round(score * 100)
                    return (
                      <div key={label} className="flex items-center gap-3">
                        <span className="w-16 text-right text-xs text-white/50 capitalize font-medium">{label}</span>
                        <div className="flex-1 h-2 bg-white/8 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${pct}%` }}
                            transition={{ duration: 0.8, ease: "easeOut", delay: 0.1 }}
                            className={`h-full rounded-full bg-gradient-to-r ${emotionColor(label)}`}
                          />
                        </div>
                        <span className="w-10 font-mono text-xs text-white/60 tabular-nums">{pct}%</span>
                      </div>
                    )
                  })}
                </div>

                {/* Transcript */}
                {result.transcript && (
                  <div className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/8">
                    <p className="text-white/35 text-[10px] uppercase tracking-widest font-semibold mb-1">Transcript</p>
                    <p className="text-white/70 text-sm italic leading-relaxed">"{result.transcript}"</p>
                  </div>
                )}

                <motion.button
                  whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
                  onClick={() => setResult(null)}
                  className="px-7 py-2.5 rounded-full bg-white text-black font-semibold text-sm hover:bg-yellow-100 transition-colors"
                >
                  Record Again
                </motion.button>
              </motion.div>
            )}

          </AnimatePresence>
        </motion.div>

        {/* ── Error ── */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="px-5 py-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm text-center"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── History panel ── */}
        <AnimatePresence>
          {history.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              className="w-full rounded-2xl border border-white/10 bg-black/50 backdrop-blur-2xl overflow-hidden shadow-xl"
            >
              {/* Header toggle */}
              <button
                onClick={() => setHistoryOpen((o) => !o)}
                className="w-full flex items-center justify-between px-6 py-4 text-sm text-white/60 hover:text-white/90 transition-colors"
              >
                <div className="flex items-center gap-2 font-semibold">
                  <Clock className="w-4 h-4 text-yellow-400" />
                  History
                  <span className="ml-1 px-2 py-0.5 rounded-full bg-yellow-400/20 text-yellow-300 text-xs font-bold">
                    {history.length}
                  </span>
                </div>
                <motion.div animate={{ rotate: historyOpen ? 180 : 0 }} transition={{ duration: 0.2 }}>
                  <ChevronDown className="w-4 h-4" />
                </motion.div>
              </button>

              <AnimatePresence>
                {historyOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 pb-4 flex flex-col gap-2 max-h-72 overflow-y-auto">
                      {history.map((entry, i) => (
                        <motion.div
                          key={entry.timestamp}
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.04 }}
                          className="flex items-center justify-between px-4 py-3 rounded-xl bg-white/4 border border-white/6 hover:bg-white/8 transition-colors"
                        >
                          <div className="flex items-center gap-3">
                            <span className={`w-2 h-2 rounded-full bg-gradient-to-r ${emotionColor(entry.predicted_label)} flex-shrink-0`} />
                            <span className="capitalize font-semibold text-sm text-white/80">{entry.predicted_label}</span>
                            {entry.transcript && (
                              <span className="text-white/35 text-xs truncate max-w-[140px] italic">
                                "{entry.transcript}"
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-3 flex-shrink-0">
                            <span className={`font-mono text-sm font-bold text-transparent bg-clip-text bg-gradient-to-r ${emotionColor(entry.predicted_label)}`}>
                              {Math.round(entry.confidence * 100)}%
                            </span>
                            <span className="text-white/25 text-xs">
                              {new Date(entry.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                            </span>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </main>
  )
}

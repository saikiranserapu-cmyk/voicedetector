from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import os

import httpx


EMOTION_HINTS = {
    "angry": ["angry", "furious", "annoyed", "mad", "frustrated"],
    "happy": ["happy", "joy", "excited", "delighted", "glad"],
    "sad": ["sad", "upset", "hurt", "cry", "depressed"],
    "fear": ["afraid", "fear", "scared", "nervous", "anxious"],
    "neutral": ["okay", "fine", "normal", "neutral", "alright"],
}


def lexical_emotion_scores(text: str) -> list[float]:
    lowered = text.lower()
    scores: list[float] = []
    for keywords in EMOTION_HINTS.values():
        score = sum(lowered.count(keyword) for keyword in keywords)
        scores.append(float(score))
    total = sum(scores)
    if total == 0:
        return [0.0 for _ in scores]
    return [score / total for score in scores]


class TranscriptProvider(Protocol):
    def transcribe(self, audio_path: str) -> str: ...


@dataclass
class NoOpTranscriptProvider:
    def transcribe(self, audio_path: str) -> str:
        return ""


@dataclass
class OpenAITranscriptProvider:
    model: str = "gpt-4o-mini-transcribe"

    def transcribe(self, audio_path: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ""
        with open(audio_path, "rb") as audio_file, httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(audio_path), audio_file, "audio/wav")},
                data={"model": self.model},
            )
            response.raise_for_status()
            payload = response.json()
        return str(payload.get("text", ""))


@dataclass
class DeepgramTranscriptProvider:
    model: str = "nova-2"

    def transcribe(self, audio_path: str) -> str:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            return ""
        with open(audio_path, "rb") as audio_file, httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "audio/wav",
                },
                params={"model": self.model, "smart_format": "true"},
                content=audio_file.read(),
            )
            response.raise_for_status()
            payload = response.json()
        channels = payload.get("results", {}).get("channels", [])
        if not channels:
            return ""
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return ""
        return str(alternatives[0].get("transcript", ""))


def create_transcript_provider(name: str) -> TranscriptProvider:
    normalized = name.strip().lower()
    if normalized == "openai":
        return OpenAITranscriptProvider()
    if normalized == "deepgram":
        return DeepgramTranscriptProvider()
    return NoOpTranscriptProvider()

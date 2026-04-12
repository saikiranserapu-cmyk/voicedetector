from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor


REQUIRED_COLUMNS = {"path", "label", "split"}


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def build_label_maps(df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(df["label"].astype(str).unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def load_audio(path: str | Path, sample_rate: int, max_audio_seconds: float) -> torch.Tensor:
    waveform, original_rate = torchaudio.load(str(path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)
    waveform = waveform.squeeze(0)
    max_frames = int(sample_rate * max_audio_seconds)
    if waveform.numel() > max_frames:
        waveform = waveform[:max_frames]
    return waveform


class EmotionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: AutoFeatureExtractor,
        label2id: dict[str, int],
        sample_rate: int,
        max_audio_seconds: float,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        waveform = load_audio(row["path"], self.sample_rate, self.max_audio_seconds)
        item: dict[str, Any] = {
            "audio": waveform.numpy(),
            "label": self.label2id[str(row["label"])],
            "path": str(row["path"]),
            "text": str(row["text"]) if "text" in row and pd.notna(row["text"]) else "",
        }
        return item


@dataclass
class BatchCollator:
    feature_extractor: AutoFeatureExtractor
    sample_rate: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        audio = [item["audio"] for item in batch]
        encoded = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded["paths"] = [item["path"] for item in batch]
        encoded["texts"] = [item["text"] for item in batch]
        return encoded

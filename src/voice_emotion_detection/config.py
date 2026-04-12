from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class TrainConfig:
    csv: str
    output_dir: str
    model_name: str = "microsoft/wavlm-base-plus"
    sample_rate: int = 16000
    batch_size: int = 8
    epochs: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_audio_seconds: float = 8.0
    num_workers: int = 0
    seed: int = 42
    transcript_provider: str = "none"
    freeze_backbone: bool = False

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir) / "train_config.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .features import lexical_emotion_scores


@dataclass
class ModelArtifacts:
    label2id: dict[str, int]
    id2label: dict[int, str]
    model_name: str
    sample_rate: int
    max_audio_seconds: float


class EmotionClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        freeze_backbone: bool = False,
        use_text_branch: bool = True,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.config.hidden_size)
        self.use_text_branch = use_text_branch
        text_dim = 5 if use_text_branch else 0
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size + text_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_size + text_dim, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_labels),
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        texts: list[str] | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        if self.use_text_branch:
            texts = texts or ["" for _ in range(pooled.size(0))]
            text_features = torch.tensor(
                [lexical_emotion_scores(text) for text in texts],
                dtype=pooled.dtype,
                device=pooled.device,
            )
            pooled = torch.cat([pooled, text_features], dim=1)

        return self.classifier(pooled)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor

from .dataset import load_audio
from .features import create_transcript_provider
from .model import EmotionClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict emotion from an audio file.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--audio", required=True, help="Path to an audio file")
    parser.add_argument("--transcript-provider", default="none", help="none, openai, or deepgram")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    label2id = checkpoint["label2id"]
    id2label = {int(k): v for k, v in checkpoint["id2label"].items()}

    feature_extractor = AutoFeatureExtractor.from_pretrained(config["model_name"])
    model = EmotionClassifier(
        model_name=config["model_name"],
        num_labels=len(label2id),
        freeze_backbone=False,
        use_text_branch=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    waveform = load_audio(args.audio, config["sample_rate"], config["max_audio_seconds"])
    encoded = feature_extractor(
        [waveform.numpy()],
        sampling_rate=config["sample_rate"],
        return_tensors="pt",
        padding=True,
    )

    transcript_provider = create_transcript_provider(args.transcript_provider)
    transcript = transcript_provider.transcribe(args.audio) if args.transcript_provider != "none" else ""

    with torch.no_grad():
        logits = model(
            input_values=encoded["input_values"],
            attention_mask=encoded.get("attention_mask"),
            texts=[transcript],
        )
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())

    result = {
        "audio": str(args.audio),
        "predicted_label": id2label[predicted_id],
        "confidence": float(probabilities[predicted_id].item()),
        "probabilities": {
            id2label[idx]: float(probabilities[idx].item()) for idx in range(probabilities.shape[0])
        },
        "transcript": transcript,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

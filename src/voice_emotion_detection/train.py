from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup

from .config import TrainConfig
from .dataset import BatchCollator, EmotionDataset, build_label_maps, load_metadata
from .model import EmotionClassifier


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a voice emotion detection model.")
    parser.add_argument("--csv", required=True, help="Metadata CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints.")
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transcript-provider", default="none")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: list[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (counts * num_labels)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    model: EmotionClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    id2label: dict[int, str],
) -> dict[str, float | str]:
    model.eval()
    losses: list[float] = []
    preds: list[int] = []
    refs: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)
            logits = model(
                input_values=input_values,
                attention_mask=attention_mask,
                texts=batch.get("texts"),
            )
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            refs.extend(labels.cpu().tolist())

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(refs, preds)) if refs else 0.0,
        "macro_f1": float(f1_score(refs, preds, average="macro")) if refs else 0.0,
        "report": classification_report(
            refs,
            preds,
            labels=sorted(id2label),
            target_names=[id2label[idx] for idx in sorted(id2label)],
            zero_division=0,
        )
        if refs
        else "",
    }
    return metrics


def save_checkpoint(
    output_dir: Path,
    model: EmotionClassifier,
    optimizer: AdamW,
    scheduler,
    config: TrainConfig,
    label2id: dict[str, int],
    id2label: dict[int, str],
    metrics: dict[str, float | str],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": vars(config),
        "label2id": label2id,
        "id2label": id2label,
        "metrics": metrics,
    }
    torch.save(checkpoint, output_dir / "best.pt")


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)
    df = load_metadata(config.csv)
    label2id, id2label = build_label_maps(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("CSV must contain non-empty train and val splits.")

    train_dataset = EmotionDataset(
        train_df,
        feature_extractor=feature_extractor,
        label2id=label2id,
        sample_rate=config.sample_rate,
        max_audio_seconds=config.max_audio_seconds,
    )
    val_dataset = EmotionDataset(
        val_df,
        feature_extractor=feature_extractor,
        label2id=label2id,
        sample_rate=config.sample_rate,
        max_audio_seconds=config.max_audio_seconds,
    )
    collator = BatchCollator(feature_extractor=feature_extractor, sample_rate=config.sample_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    model = EmotionClassifier(
        model_name=config.model_name,
        num_labels=len(label2id),
        freeze_backbone=config.freeze_backbone,
        use_text_branch=True,
    ).to(device)

    class_weights = compute_class_weights(train_df["label"].map(label2id).tolist(), len(label2id)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0
    history: list[dict[str, float | str | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in progress:
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_values=input_values,
                attention_mask=attention_mask,
                texts=batch.get("texts"),
            )
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, loss_fn, device, id2label)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_summary)

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            save_checkpoint(output_dir, model, optimizer, scheduler, config, label2id, id2label, val_metrics)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if not test_df.empty:
        test_dataset = EmotionDataset(
            test_df,
            feature_extractor=feature_extractor,
            label2id=label2id,
            sample_rate=config.sample_rate,
            max_audio_seconds=config.max_audio_seconds,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collator,
        )
        best_state = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(best_state["model_state_dict"])
        test_metrics = evaluate(model, test_loader, loss_fn, device, id2label)
        (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd


RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

CREMA_D_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SER dataset metadata CSV.")
    parser.add_argument("--dataset", required=True, choices=["ravdess", "crema_d"])
    parser.add_argument("--input-dir", required=True, help="Path to raw dataset directory.")
    parser.add_argument("--output-csv", required=True, help="Path to save metadata CSV.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def split_by_speaker(
    speaker_ids: list[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")

    unique_speakers = sorted(set(speaker_ids))
    random.Random(seed).shuffle(unique_speakers)

    total = len(unique_speakers)
    val_count = max(1, round(total * val_ratio)) if total >= 3 else 1 if total >= 2 else 0
    test_count = max(1, round(total * test_ratio)) if total >= 4 else 1 if total >= 3 else 0
    train_count = max(1, total - val_count - test_count)

    if train_count + val_count + test_count > total:
        overflow = train_count + val_count + test_count - total
        test_count = max(0, test_count - overflow)

    train_speakers = set(unique_speakers[:train_count])
    val_speakers = set(unique_speakers[train_count:train_count + val_count])
    test_speakers = set(unique_speakers[train_count + val_count:])

    split_map: dict[str, str] = {}
    for speaker_id in unique_speakers:
        if speaker_id in train_speakers:
            split_map[speaker_id] = "train"
        elif speaker_id in val_speakers:
            split_map[speaker_id] = "val"
        else:
            split_map[speaker_id] = "test"
    return split_map


def build_ravdess_rows(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for audio_path in input_dir.rglob("*.wav"):
        parts = audio_path.stem.split("-")
        if len(parts) != 7:
            continue
        emotion_code = parts[2]
        speaker_id = parts[6]
        label = RAVDESS_EMOTIONS.get(emotion_code)
        if not label:
            continue
        rows.append(
            {
                "path": str(audio_path.resolve()),
                "label": label,
                "speaker_id": speaker_id,
                "text": "",
            }
        )
    return rows


def build_crema_d_rows(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for audio_path in input_dir.rglob("*.wav"):
        parts = audio_path.stem.split("_")
        if len(parts) < 3:
            continue
        speaker_id = parts[0]
        emotion_code = parts[2]
        label = CREMA_D_EMOTIONS.get(emotion_code)
        if not label:
            continue
        rows.append(
            {
                "path": str(audio_path.resolve()),
                "label": label,
                "speaker_id": speaker_id,
                "text": "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if args.dataset == "ravdess":
        rows = build_ravdess_rows(input_dir)
    else:
        rows = build_crema_d_rows(input_dir)

    if not rows:
        raise ValueError(f"No valid audio files found for dataset {args.dataset} in {input_dir}")

    df = pd.DataFrame(rows)
    split_map = split_by_speaker(
        speaker_ids=df["speaker_id"].astype(str).tolist(),
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    df["split"] = df["speaker_id"].astype(str).map(split_map)
    df = df[["path", "label", "split", "speaker_id", "text"]].sort_values(
        ["split", "speaker_id", "path"]
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    summary = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"Saved metadata to: {output_csv}")
    print(summary.to_string())


if __name__ == "__main__":
    main()

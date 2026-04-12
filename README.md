# Voice Emotion Detection

This project provides a practical speech emotion recognition (SER) pipeline built around a pretrained audio foundation model, with optional transcript enrichment from services such as Whisper, OpenAI, or Deepgram.

## Important accuracy note

No honest implementation can guarantee `>97%` accuracy without a specific dataset, label schema, and evaluation protocol. In SER, very high scores are often caused by:

- speaker leakage between train and test
- binary or overly simplified label sets
- testing on the same corpus characteristics seen in training

This repo is designed to maximize your chances of strong performance on common datasets such as RAVDESS, CREMA-D, EMO-DB, SAVEE, or IEMOCAP, while keeping evaluation speaker-aware and reproducible.

## Approach

- Audio backbone: pretrained transformer encoder such as `facebook/wav2vec2-base` or `microsoft/wavlm-base-plus`
- Classifier head: pooled audio embeddings -> dropout -> MLP logits
- Optional transcript branch:
  - use local Whisper or API transcripts
  - derive text-side emotion hints and concatenate with audio embeddings
- Training:
  - class-weighted loss
  - optional mixup-friendly augmentation hooks
  - macro F1 and accuracy tracking

## Dataset format

Prepare a CSV file with these columns:

```csv
path,label,split,speaker_id,text
data/audio/001.wav,happy,train,spk01,
data/audio/002.wav,sad,train,spk01,
data/audio/101.wav,angry,val,spk12,
data/audio/205.wav,neutral,test,spk22,
```

Required columns:

- `path`: path to audio file
- `label`: emotion class name
- `split`: `train`, `val`, or `test`

Optional columns:

- `speaker_id`: strongly recommended for speaker-aware dataset construction
- `text`: transcript if already available

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm.cmd install
```

If PowerShell blocks `npm`, use `npm.cmd` on Windows.

## Train

```powershell
python -m src.voice_emotion_detection.train `
  --csv data/metadata.csv `
  --output-dir outputs/run1 `
  --model-name microsoft/wavlm-base-plus `
  --batch-size 8 `
  --epochs 8
```

## Prepare A Real Dataset

The model only becomes genuinely useful after you train it on a real speech-emotion dataset. Two good starting points are `RAVDESS` and `CREMA-D`.

### RAVDESS

If your raw files are under `data/RAVDESS`, generate metadata like this:

```powershell
python -m src.voice_emotion_detection.prepare_dataset `
  --dataset ravdess `
  --input-dir data/RAVDESS `
  --output-csv data/ravdess_metadata.csv
```

### CREMA-D

If your raw files are under `data/CREMA-D`, generate metadata like this:

```powershell
python -m src.voice_emotion_detection.prepare_dataset `
  --dataset crema_d `
  --input-dir data/CREMA-D `
  --output-csv data/crema_d_metadata.csv
```

The prep script creates speaker-aware `train`, `val`, and `test` splits automatically so evaluation is more realistic.

### Then Train A Real Checkpoint

```powershell
python -m src.voice_emotion_detection.train `
  --csv data/ravdess_metadata.csv `
  --output-dir outputs/run1 `
  --model-name microsoft/wavlm-base-plus `
  --batch-size 8 `
  --epochs 8
```

After training, the real model checkpoint should exist at:

```text
outputs/run1/best.pt
```

Once that file exists, the backend will use the trained model instead of demo fallback.

## Predict

```powershell
python -m src.voice_emotion_detection.predict `
  --checkpoint outputs/run1/best.pt `
  --audio path\to\sample.wav
```

## NPM backend

Start the backend:

```powershell
npm.cmd install
npm.cmd run dev
```

The API exposes:

- `GET /api/health`
- `GET /api/jobs`
- `GET /api/jobs/:jobId`
- `POST /api/train`
- `POST /api/predict`

### Train via API

```powershell
curl -X POST http://localhost:3000/api/train ^
  -H "Content-Type: application/json" ^
  -d "{\"csv\":\"data/metadata.csv\",\"outputDir\":\"outputs/run1\",\"epochs\":8}"
```

### Predict via API

With multipart upload:

```powershell
curl -X POST http://localhost:3000/api/predict ^
  -F "audio=@sample.wav" ^
  -F "checkpoint=outputs/run1/best.pt"
```

With a local path already on disk:

```powershell
curl -X POST http://localhost:3000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"audioPath\":\"uploads/sample.wav\",\"checkpoint\":\"outputs/run1/best.pt\"}"
```

## Foundation model options

This project keeps transcript generation optional and pluggable:

- `none`: audio-only classifier
- `openai`: use OpenAI transcription API when an API key is available
- `deepgram`: use Deepgram transcription API when an API key is available
- `local`: reserve a local integration point for Whisper or another ASR tool

The training pipeline does not require transcripts. Transcript enrichment is most helpful during inference or when your dataset already includes text.

## Practical path toward very high accuracy

If you want the best shot at approaching the requested target:

1. Use a clean, speaker-disjoint dataset split.
2. Normalize labels to 4 to 6 robust emotions.
3. Start with `microsoft/wavlm-base-plus`.
4. Fine-tune with at least 5 to 10 epochs and early stopping.
5. Ensemble multiple checkpoints or corpora.
6. Fuse transcript signals only after validating the audio baseline.

## Files

- `src/voice_emotion_detection/train.py`: training entrypoint
- `src/voice_emotion_detection/predict.py`: inference entrypoint
- `src/voice_emotion_detection/prepare_dataset.py`: metadata builder for RAVDESS and CREMA-D
- `src/voice_emotion_detection/model.py`: model definitions
- `src/voice_emotion_detection/dataset.py`: dataset and collator
- `src/voice_emotion_detection/features.py`: optional transcript provider utilities

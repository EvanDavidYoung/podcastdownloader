# Modal Cloud Transcription Setup

This document covers setting up and running cloud-based transcription using [Modal](https://modal.com) with GPU acceleration.

## Prerequisites

- Modal account (free tier available)
- HuggingFace account with access to pyannote models

## Installation

```bash
pip install modal numpy
modal setup  # Authenticate with Modal
```

## HuggingFace Token Setup

Speaker diarization requires a HuggingFace token with access to pyannote models.

1. Get a token at https://huggingface.co/settings/tokens
2. Accept the model terms at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Create a Modal secret:

```bash
modal secret create huggingface HF_TOKEN=your_token_here
```

## Usage

```bash
# Transcribe a local file
modal run scripts/modal/transcribe_modal.py --audio-path "./downloads/episode.mp3"

# Transcribe from URL
modal run scripts/modal/transcribe_modal.py --audio-url "https://example.com/podcast.mp3"

# Transcribe latest episode from RSS feed
modal run scripts/modal/transcribe_modal.py --rss-url "https://example.com/feed.xml"

# With options
modal run scripts/modal/transcribe_modal.py --audio-path "file.mp3" --language zh --to-traditional
```

## Output

Transcripts are saved to `transcript.json` by default. Use `--output` to specify a different path.

## Features

- **GPU acceleration**: Runs on NVIDIA T4 GPU in the cloud
- **Speaker diarization**: Automatically identifies different speakers
- **Word merging**: Chinese characters merged into words using jieba
- **Traditional conversion**: Optional simplified to traditional Chinese conversion

## Architecture

The Modal app:
1. Uploads audio to Modal's cloud infrastructure
2. Runs WhisperX transcription on a T4 GPU
3. Aligns transcript for word-level timestamps
4. Runs speaker diarization (if HF token available)
5. Merges Chinese characters into words
6. Returns the transcript to your local machine

Model weights are cached in a Modal Volume (`whisperx-models`) to speed up subsequent runs.

## Compatibility Notes

The script includes several workarounds for dependency compatibility issues:

### PyTorch 2.6+ weights_only change

PyTorch 2.6 changed `torch.load()` to use `weights_only=True` by default, which breaks loading pyannote models that use omegaconf. The script monkey-patches `torch.load` to force `weights_only=False`.

### HuggingFace Hub API change

Newer versions of `huggingface_hub` removed the `use_auth_token` parameter that pyannote-audio still uses. The script pins `huggingface_hub<0.25.0` to maintain compatibility.

### Modal API updates

- `@modal.web_endpoint` renamed to `@modal.fastapi_endpoint`
- FastAPI must be explicitly installed in the image
- Volume mounts cannot use `/root/.cache` (non-empty in base image)

### WhisperX API changes

The diarization API moved from `whisperx.DiarizationPipeline` to `whisperx.diarize.DiarizationPipeline`.

## Troubleshooting

### "numpy is required locally"

Install numpy in your local environment:
```bash
pip install numpy
```

### "cannot mount volume on non-empty path"

This is handled by mounting to `/cache/models` instead of `/root/.cache`.

### Speaker diarization not running

Ensure you've:
1. Created the Modal secret with your HF token
2. Accepted the pyannote model terms on HuggingFace

### Slow first run

The first run downloads ~3GB of model weights. Subsequent runs use cached weights from the Modal Volume.

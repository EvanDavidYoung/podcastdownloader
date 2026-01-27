# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A webapp for downloading podcasts and generating transcripts. Currently in early development with local hosting for both frontend and backend.

## Python Environment

**Use miniforge (conda), not the .venv.**

The project uses miniforge for Python environment management. The `.venv` directory is legacy and should not be used.

```bash
# Modal and transcription commands use miniforge base environment
# Modal is installed at: /Users/evanyoung/miniforge3/bin/modal

# For local WhisperX transcription, use the whisperx conda environment
conda activate whisperx
```

## Development Commands

```bash
# Run the podcast downloader
python scripts/download_podcast.py

# Local transcription (requires whisperx conda env)
conda activate whisperx
python scripts/transcribe_local.py

# Cloud transcription via Modal (uses miniforge base)
modal run scripts/transcribe_modal.py --audio-path "downloads/episode.mp3"

# Run the web API server (dev mode with hot reload)
modal serve scripts/app.py

# Deploy to production
modal deploy scripts/transcribe_modal.py
modal deploy scripts/app.py
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for full system documentation.

**Pipeline**: Download → Transcribe → Post-Process → View

**Scripts** (in `scripts/`):
- `download_podcast.py` - Download latest episode from RSS feed
- `transcribe_local.py` - Batch transcribe audio locally via WhisperX
- `transcribe_modal.py` - Cloud transcription via Modal GPU
- `merge_chinese_words.py` - Merge character-level to word-level Chinese
- `convert_to_traditional.py` - Simplified → Traditional Chinese conversion

**Frontend** (in `web/`):
- `transcript-player.html` - Browser-based transcript player with word sync

## Key Directories

- `scripts/` - All CLI tools
- `web/` - Frontend files
- `docs/` - Project documentation
- `downloads/` - Downloaded podcast audio files (gitignored)

## Secrets

Secrets are stored in `.env` (gitignored). To set up:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your token
```

To run a command with your token loaded:

```bash
# Option 1: Load .env and run
export $(cat .env | xargs) && python scripts/download_podcast.py

# Option 2: Use python-dotenv (if installed)
# The script can load .env automatically with: from dotenv import load_dotenv; load_dotenv()
```

## Transcription

The `scripts/transcribe_local.py` script transcribes all `.mp3` and `.m4a` files in `downloads/` using WhisperX.

```bash
conda activate whisperx
python scripts/transcribe_local.py
```

Configuration (set in `scripts/transcribe_local.py`):
- Language: Chinese (`zh`)
- Device: CPU (for M1 Macs without CUDA)
- Compute type: `int8`
- VAD method: Silero (avoids pyannote compatibility issues)

Output files are saved in `downloads/` alongside the audio:
- `.json` - Full transcript with word-level timestamps
- `.srt` - SubRip subtitle format
- `.vtt` - WebVTT subtitle format
- `.txt` - Plain text
- `.tsv` - Tab-separated values

## Testing

**Tests must be run in the whisperx conda environment** (requires jieba, opencc, and other dependencies).

```bash
# Activate the whisperx environment first
conda activate whisperx

# Install test dependencies (one-time)
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=scripts --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=scripts --cov-report=html

# Run specific test file
pytest tests/test_merge_chinese_words.py

# Run tests matching a pattern
pytest -k "test_merge"
```

Coverage reports are generated in `htmlcov/`. Open `htmlcov/index.html` in a browser to view.

## Dependencies

**Miniforge base environment:**
- `modal` - Cloud GPU transcription and web deployment
- `numpy` - Required for Modal result deserialization
- `fastapi`, `pydantic` - Web API dependencies
- `pytest`, `pytest-cov` - Testing (install via `requirements-dev.txt`)

**Whisperx conda environment (`conda activate whisperx`):**
- WhisperX and PyTorch for local transcription

**Modal cloud (installed in container):**
- WhisperX, torch, jieba, opencc, feedparser, etc.
- See `scripts/transcribe_modal.py` for full list

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
python scripts/local/download_podcast.py

# Local transcription (requires whisperx conda env)
conda activate whisperx
python scripts/local/transcribe_local.py

# Cloud transcription via Modal (uses miniforge base)
modal run scripts/modal/transcribe_modal.py --audio-path "downloads/episode.mp3"

# Run the web API server (dev mode with hot reload)
modal serve src/app.py

# Deploy to production
modal deploy scripts/modal/transcribe_modal.py
modal deploy src/app.py
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for full system documentation.

**Pipeline**: Download → Transcribe → Post-Process → View

**Backend server** (in `src/`):
- `app.py` - FastAPI web server deployed on Modal

**Modal cloud functions** (in `scripts/modal/`):
- `transcribe_modal.py` - Cloud transcription via Modal GPU

**Local scripts** (in `scripts/local/`):
- `download_podcast.py` - Download latest episode from RSS feed
- `transcribe_local.py` - Batch transcribe audio locally via WhisperX
- `merge_chinese_words.py` - Merge character-level to word-level Chinese
- `convert_to_traditional.py` - Simplified → Traditional Chinese conversion

**Frontend** (in `web/`):
- `transcript-player.html` - Browser-based transcript player with word sync

## Key Directories

- `src/` - Backend server (FastAPI app deployed to Modal)
- `scripts/local/` - Local CLI tools
- `scripts/modal/` - Modal cloud functions
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
export $(cat .env | xargs) && python scripts/local/download_podcast.py

# Option 2: Use python-dotenv (if installed)
# The script can load .env automatically with: from dotenv import load_dotenv; load_dotenv()
```

## Transcription

The `scripts/local/transcribe_local.py` script transcribes all `.mp3` and `.m4a` files in `downloads/` using WhisperX.

```bash
conda activate whisperx
python scripts/local/transcribe_local.py
```

Configuration (set in `scripts/local/transcribe_local.py`):
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

Tests run via `uv`. All dependencies are declared in `pyproject.toml`.

```bash
# Install all dependencies (one-time, or after pyproject.toml changes)
uv sync --dev

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov=scripts/local --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src --cov=scripts/local --cov-report=html

# Run a specific test file
uv run pytest tests/test_app.py

# Run tests matching a pattern
uv run pytest -k "test_merge"
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
- See `scripts/modal/transcribe_modal.py` for full list

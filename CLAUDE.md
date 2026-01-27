# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A webapp for downloading podcasts and generating transcripts. Currently in early development with local hosting for both frontend and backend.

## Development Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Run the podcast downloader
python scripts/download_podcast.py

# For transcription, use the whisperx conda environment
conda activate whisperx
python scripts/transcribe_local.py  # Transcribe all audio files in downloads/
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

## Dependencies

Core packages: `feedparser`, `requests`, `jieba`, `opencc-python-reimplemented`

Transcription: WhisperX installed in conda environment (`conda activate whisperx`)

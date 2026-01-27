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
python downloadtranscript.py

# For transcription, use the whisperx conda environment
conda activate whisperx
python transcribe.py  # Transcribe all audio files in downloads/
```

## Architecture

**Current State**: Single-script prototype (`downloadtranscript.py`) that downloads the latest episode from a podcast RSS feed.

**Planned Architecture** (from Docs/planning.md):
- Backend: Download podcasts from URL, generate transcripts, produce JSON with metadata
- Frontend: Search interface for podcasts with title, description, and image

## Key Directories

- `downloads/` - Downloaded podcast audio files (gitignored)
- `Docs/` - Project planning documentation

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
export $(cat .env | xargs) && python downloadtranscript.py

# Option 2: Use python-dotenv (if installed)
# The script can load .env automatically with: from dotenv import load_dotenv; load_dotenv()
```

## Transcription

The `transcribe.py` script transcribes all `.mp3` and `.m4a` files in `downloads/` using WhisperX.

```bash
conda activate whisperx
python transcribe.py
```

Configuration (set in `transcribe.py`):
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

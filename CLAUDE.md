# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A webapp for downloading podcasts and generating transcripts. Currently in early development with local hosting for both frontend and backend.

## Development Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install feedparser requests

# Run the podcast downloader
python downloadtranscript.py
```

## Architecture

**Current State**: Single-script prototype (`downloadtranscript.py`) that downloads the latest episode from a podcast RSS feed.

**Planned Architecture** (from Docs/planning.md):
- Backend: Download podcasts from URL, generate transcripts, produce JSON with metadata
- Frontend: Search interface for podcasts with title, description, and image

## Key Directories

- `downloads/` - Downloaded podcast audio files (gitignored)
- `data/` - Contains WhisperX environment for transcription, database, and transcripts
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

## Dependencies

Core packages: `feedparser`, `requests`

Transcription: WhisperX setup in `data/whisperx-env/` (includes pyannote for speaker diarization)

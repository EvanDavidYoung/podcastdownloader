# PodcastDownloader

Download podcasts and generate word-synced transcripts with speaker diarization. Built for Chinese audio with cloud transcription via Modal GPU.

## Important Links

- Dev API: https://evandavidyoung--podcast-web-serve-dev.modal.run/docs
- Prod API: https://evandavidyoung--podcast-web-serve.modal.run/docs

## Project Structure

```
src/                        # Backend server (deployed to Modal)
  app.py                    # FastAPI web server
scripts/
  local/                    # Local CLI tools
    download_podcast.py     # Download episodes from RSS
    transcribe_local.py     # Transcribe audio locally (requires WhisperX)
    merge_chinese_words.py  # Merge character-level Chinese into words
    convert_to_traditional.py # Simplified → Traditional Chinese
  modal/                    # Modal cloud functions
    transcribe_modal.py     # GPU transcription on Modal
web/
  transcript-player.html    # Browser-based transcript player
```

## Setup

### Development & testing

```bash
# Install uv (if not already installed)
brew install uv

# Install all dependencies
uv sync --dev
```

### Cloud transcription (Modal)

```bash
# Install Modal CLI (uses miniforge or any Python env)
pip install modal numpy
modal setup

# Add your HuggingFace token for speaker diarization
modal secret create huggingface HF_TOKEN=<your_token>
```

### Local transcription (optional, requires WhisperX)

```bash
brew install miniforge
conda create -n whisperx python=3.11
conda activate whisperx
pip install whisperx
```

## Secrets

Copy `.env.example` to `.env` and fill in your tokens:

```bash
cp .env.example .env
```

## Backend API

FastAPI server deployed on Modal:

```bash
# Dev mode (hot reload)
modal serve src/app.py

# Production deploy
modal deploy src/app.py
```

Authentication uses `Authorization: Bearer <key>`. Keys are stored in the Modal `api-auth` secret (`API_KEY` and `SLACK_BOT_API_KEY`).

## Scripts

### Download

**`scripts/local/download_podcast.py`** — Download the latest episode from an RSS feed

```bash
python scripts/local/download_podcast.py
```

### Transcription

**`scripts/modal/transcribe_modal.py`** — Cloud transcription via Modal GPU (no local GPU required)

```bash
# From a URL
modal run scripts/modal/transcribe_modal.py --audio-url "https://example.com/podcast.mp3"

# From an RSS feed (latest episode)
modal run scripts/modal/transcribe_modal.py --rss-url "https://example.com/feed.xml"

# From a local file (uploads to Modal)
modal run scripts/modal/transcribe_modal.py --audio-path "./downloads/episode.mp3"

# With options
modal run scripts/modal/transcribe_modal.py --audio-url "..." --language zh --to-traditional

# Deploy as standalone API endpoint
modal deploy scripts/modal/transcribe_modal.py
```

**`scripts/local/transcribe_local.py`** — Transcribe locally using WhisperX (requires WhisperX env)

```bash
conda activate whisperx
python scripts/local/transcribe_local.py
```

### Post-Processing

**`scripts/local/merge_chinese_words.py`** — Merge character-level Chinese into semantic words using jieba

WhisperX outputs Chinese as individual characters; this script groups them into words.

```bash
python scripts/local/merge_chinese_words.py downloads/transcript.json
python scripts/local/merge_chinese_words.py --preview downloads/transcript.json  # dry run
```

Example: `大 | 家 | 好` → `大家 | 好`

**`scripts/local/convert_to_traditional.py`** — Convert simplified Chinese to traditional using OpenCC

```bash
python scripts/local/convert_to_traditional.py downloads/transcript.json
python scripts/local/convert_to_traditional.py --config s2tw downloads/transcript.json  # Taiwan
python scripts/local/convert_to_traditional.py --config s2hk downloads/transcript.json  # HK
```

Config options: `s2t` (default), `s2tw` (Taiwan), `s2hk` (Hong Kong), `t2s` (reverse)

### Viewer

**`web/transcript-player.html`** — Browser-based player for word-synced transcripts

1. Open `web/transcript-player.html` in a browser
2. Drop your audio file and `.json` transcript onto the drop zone
3. Words highlight as audio plays; click any word to seek

Features: word-by-word sync, click-to-seek, speaker diarization, low-confidence indicators, speed control (0.5×–2×)

## Typical Workflow

```bash
# 1. Download episode
python scripts/local/download_podcast.py

# 2. Transcribe via Modal (cloud GPU)
modal run scripts/modal/transcribe_modal.py --audio-path "downloads/episode.mp3"

# 3. Merge characters into words
python scripts/local/merge_chinese_words.py downloads/episode.json

# 4. Optionally convert to traditional
python scripts/local/convert_to_traditional.py downloads/episode_merged.json

# 5. Preview in browser
open web/transcript-player.html
```

## Testing

```bash
uv run pytest
uv run pytest --cov=src --cov=scripts/local --cov-report=term-missing
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Modal Setup](docs/modal-setup.md)

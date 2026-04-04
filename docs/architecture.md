# Architecture

This document describes the PodcastDownloader system architecture, pipeline, and future roadmap.

## Pipeline Overview

The podcast transcription workflow consists of four stages:

```
Download → Transcribe → Post-Process → View
```

1. **Download**: Fetch podcast audio from RSS feeds or direct URLs
2. **Transcribe**: Convert audio to text with word-level timestamps
3. **Post-Process**: Merge Chinese characters into words, convert to traditional
4. **View**: Interactive browser player with synced highlighting

## Scripts

All CLI scripts are located in the `scripts/` directory.

### Download

| Script | Description |
|--------|-------------|
| `download_podcast.py` | Download the latest episode from an RSS feed |

### Transcription

| Script | Description |
|--------|-------------|
| `transcribe_local.py` | Batch transcribe audio files locally using WhisperX (CPU) |
| `transcribe_modal.py` | Cloud transcription via Modal GPU (no local GPU required) |

See [modal-setup.md](modal-setup.md) for detailed Modal configuration, including HuggingFace token setup for speaker diarization.

### Post-Processing

| Script | Description |
|--------|-------------|
| `merge_chinese_words.py` | Merge character-level Chinese into semantic words using jieba |
| `convert_to_traditional.py` | Convert simplified Chinese to traditional using OpenCC |

## Data Flow

```
RSS Feed / URL
      │
      ▼
┌─────────────────┐
│ download_podcast│  → downloads/*.mp3
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ transcribe_*    │  → downloads/*.json (character-level)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ merge_chinese   │  → downloads/*_merged.json (word-level)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ convert_to_trad │  → downloads/*_traditional.json
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ transcript-     │  Browser playback with
│ player.html     │  word-synced highlighting
└─────────────────┘
```

## Output Formats

Transcription generates multiple output formats in `downloads/`:

| Extension | Format | Description |
|-----------|--------|-------------|
| `.json` | WhisperX JSON | Full transcript with word-level timestamps |
| `.srt` | SubRip | Standard subtitle format |
| `.vtt` | WebVTT | Web Video Text Tracks |
| `.txt` | Plain text | Raw transcript text |
| `.tsv` | Tab-separated | Timestamp and text columns |

## Future Roadmap

### Backend
- Download podcasts from arbitrary URLs
- Generate transcripts with metadata as JSON
- ~~Add speaker diarization~~ (implemented in Modal)
- Input/output directory polling for batch processing

### Frontend
- Search interface for podcasts
- Display title, description, and cover image
- File selector for transcript generation
- Integrated playback with transcript sync

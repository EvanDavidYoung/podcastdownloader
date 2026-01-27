# PodcastDownloader

Download podcasts and generate word-synced transcripts for Chinese audio.

## Quick Start

```bash
# Install Miniforge (if not already installed)
brew install miniforge
conda init "$(basename "$SHELL")"
# Restart terminal

# Install Modal for cloud transcription
pip install modal numpy
modal setup
```

See [docs/getting-started.md](docs/getting-started.md) for full installation instructions.

## Scripts

### Download

**`scripts/download_podcast.py`** - Download the latest episode from an RSS feed

```bash
python scripts/download_podcast.py
```

### Transcription

**`scripts/transcribe_local.py`** - Batch transcribe audio files locally using WhisperX

```bash
conda activate whisperx
python scripts/transcribe_local.py
```

**`scripts/transcribe_modal.py`** - Cloud transcription via Modal GPU (no local GPU required)

```bash
pip install modal
modal setup
modal run scripts/transcribe_modal.py --audio-url "https://example.com/podcast.mp3"
```

Options:
```bash
# Transcribe from RSS feed (latest episode)
modal run scripts/transcribe_modal.py --rss-url "https://example.com/feed.xml"

# Transcribe local file (uploads to Modal)
modal run scripts/transcribe_modal.py --audio-path "./downloads/episode.mp3"

# With options
modal run scripts/transcribe_modal.py --audio-url "..." --language zh --to-traditional
```

Deploy as API:
```bash
modal deploy scripts/transcribe_modal.py
```

### Post-Processing

**`scripts/merge_chinese_words.py`** - Merge character-level Chinese into semantic words using jieba

WhisperX outputs Chinese as individual characters. This script merges them into words.

```bash
# Preview (no file saved)
python scripts/merge_chinese_words.py --preview downloads/transcript.json

# Merge and save to transcript_merged.json
python scripts/merge_chinese_words.py downloads/transcript.json

# Merge and save to specific path
python scripts/merge_chinese_words.py downloads/transcript.json output.json
```

Example: `大 | 家 | 好` → `大家 | 好`

**`scripts/convert_to_traditional.py`** - Convert simplified Chinese to traditional using OpenCC

```bash
# Convert to traditional (saves to transcript_traditional.json)
python scripts/convert_to_traditional.py downloads/transcript.json

# Preview conversion
python scripts/convert_to_traditional.py --preview downloads/transcript.json

# Use Taiwan standard
python scripts/convert_to_traditional.py --config s2tw downloads/transcript.json

# Use Hong Kong standard
python scripts/convert_to_traditional.py --config s2hk downloads/transcript.json
```

Config options: `s2t` (default), `s2tw` (Taiwan), `s2hk` (Hong Kong), `t2s` (reverse)

### Viewer

**`web/transcript-player.html`** - Browser-based player for testing word-synced transcripts

1. Open `web/transcript-player.html` in a browser
2. Drag and drop your `.mp3` and `.json` files onto the drop zone
3. Words highlight as audio plays; click any word to jump to that time

Features:
- Word-by-word highlighting synced to audio
- Click-to-seek on any word
- Low confidence word indicators (toggleable)
- Playback speed control (0.5x - 2x)

## Typical Workflow

```bash
# 1. Download podcast episode
python scripts/download_podcast.py

# 2. Transcribe via Modal (cloud GPU)
modal run scripts/transcribe_modal.py --audio-path "downloads/episode.mp3"

# 3. Merge characters into words
python scripts/merge_chinese_words.py downloads/episode.json

# 4. Optionally convert to traditional
python scripts/convert_to_traditional.py downloads/episode_merged.json

# 5. Test in browser player
open web/transcript-player.html
# Drop the MP3 and merged JSON file
```

Or use local transcription with `conda activate whisperx && python scripts/transcribe_local.py`.

## Documentation

- [Getting Started](docs/getting-started.md) - Installation and setup
- [Architecture](docs/architecture.md) - System design and data flow
- [Modal Setup](docs/modal-setup.md) - Cloud transcription configuration
- [Frontend Plan](docs/frontend-plan.md) - Web app roadmap

# Getting Started

## Prerequisites

- Python 3.10+
- macOS, Linux, or Windows

## Installation

### 1. Install uv (recommended)

uv is a fast Python package manager that replaces pip.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone <repo-url>
cd PodcastDownloader
```

### 3. Create virtual environment

```bash
uv venv
```

### 4. Activate the environment

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
uv pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "import feedparser, requests, jieba, opencc; print('All dependencies installed!')"
```

## Quick Start

### Download a podcast episode

```bash
python scripts/download_podcast.py
```

### Transcribe audio files

```bash
# Using the batch transcription script
conda activate whisperx
python scripts/transcribe_local.py

# Or run WhisperX directly
whisperx "/path/to/audio.mp3" --language zh --device cpu --compute_type int8 --vad_method silero
```

### Semantic Word Merging

```bash
# Merge Chinese characters into words
python scripts/merge_chinese_words.py downloads/your_transcript.json

# Convert to traditional Chinese (optional)
python scripts/convert_to_traditional.py downloads/your_transcript_merged.json
```

### Test in the player

1. Open `web/transcript-player.html` in your browser
2. Drag and drop your `.mp3` and `_merged.json` files
3. Click play and watch words highlight as they're spoken

## Secrets Setup

Some features require API tokens. Copy the example file and add your credentials:

```bash
cp .env.example .env
# Edit .env with your tokens
```

## Cloud Transcription (Optional)

If you don't have a local GPU, use Modal to run transcription in the cloud:

```bash
pip install modal
modal setup
modal run scripts/transcribe_modal.py --audio-url "https://example.com/podcast.mp3"
```

See [README.md](../README.md) for full Modal documentation.

## Next Steps

- See [README.md](../README.md) for full tool documentation
- See [architecture.md](architecture.md) for system design

## Troubleshooting

### Python version issues on macOS

If you're getting version conflicts or errors with the system Python, install Miniforge (or Miniconda) to manage Python versions independently.

```bash
# Install Miniforge via Homebrew
brew install miniforge

# Initialize conda for your shell
conda init "$(basename "$SHELL")"

# Restart your terminal, then create an environment with Python 3.12
conda create -n podcast python=3.12
conda activate podcast
```

Then continue with the uv setup from step 1.

Alternatively, you can use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) which works the same way.

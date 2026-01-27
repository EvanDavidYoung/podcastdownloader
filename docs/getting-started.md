# Getting Started

## Prerequisites

- macOS (Apple Silicon or Intel)
- Homebrew

## Installation

### 1. Install Miniforge

Miniforge provides conda for managing Python environments.

```bash
brew install miniforge
conda init "$(basename "$SHELL")"
```

Restart your terminal after running `conda init`.

### 2. Clone the repository

```bash
git clone <repo-url>
cd PodcastDownloader
```

### 3. Install Modal

Modal is used for cloud GPU transcription and the web API.

```bash
# Install in base conda environment
pip install modal numpy

# Authenticate with Modal (one-time setup)
modal setup
```

### 4. Set up Modal secrets

Create the HuggingFace secret for speaker diarization:

```bash
# Get a token from https://huggingface.co/settings/tokens
# Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

### 5. (Optional) Set up WhisperX for local transcription

If you want to run transcription locally instead of on Modal:

```bash
# Create whisperx environment
conda create -n whisperx python=3.10
conda activate whisperx

# Install PyTorch (CPU for macOS)
pip install torch torchaudio

# Install WhisperX
pip install git+https://github.com/m-bain/whisperx.git
```

## Quick Start

### Cloud transcription (recommended)

```bash
# Transcribe from URL
modal run scripts/transcribe_modal.py --audio-url "https://example.com/podcast.mp3"

# Transcribe from RSS feed (latest episode)
modal run scripts/transcribe_modal.py --rss-url "https://example.com/feed.xml"

# Transcribe local file
modal run scripts/transcribe_modal.py --audio-path "downloads/episode.mp3"
```

### Local transcription

```bash
conda activate whisperx
python scripts/transcribe_local.py  # Transcribes all audio in downloads/
```

### Post-processing

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

## Web API (Development)

```bash
# Deploy the transcription functions first
modal deploy scripts/transcribe_modal.py

# Run the web API server (hot reload)
modal serve scripts/app.py
```

The API will be available at the URL printed by Modal. Visit `/docs` for Swagger UI.

## Secrets Setup

For podcast downloading, copy the example file and add your credentials:

```bash
cp .env.example .env
# Edit .env with your tokens
```

## Next Steps

- See [modal-setup.md](modal-setup.md) for detailed Modal configuration
- See [architecture.md](architecture.md) for system design
- See [frontend-plan.md](frontend-plan.md) for the web app roadmap

## Troubleshooting

### Modal not found

Make sure you're using the miniforge base environment:

```bash
# Check where modal is installed
which modal
# Should be: /Users/<you>/miniforge3/bin/modal

# If not found, install it
pip install modal
```

### WhisperX environment issues

If WhisperX has issues, try recreating the environment:

```bash
conda deactivate
conda env remove -n whisperx
conda create -n whisperx python=3.10
conda activate whisperx
pip install torch torchaudio
pip install git+https://github.com/m-bain/whisperx.git
```

### Speaker diarization not working

1. Verify the Modal secret exists: `modal secret list`
2. Check you accepted the pyannote model terms on HuggingFace
3. See [modal-setup.md](modal-setup.md) for detailed troubleshooting

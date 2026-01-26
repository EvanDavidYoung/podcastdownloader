Simple Webapp for downloading and generating transcripts for podcasts.


## Setup

See [Docs/gettingStarted.md](Docs/gettingStarted.md) for full installation instructions.

Quick start:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv
uv venv && source .venv/bin/activate              # Create & activate venv
uv pip install -r requirements.txt                # Install dependencies
```

## Transcript Tools

### Transcript Player (`transcript-player.html`)

A browser-based player for testing word-synced transcripts.

1. Open `transcript-player.html` in a browser
2. Drag and drop your `.mp3` and `.json` files onto the drop zone
3. Words highlight as audio plays; click any word to jump to that time

Features:
- Word-by-word highlighting synced to audio
- Click-to-seek on any word
- Low confidence word indicators (toggleable)
- Playback speed control (0.5x - 2x)

### Merge Chinese Words (`merge_chinese_words.py`)

WhisperX outputs Chinese as individual characters. This script merges them into semantic words using jieba segmentation.

```bash
# Preview the merge (no file saved)
python merge_chinese_words.py --preview input.json

# Merge and save to input_merged.json
python merge_chinese_words.py input.json

# Merge and save to specific path
python merge_chinese_words.py input.json output.json
```

Example: `大 | 家 | 好` → `大家 | 好`

### Convert to Traditional Chinese (`convert_to_traditional.py`)

Convert simplified Chinese transcripts to traditional Chinese using OpenCC.

```bash
# Preview conversion
python convert_to_traditional.py --preview input.json

# Convert to traditional (saves to input_traditional.json)
python convert_to_traditional.py input.json

# Use Taiwan standard
python convert_to_traditional.py --config s2tw input.json

# Use Hong Kong standard
python convert_to_traditional.py --config s2hk input.json

# Reverse: traditional to simplified
python convert_to_traditional.py --config t2s input.json
```

Config options:
- `s2t` - Simplified → Traditional (default)
- `s2tw` - Simplified → Traditional (Taiwan)
- `s2hk` - Simplified → Traditional (Hong Kong)
- `t2s` - Traditional → Simplified

### Typical Workflow

```bash
source .venv/bin/activate

# 1. Generate transcript with WhisperX (creates character-level JSON)
# 2. Merge characters into words
python merge_chinese_words.py transcript.json

# 3. Optionally convert to traditional
python convert_to_traditional.py transcript_merged.json

# 4. Test in browser player
open transcript-player.html
# Drop the MP3 and merged JSON file
```

"""
Modal app for podcast transcription with WhisperX on GPU.

Usage:
    # Install modal
    pip install modal

    # Authenticate (first time only)
    modal setup

    # Run transcription on a URL
    modal run modal_app.py --audio-url "https://example.com/podcast.mp3"

    # Run on a local file (uploads to Modal)
    modal run modal_app.py --audio-path "./downloads/episode.mp3"

    # Deploy as a web endpoint
    modal deploy modal_app.py
"""

import modal

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "whisperx @ git+https://github.com/m-bain/whisperx.git",
        "feedparser",
        "requests",
        "jieba",
        "opencc-python-reimplemented",
    )
)

app = modal.App("podcast-transcriber", image=image)

# Create a volume to cache models (saves download time on subsequent runs)
model_cache = modal.Volume.from_name("whisperx-models", create_if_missing=True)


@app.function(
    gpu="T4",  # Options: "T4", "A10G", "A100", "H100"
    timeout=1800,  # 30 minutes max
    volumes={"/root/.cache": model_cache},
)
def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.mp3",
    language: str = "zh",
    merge_words: bool = True,
    to_traditional: bool = False,
    hf_token: str = None,
) -> dict:
    """
    Transcribe audio using WhisperX on GPU.

    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename (for output naming)
        language: Language code (e.g., "zh", "en")
        merge_words: Merge Chinese characters into words using jieba
        to_traditional: Convert simplified Chinese to traditional
        hf_token: HuggingFace token for speaker diarization (optional)

    Returns:
        dict with transcript data
    """
    import tempfile
    import os
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Using device: {device}, compute_type: {compute_type}")

    # Write audio to temp file
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        # Load model and transcribe
        print("Loading WhisperX model...")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type, language=language)

        print("Transcribing...")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)

        # Align whisper output
        print("Aligning transcript...")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Optional: Speaker diarization
        if hf_token:
            print("Running speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        transcript = {
            "segments": result["segments"],
            "word_segments": result.get("word_segments", []),
            "language": language,
        }

        # Merge Chinese words if requested
        if merge_words and language in ["zh", "ja"]:
            print("Merging words with jieba...")
            transcript = merge_chinese_words(transcript)

        # Convert to traditional if requested
        if to_traditional and language == "zh":
            print("Converting to traditional Chinese...")
            transcript = convert_to_traditional(transcript)

        return transcript

    finally:
        os.unlink(audio_path)


def merge_chinese_words(data: dict) -> dict:
    """Merge character-level Chinese into words using jieba."""
    import jieba

    def merge_words_in_segment(words):
        if not words:
            return words

        full_text = ''.join(w['word'] for w in words)
        segmented = list(jieba.cut(full_text))

        merged_words = []
        char_idx = 0

        for seg_word in segmented:
            if not seg_word.strip():
                continue

            seg_len = len(seg_word)
            start_idx = char_idx
            chars_consumed = 0
            end_idx = char_idx

            while chars_consumed < seg_len and end_idx < len(words):
                chars_consumed += len(words[end_idx]['word'])
                end_idx += 1

            if start_idx < len(words) and end_idx <= len(words):
                merged_word = {
                    'word': seg_word,
                    'start': words[start_idx]['start'],
                    'end': words[end_idx - 1]['end'],
                    'score': sum(w.get('score', 1.0) for w in words[start_idx:end_idx]) / (end_idx - start_idx)
                }
                merged_words.append(merged_word)

            char_idx = end_idx

        return merged_words

    for segment in data.get('segments', []):
        if 'words' in segment:
            segment['words'] = merge_words_in_segment(segment['words'])

    if 'word_segments' in data:
        data['word_segments'] = merge_words_in_segment(data['word_segments'])

    return data


def convert_to_traditional(data: dict, config: str = 's2t') -> dict:
    """Convert simplified Chinese to traditional."""
    from opencc import OpenCC
    cc = OpenCC(config)

    for segment in data.get('segments', []):
        if 'text' in segment:
            segment['text'] = cc.convert(segment['text'])
        if 'words' in segment:
            for word in segment['words']:
                if 'word' in word:
                    word['word'] = cc.convert(word['word'])

    if 'word_segments' in data:
        for word in data['word_segments']:
            if 'word' in word:
                word['word'] = cc.convert(word['word'])

    return data


@app.function(gpu="T4", timeout=1800, volumes={"/root/.cache": model_cache})
def transcribe_from_url(
    url: str,
    language: str = "zh",
    merge_words: bool = True,
    to_traditional: bool = False,
) -> dict:
    """Download and transcribe audio from a URL."""
    import requests

    print(f"Downloading from {url}...")
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    filename = url.split("/")[-1].split("?")[0] or "audio.mp3"

    return transcribe_audio.local(
        audio_bytes=response.content,
        filename=filename,
        language=language,
        merge_words=merge_words,
        to_traditional=to_traditional,
    )


@app.function(gpu="T4", timeout=1800, volumes={"/root/.cache": model_cache})
def transcribe_from_rss(
    rss_url: str,
    episode_index: int = 0,
    language: str = "zh",
    merge_words: bool = True,
    to_traditional: bool = False,
) -> dict:
    """Download and transcribe the latest (or specified) episode from an RSS feed."""
    import feedparser
    import requests

    print(f"Fetching RSS feed: {rss_url}")
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        raise ValueError("No episodes found in feed")

    episode = feed.entries[episode_index]
    title = episode.get("title", "Unknown")
    print(f"Episode: {title}")

    # Find audio URL
    audio_url = None
    for link in episode.get("links", []):
        if link.get("type", "").startswith("audio/"):
            audio_url = link.get("href")
            break

    if not audio_url:
        for enclosure in episode.get("enclosures", []):
            if enclosure.get("type", "").startswith("audio/"):
                audio_url = enclosure.get("url")
                break

    if not audio_url:
        raise ValueError(f"No audio found for episode: {title}")

    print(f"Downloading: {audio_url}")
    response = requests.get(audio_url, timeout=300)
    response.raise_for_status()

    result = transcribe_audio.local(
        audio_bytes=response.content,
        filename=f"{title}.mp3",
        language=language,
        merge_words=merge_words,
        to_traditional=to_traditional,
    )

    result["episode_title"] = title
    return result


# Web endpoint for API access
@app.function(gpu="T4", timeout=1800, volumes={"/root/.cache": model_cache})
@modal.web_endpoint(method="POST")
def transcribe_endpoint(request: dict) -> dict:
    """
    Web endpoint for transcription.

    POST body:
    {
        "url": "https://example.com/audio.mp3",  // or "rss_url" for RSS feeds
        "language": "zh",
        "merge_words": true,
        "to_traditional": false
    }
    """
    if "rss_url" in request:
        return transcribe_from_rss.local(
            rss_url=request["rss_url"],
            episode_index=request.get("episode_index", 0),
            language=request.get("language", "zh"),
            merge_words=request.get("merge_words", True),
            to_traditional=request.get("to_traditional", False),
        )
    elif "url" in request:
        return transcribe_from_url.local(
            url=request["url"],
            language=request.get("language", "zh"),
            merge_words=request.get("merge_words", True),
            to_traditional=request.get("to_traditional", False),
        )
    else:
        return {"error": "Please provide 'url' or 'rss_url' in request body"}


@app.local_entrypoint()
def main(
    audio_url: str = None,
    audio_path: str = None,
    rss_url: str = None,
    language: str = "zh",
    merge_words: bool = True,
    to_traditional: bool = False,
    output: str = None,
):
    """CLI entrypoint for running transcription."""
    import json

    if rss_url:
        result = transcribe_from_rss.remote(
            rss_url=rss_url,
            language=language,
            merge_words=merge_words,
            to_traditional=to_traditional,
        )
    elif audio_url:
        result = transcribe_from_url.remote(
            url=audio_url,
            language=language,
            merge_words=merge_words,
            to_traditional=to_traditional,
        )
    elif audio_path:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        result = transcribe_audio.remote(
            audio_bytes=audio_bytes,
            filename=audio_path,
            language=language,
            merge_words=merge_words,
            to_traditional=to_traditional,
        )
    else:
        print("Please provide --audio-url, --audio-path, or --rss-url")
        return

    # Save output
    output_path = output or "transcript.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to: {output_path}")

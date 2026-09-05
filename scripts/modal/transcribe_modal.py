"""
Modal app for podcast transcription with WhisperX on GPU.

Usage:
    # Install modal and numpy
    pip install modal numpy

    # Authenticate (first time only)
    modal setup

    # Run transcription on a URL — any site yt-dlp supports, or a direct file
    modal run scripts/modal/transcribe_modal.py --audio-url "https://youtube.com/watch?v=..."

    # See which caption tracks a URL already offers
    modal run scripts/modal/transcribe_modal.py --audio-url "..." --list-subs

    # Reuse those captions instead of running WhisperX (no GPU)
    modal run scripts/modal/transcribe_modal.py --audio-url "..." --subtitles --subtitle-langs en

    # Run on a local file (uploads to Modal)
    modal run scripts/modal/transcribe_modal.py --audio-path "./downloads/episode.mp3"

    # Deploy as a web endpoint
    modal deploy scripts/modal/transcribe_modal.py
"""

try:
    import numpy  # noqa: F401 - Required locally to deserialize Modal results
except ImportError:
    raise ImportError("numpy is required locally to deserialize Modal results. Install with: pip install numpy")

import modal

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "omegaconf",
        # huggingface_hub was pinned <0.25.0 for an old pyannote incompatibility.
        # whisperx now requires >=0.28.1, so the pin made the image unbuildable.
        # Let whisperx and pyannote resolve it themselves.
        "whisperx @ git+https://github.com/m-bain/whisperx.git",
        "feedparser",
        "requests",
        "yt-dlp",
        "jieba",
        "opencc-python-reimplemented",
        "fastapi[standard]",
    )
)

# The CPU-only paths (probe, caption reuse) need none of the whisperx stack, so
# they get their own image and cold-start in seconds instead of pulling torch.
light_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("yt-dlp", "requests")
)

app = modal.App("podcast-transcriber", image=image)

# Create a volume to cache models (saves download time on subsequent runs)
model_cache = modal.Volume.from_name("whisperx-models", create_if_missing=True)
MODEL_CACHE_PATH = "/cache/models"

# Persistent volume for completed job artifacts (transcript, audio, metadata)
jobs_volume = modal.Volume.from_name("podcast-jobs", create_if_missing=True)
JOBS_PATH = "/jobs"


def build_diarization_pipeline(pipeline_cls, hf_token, device):
    """Construct a whisperx DiarizationPipeline across signature versions.

    whisperx is installed from git main, and it renamed the auth argument from
    `use_auth_token` to `token`. Passing the wrong one raises TypeError at
    construction and fails the whole job, so pick the name this build actually
    accepts rather than pinning against a moving upstream.
    """
    import inspect

    params = inspect.signature(pipeline_cls.__init__).parameters
    key = "token" if "token" in params else "use_auth_token"
    return pipeline_cls(**{key: hf_token}, device=device)


@app.function(
    gpu="T4",  # Options: "T4", "A10G", "A100", "H100"
    timeout=1800,  # 30 minutes max
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
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
    import torch

    # Workaround for PyTorch 2.6+ weights_only issue with pyannote/omegaconf
    # Monkey-patch torch.load to force weights_only=False for pyannote models
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    import whisperx

    # Use mounted volume for model cache
    os.environ["HF_HOME"] = MODEL_CACHE_PATH
    os.environ["TORCH_HOME"] = MODEL_CACHE_PATH

    # Use HF_TOKEN from Modal secret if not passed directly
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

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

        # Optional: Speaker diarization.
        # The transcript is already complete by this point, so a diarization
        # failure must not discard it. pyannote's default model is a gated HF
        # repo, and callers that only want text shouldn't be blocked on being
        # granted access to it.
        diarization_error = None
        if hf_token:
            print("Running speaker diarization...")
            try:
                from whisperx.diarize import DiarizationPipeline, assign_word_speakers
                diarize_model = build_diarization_pipeline(DiarizationPipeline, hf_token, device)
                diarize_segments = diarize_model(audio)
                result = assign_word_speakers(diarize_segments, result)
            except Exception as exc:
                diarization_error = f"{type(exc).__name__}: {exc}"
                print(f"Diarization failed, returning transcript without speakers: {diarization_error}")

        transcript = {
            "segments": result["segments"],
            "word_segments": result.get("word_segments", []),
            "language": language,
        }
        if diarization_error:
            transcript["diarization_error"] = diarization_error

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


def parse_silence_intervals(ffmpeg_stderr: str) -> list:
    """Parse ffmpeg silencedetect filter output into a list of (silence_start, silence_end)."""
    starts, ends = [], []
    for line in ffmpeg_stderr.splitlines():
        if "silence_start:" in line:
            starts.append(float(line.split("silence_start:")[1].strip()))
        elif "silence_end:" in line:
            ends.append(float(line.split("silence_end:")[1].split("|")[0].strip()))
    if len(starts) != len(ends):
        starts = starts[: len(ends)]
    return list(zip(starts, ends))


def speech_intervals_from_silence(silence_intervals: list, total_duration: float, min_speech: float = 0.15) -> list:
    """Invert silence intervals against total duration to get speech intervals."""
    speech = []
    cursor = 0.0
    for s, e in silence_intervals:
        if s > cursor:
            speech.append((cursor, s))
        cursor = e
    if cursor < total_duration:
        speech.append((cursor, total_duration))
    return [(s, e) for s, e in speech if e - s > min_speech]


def merge_speech_intervals(speech_intervals: list, merge_silence_max: float = 1.2, max_chunk_sec: float = 25.0) -> list:
    """Bridge speech intervals separated by a short pause, capped at max_chunk_sec so
    language auto-detection stays granular enough to catch a real speaker/language switch."""
    if not speech_intervals:
        return []
    merged = [speech_intervals[0]]
    for s, e in speech_intervals[1:]:
        cur_s, cur_e = merged[-1]
        gap = s - cur_e
        if gap <= merge_silence_max and (e - cur_s) <= max_chunk_sec:
            merged[-1] = (cur_s, e)
        else:
            merged.append((s, e))
    return merged


def merge_speaker_turns(rows: list, merge_gap: float = 0.5, min_turn: float = 0.4) -> list:
    """Collapse raw diarization rows into contiguous speaker turns.

    Diarization emits many short rows per speaker; transcribing each one separately
    would cut mid-sentence. Consecutive rows from the same speaker separated by less
    than merge_gap are bridged, and turns shorter than min_turn are dropped as noise.

    Args:
        rows: iterable of dicts with "start", "end", "speaker"
        merge_gap: max silence (seconds) to bridge between same-speaker rows
        min_turn: drop turns shorter than this (seconds)

    Returns:
        list of {"start", "end", "speaker"} sorted by start
    """
    ordered = sorted(
        ({"start": float(r["start"]), "end": float(r["end"]), "speaker": r["speaker"]} for r in rows),
        key=lambda r: (r["start"], r["end"]),
    )
    if not ordered:
        return []

    merged = [ordered[0]]
    for row in ordered[1:]:
        cur = merged[-1]
        if row["speaker"] == cur["speaker"] and row["start"] - cur["end"] <= merge_gap:
            cur["end"] = max(cur["end"], row["end"])
        else:
            merged.append(row)

    return [t for t in merged if t["end"] - t["start"] >= min_turn]


def normalize_word_timings(segment: dict, anomaly_factor: float = 3.0) -> dict:
    """Repair segments where forced alignment dumps a multi-second span on one word.

    WhisperX's Chinese char-level alignment periodically assigns the whole leading
    silence of a segment to its first character (e.g. 请 spanning 0.258-6.479s while
    every following char spans ~0.02s), which makes word-by-word playback highlighting
    useless. When the first word's duration exceeds anomaly_factor x the median word
    duration, redistribute the segment's word timings proportionally by character
    count across the segment span.

    Mutates and returns the segment.
    """
    words = [w for w in segment.get("words", []) if w.get("start") is not None and w.get("end") is not None]
    if len(words) < 3:
        return segment

    durations = sorted(w["end"] - w["start"] for w in words)
    median = durations[len(durations) // 2]
    first = words[0]["end"] - words[0]["start"]
    if median <= 0 or first <= median * anomaly_factor:
        return segment

    span_start, span_end = segment["start"], segment["end"]
    total_chars = sum(max(len(w["word"]), 1) for w in words)
    span = span_end - span_start
    if total_chars == 0 or span <= 0:
        return segment

    cursor = span_start
    for w in words:
        share = span * (max(len(w["word"]), 1) / total_chars)
        w["start"] = round(cursor, 3)
        w["end"] = round(cursor + share, 3)
        cursor += share

    return segment


def detect_speech_chunks(
    audio_path: str, noise_db: int = -30, min_silence: float = 0.5,
    merge_silence_max: float = 1.2, max_chunk_sec: float = 25.0,
) -> list:
    """Run ffmpeg silencedetect and return merged (start, end) speech chunks."""
    import subprocess

    proc = subprocess.run(
        ["ffmpeg", "-i", str(audio_path), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    duration = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    silence = parse_silence_intervals(proc.stderr)
    speech = speech_intervals_from_silence(silence, duration)
    return merge_speech_intervals(speech, merge_silence_max, max_chunk_sec)


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def transcribe_bilingual_audio(
    audio_bytes: bytes,
    filename: str = "audio.mp4",
    align: bool = True,
) -> dict:
    """
    Transcribe audio/video with per-segment language auto-detection, for recordings
    that alternate languages within one track (e.g. a speaker + live interpreter).

    Unlike transcribe_audio(), language is not forced -- the file is split into
    speech chunks via silence detection, and WhisperX auto-detects language
    independently for each chunk, so each segment is tagged with the language
    actually spoken instead of one language forced across the whole file.

    When align=True (default), each segment also gets word-level timestamps via
    WhisperX's forced-alignment step (one align model loaded per language seen),
    so callers can split a segment into sentence-length clips instead of being
    stuck with chunk-length ones.

    Returns dict with {"segments": [{"start", "end", "language", "text", "words"}, ...]}.
    """
    import tempfile
    import os
    import torch

    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    import whisperx
    from whisperx.audio import SAMPLE_RATE

    os.environ["HF_HOME"] = MODEL_CACHE_PATH
    os.environ["TORCH_HOME"] = MODEL_CACHE_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Using device: {device}, compute_type: {compute_type}")

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        print("Detecting speech chunks...")
        chunks = detect_speech_chunks(audio_path)
        print(f"{len(chunks)} chunks")

        audio = whisperx.load_audio(audio_path)

        print("Loading WhisperX model (auto language-detect mode)...")
        # language=None keeps the pipeline in auto-detect mode; WhisperX resets its
        # cached tokenizer after every transcribe() call when no language is preset,
        # so each chunk below gets a fresh language-detection pass instead of reusing
        # whatever was detected for the first chunk.
        model = whisperx.load_model("large-v3", device, compute_type=compute_type, language=None, vad_method="silero")

        segments = []
        for i, (start, end) in enumerate(chunks):
            s_idx, e_idx = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
            result = model.transcribe(audio[s_idx:e_idx], batch_size=16, language=None)
            lang = result["language"]
            for seg in result["segments"]:
                text = seg["text"].strip()
                if not text:
                    continue
                segments.append({
                    "start": round(start + seg["start"], 3),
                    "end": round(start + seg["end"], 3),
                    "language": lang,
                    "text": text,
                })
            print(f"[{i+1}/{len(chunks)}] {start:.1f}-{end:.1f}s lang={lang}")

        if align:
            print("Aligning for word-level timestamps...")
            align_models = {}  # language -> (model, metadata), loaded lazily and reused
            by_language = {}
            for seg in segments:
                by_language.setdefault(seg["language"], []).append(seg)

            for lang, lang_segments in by_language.items():
                if lang not in align_models:
                    try:
                        align_models[lang] = whisperx.load_align_model(language_code=lang, device=device)
                    except Exception as e:
                        print(f"No align model for '{lang}', skipping word timestamps for it: {e}")
                        continue
                model_a, metadata = align_models[lang]
                aligned = whisperx.align(lang_segments, model_a, metadata, audio, device, return_char_alignments=False)
                for orig, aligned_seg in zip(lang_segments, aligned["segments"]):
                    orig["words"] = [
                        {"word": w["word"], "start": w.get("start"), "end": w.get("end")}
                        for w in aligned_seg.get("words", [])
                    ]

        return {"segments": segments}

    finally:
        os.unlink(audio_path)


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def transcribe_diarized_bilingual(
    audio_bytes: bytes,
    filename: str = "audio.mp3",
    hf_token: str = None,
    num_speakers: int = None,
    align: bool = True,
    merge_words: bool = True,
    to_traditional: bool = True,
    speaker_languages: dict = None,
    fix_alignment: bool = True,
) -> dict:
    """
    Transcribe a two-language recording by speaker, one language locked per speaker.

    Where transcribe_bilingual_audio() infers who is talking from the language it
    detects per chunk, this runs three passes:

      1. Diarize the whole file to find who speaks when.
      2. For each speaker, detect their language once from their longest turns
         (long slices; whisper's language detection is unreliable under 30s).
      3. Transcribe each speaker's turns with that language forced, so neither
         pass fights language detection and neither hears the other speaker.

    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename (for suffix detection)
        hf_token: HuggingFace token for diarization (falls back to HF_TOKEN secret)
        num_speakers: Pin the speaker count when known (sets min and max)
        align: Produce word-level timestamps via forced alignment
        merge_words: Merge Chinese characters into words with jieba (zh/ja speakers)
        to_traditional: Convert simplified to traditional Chinese (zh speakers)
        speaker_languages: Skip detection, e.g. {"SPEAKER_00": "en", "SPEAKER_01": "zh"}
        fix_alignment: Repair anomalous first-word spans (see normalize_word_timings)

    Returns:
        dict with "speakers" (pass 1 + detected language), "per_speaker" (one
        player-schema transcript per speaker) and "combined" (all speakers merged,
        sorted by start time).
    """
    import tempfile
    import os
    from collections import Counter

    import torch

    # Workaround for PyTorch 2.6+ weights_only issue with pyannote/omegaconf
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    import whisperx
    from whisperx.audio import SAMPLE_RATE

    os.environ["HF_HOME"] = MODEL_CACHE_PATH
    os.environ["TORCH_HOME"] = MODEL_CACHE_PATH

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Speaker diarization requires an HF token (HF_TOKEN secret or hf_token arg)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Using device: {device}, compute_type: {compute_type}")

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    def slice_audio(audio, start, end):
        return audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]

    try:
        audio = whisperx.load_audio(audio_path)

        # --- Pass 1: who speaks when -------------------------------------------------
        print("Pass 1: speaker diarization...")
        from whisperx.diarize import DiarizationPipeline

        diarize_model = build_diarization_pipeline(DiarizationPipeline, hf_token, device)
        diarize_kwargs = {}
        if num_speakers:
            diarize_kwargs = {"min_speakers": num_speakers, "max_speakers": num_speakers}
        diarize_segments = diarize_model(audio, **diarize_kwargs)

        rows = diarize_segments.to_dict("records")
        turns = merge_speaker_turns(rows)
        if not turns:
            raise RuntimeError("Diarization produced no usable speaker turns")

        turns_by_speaker = {}
        for turn in turns:
            turns_by_speaker.setdefault(turn["speaker"], []).append(turn)

        for speaker, spk_turns in sorted(turns_by_speaker.items()):
            total = sum(t["end"] - t["start"] for t in spk_turns)
            print(f"  {speaker}: {len(spk_turns)} turns, {total:.1f}s of speech")

        # --- Pass 2a: one language per speaker ---------------------------------------
        speaker_languages = dict(speaker_languages or {})
        lang_votes = {}

        undetected = [s for s in turns_by_speaker if s not in speaker_languages]
        if undetected:
            print("Pass 2a: detecting language per speaker...")
            detect_model = whisperx.load_model(
                "large-v3", device, compute_type=compute_type, language=None, vad_method="silero"
            )
            for speaker in undetected:
                # Longest turns first -- short slices give unreliable language detection.
                sampled, budget = [], 60.0
                for turn in sorted(turns_by_speaker[speaker], key=lambda t: t["end"] - t["start"], reverse=True):
                    if budget <= 0:
                        break
                    sampled.append(turn)
                    budget -= turn["end"] - turn["start"]

                votes = Counter()
                for turn in sampled:
                    result = detect_model.transcribe(
                        slice_audio(audio, turn["start"], turn["end"]), batch_size=16, language=None
                    )
                    votes[result["language"]] += 1

                lang_votes[speaker] = dict(votes)
                speaker_languages[speaker] = votes.most_common(1)[0][0]
                print(f"  {speaker}: {speaker_languages[speaker]} (votes: {dict(votes)})")

            del detect_model

        # --- Pass 2b/3: transcribe each speaker with their language locked -----------
        print("Pass 2b/3: transcribing each speaker with language locked...")
        models = {}  # language -> model, so en + zh costs two loads not one per speaker
        per_speaker_segments = {}

        for speaker in sorted(turns_by_speaker):
            lang = speaker_languages[speaker]
            if lang not in models:
                models[lang] = whisperx.load_model(
                    "large-v3", device, compute_type=compute_type, language=lang, vad_method="silero"
                )
            model = models[lang]

            segments = []
            spk_turns = turns_by_speaker[speaker]
            for i, turn in enumerate(spk_turns):
                result = model.transcribe(
                    slice_audio(audio, turn["start"], turn["end"]), batch_size=16, language=lang
                )
                for seg in result["segments"]:
                    text = seg["text"].strip()
                    if not text:
                        continue
                    segments.append({
                        "start": round(turn["start"] + seg["start"], 3),
                        "end": round(turn["start"] + seg["end"], 3),
                        "text": text,
                        "speaker": speaker,
                        "language": lang,
                    })
                print(f"  [{speaker} {i+1}/{len(spk_turns)}] {turn['start']:.1f}-{turn['end']:.1f}s")

            per_speaker_segments[speaker] = segments

        # --- Word-level alignment ----------------------------------------------------
        if align:
            print("Aligning for word-level timestamps...")
            align_models = {}  # language -> (model, metadata)
            for speaker in sorted(per_speaker_segments):
                lang = speaker_languages[speaker]
                if lang not in align_models:
                    try:
                        align_models[lang] = whisperx.load_align_model(language_code=lang, device=device)
                    except Exception as e:
                        print(f"No align model for '{lang}', skipping word timestamps for it: {e}")
                        align_models[lang] = None
                if align_models[lang] is None:
                    continue

                model_a, metadata = align_models[lang]
                # Align against each turn's own slice so the aligner never sees the
                # other speaker, then shift word times back to the global timeline.
                for turn in turns_by_speaker[speaker]:
                    in_turn = [s for s in per_speaker_segments[speaker]
                               if turn["start"] <= s["start"] < turn["end"]]
                    if not in_turn:
                        continue
                    local = [{"start": s["start"] - turn["start"],
                              "end": s["end"] - turn["start"],
                              "text": s["text"]} for s in in_turn]
                    try:
                        aligned = whisperx.align(
                            local, model_a, metadata,
                            slice_audio(audio, turn["start"], turn["end"]),
                            device, return_char_alignments=False,
                        )
                    except Exception as e:
                        print(f"  align failed for {speaker} turn {turn['start']:.1f}s: {e}")
                        continue
                    for orig, aligned_seg in zip(in_turn, aligned["segments"]):
                        orig["words"] = [
                            {
                                "word": w["word"],
                                "start": round(turn["start"] + w["start"], 3) if w.get("start") is not None else None,
                                "end": round(turn["start"] + w["end"], 3) if w.get("end") is not None else None,
                                "score": w.get("score"),
                            }
                            for w in aligned_seg.get("words", [])
                        ]
                        if fix_alignment:
                            normalize_word_timings(orig)

        # --- Per-speaker post-processing and packaging -------------------------------
        per_speaker = {}
        for speaker, segments in per_speaker_segments.items():
            lang = speaker_languages[speaker]
            transcript = {"segments": segments, "language": lang, "speaker": speaker}
            if merge_words and lang in ["zh", "ja"]:
                print(f"Merging words with jieba for {speaker}...")
                transcript = merge_chinese_words(transcript)
            if to_traditional and lang == "zh":
                print(f"Converting {speaker} to traditional Chinese...")
                transcript = convert_to_traditional(transcript)
            # Built last so the flat list mirrors the post-processed segment words
            # instead of aliasing dicts that jieba/OpenCC would then touch twice.
            transcript["word_segments"] = [w for s in transcript["segments"] for w in s.get("words", [])]
            per_speaker[speaker] = transcript

        combined_segments = sorted(
            (seg for t in per_speaker.values() for seg in t["segments"]),
            key=lambda s: (s["start"], s["end"]),
        )
        detected = {speaker_languages[s] for s in per_speaker}
        combined = {
            "segments": combined_segments,
            "word_segments": [w for s in combined_segments for w in s.get("words", [])],
            "language": detected.pop() if len(detected) == 1 else "mixed",
        }

        speakers_meta = {
            speaker: {
                "language": speaker_languages[speaker],
                "lang_votes": lang_votes.get(speaker, {}),
                "total_speech": round(sum(t["end"] - t["start"] for t in spk_turns), 3),
                "turn_count": len(spk_turns),
                "segment_count": len(per_speaker[speaker]["segments"]),
                "turns": [{"start": t["start"], "end": t["end"]} for t in spk_turns],
            }
            for speaker, spk_turns in sorted(turns_by_speaker.items())
        }

        return {"speakers": speakers_meta, "per_speaker": per_speaker, "combined": combined}

    finally:
        os.unlink(audio_path)


# ---------------------
# Media source resolution (yt-dlp)
# ---------------------
#
# A URL that already points at an audio file is streamed directly. Anything
# else — a YouTube/Vimeo/SoundCloud/Twitter page, a news article with an
# embedded player, a bare .mp4 — is handed to yt-dlp, which extracts the audio
# stream and transcodes it to mp3. yt-dlp also reports the site's subtitles and
# auto-captions, so a job can reuse an existing transcript instead of paying
# for GPU transcription.

DIRECT_AUDIO_EXTENSIONS = (
    ".mp3", ".m4a", ".m4b", ".aac", ".oga", ".ogg", ".opus", ".flac", ".wav", ".wma",
)

# Subtitle formats we can parse, best first. json3 carries per-word offsets;
# vtt only does when the site emits inline <hh:mm:ss.mmm> cue tags (YouTube does).
SUBTITLE_FORMAT_PREFERENCE = ("json3", "vtt", "srt")

# Some CDNs serve podcast audio only to browser-shaped clients.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def looks_like_direct_audio(url: str, content_type: str = None) -> bool:
    """True when a URL can be fetched with a plain GET instead of yt-dlp.

    Video types deliberately return False — yt-dlp pulls just the audio stream
    rather than making us ship a whole video file to the GPU.
    """
    import os
    from urllib.parse import urlparse

    if content_type:
        main = content_type.split(";")[0].strip().lower()
        if main.startswith("audio/") or main == "application/ogg":
            return True
        if main.startswith(("text/", "video/", "application/xml", "application/rss")):
            return False
        if main == "application/json":
            return False

    ext = os.path.splitext(urlparse(url).path)[1].lower()
    return ext in DIRECT_AUDIO_EXTENSIONS


def _timestamp_to_seconds(stamp: str) -> float:
    """Parse HH:MM:SS.mmm / MM:SS.mmm / SRT's comma variant into seconds."""
    parts = stamp.strip().replace(",", ".").split(":")
    seconds = float(parts[-1])
    if len(parts) > 1:
        seconds += int(parts[-2]) * 60
    if len(parts) > 2:
        seconds += int(parts[-3]) * 3600
    return seconds


def _finalize_subtitle_transcript(segments: list, language: str = None) -> dict:
    """Drop rolling-caption repeats, backfill open end times, build word_segments."""
    cleaned = []
    for segment in segments:
        # Rolling captions re-emit the previous line verbatim before extending it.
        if cleaned and cleaned[-1]["text"] == segment["text"]:
            continue
        cleaned.append(segment)

    for i, segment in enumerate(cleaned):
        next_start = cleaned[i + 1]["start"] if i + 1 < len(cleaned) else None
        if segment.get("end") is None:
            segment["end"] = next_start if next_start is not None else segment["start"]
        # Rolling caption windows stay on screen into the next cue, so raw ends
        # overlap the following start and word timings run backwards. Trim them
        # back so the player can scan word_segments in order.
        if next_start is not None and segment["end"] > next_start:
            segment["end"] = next_start
        for word in segment.get("words") or []:
            end = segment["end"] if word.get("end") is None else min(word["end"], segment["end"])
            word["end"] = max(end, word["start"])

    return {
        "segments": cleaned,
        "word_segments": [w for s in cleaned for w in s.get("words") or []],
        "language": language,
    }


def parse_json3_subtitles(data: dict, language: str = None) -> dict:
    """Convert YouTube's json3 caption format into the transcript shape.

    json3 gives a per-word `tOffsetMs` inside each event, which is what makes
    it worth preferring over vtt for the word-synced player.
    """
    segments = []

    for event in data.get("events") or []:
        if event.get("aAppend"):
            continue  # rolling-window repeat of the previous event
        segs = event.get("segs")
        start_ms = event.get("tStartMs")
        if not segs or start_ms is None:
            continue

        start = start_ms / 1000.0
        duration_ms = event.get("dDurationMs")
        end = start + duration_ms / 1000.0 if duration_ms else None

        text = "".join(seg.get("utf8", "") for seg in segs).strip()
        if not text:
            continue

        # A lone seg is a whole caption line, not a word — only multi-seg events
        # (auto-captions) carry real per-word offsets. Faking words from a line
        # would hand the player one multi-second "word".
        words = []
        if len(segs) > 1:
            for seg in segs:
                piece = seg.get("utf8", "")
                if not piece.strip():
                    continue
                words.append({
                    "word": piece.strip(),
                    "start": start + seg.get("tOffsetMs", 0) / 1000.0,
                    "end": None,
                })
            for i, word in enumerate(words[:-1]):
                word["end"] = words[i + 1]["start"]
            if words:
                words[-1]["end"] = end

        segments.append({"start": start, "end": end, "text": text, "words": words})

    return _finalize_subtitle_transcript(segments, language)


def _parse_vtt_payload(payload: str, start: float, end: float) -> tuple:
    """Split one cue body into (plain text, word list).

    Words come back empty unless the cue carries inline <hh:mm:ss.mmm> timing
    tags, which only some sources (YouTube auto-captions) emit.
    """
    import re

    inline_time = re.compile(r"<(\d{1,2}:\d{2}:\d{2}[.,]\d{3})>")
    tag = re.compile(r"</?[a-zA-Z][^>]*>")

    plain = " ".join(tag.sub("", inline_time.sub("", payload)).split())
    if not inline_time.search(payload):
        return plain, []

    words = []
    word_start = start
    # re.split with a capturing group alternates text, timestamp, text, ...
    for index, part in enumerate(inline_time.split(payload)):
        if index % 2:
            word_start = _timestamp_to_seconds(part)
            continue
        token = " ".join(tag.sub("", part).split())
        if token:
            words.append({"word": token, "start": word_start, "end": None})

    for i, word in enumerate(words[:-1]):
        word["end"] = words[i + 1]["start"]
    if words:
        words[-1]["end"] = end

    return plain, words


def parse_vtt_subtitles(text: str, language: str = None) -> dict:
    """Convert a WebVTT or SRT subtitle file into the transcript shape."""
    import re

    cue_timing = re.compile(
        r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})"
    )

    lines = text.replace("\r\n", "\n").split("\n")
    segments = []
    i = 0

    while i < len(lines):
        match = cue_timing.search(lines[i])
        if not match:
            i += 1
            continue

        start = _timestamp_to_seconds(match.group(1))
        end = _timestamp_to_seconds(match.group(2))
        i += 1

        payload = []
        while i < len(lines) and lines[i].strip():
            payload.append(lines[i])
            i += 1

        if not payload:
            continue
        cue_text, words = _parse_vtt_payload("\n".join(payload), start, end)
        if cue_text:
            segments.append({"start": start, "end": end, "text": cue_text, "words": words})

    # YouTube pairs each timed cue with an untimed duplicate of the same line;
    # once any cue has word timings, the wordless ones are those duplicates.
    if any(s["words"] for s in segments):
        segments = [s for s in segments if s["words"]]

    return _finalize_subtitle_transcript(segments, language)


def parse_subtitle_data(raw, ext: str, language: str = None) -> dict:
    """Dispatch raw subtitle bytes to the parser for its format."""
    import json

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    if ext == "json3":
        return parse_json3_subtitles(json.loads(raw), language)
    if ext in ("vtt", "srt"):
        return parse_vtt_subtitles(raw, language)
    raise ValueError(f"Unsupported subtitle format: {ext}")


def _match_language(available, languages) -> str:
    """Pick a track language, tolerating regional suffixes (en matches en-US)."""
    available = list(available)
    if not available:
        return None
    if not languages:
        return available[0]

    by_lower = {code.lower(): code for code in available}
    for wanted in languages:
        wanted = wanted.lower()
        if wanted in by_lower:
            return by_lower[wanted]
        for code_lower, code in by_lower.items():
            if code_lower.split("-")[0] == wanted.split("-")[0]:
                return code
    return None


def _preferred_format(tracks: list) -> dict:
    for ext in SUBTITLE_FORMAT_PREFERENCE:
        for track in tracks:
            if track.get("ext") == ext:
                return track
    return None


def select_subtitle_track(info: dict, languages=None, allow_auto: bool = True):
    """Choose the best subtitle track from a yt-dlp info dict.

    Returns (language, kind, track) where kind is "subtitles" or
    "automatic_captions", or None when nothing usable is available.
    Human-authored subtitles always beat auto-captions.
    """
    if not languages and info.get("language"):
        languages = [info["language"]]

    sources = [("subtitles", info.get("subtitles") or {})]
    if allow_auto:
        sources.append(("automatic_captions", info.get("automatic_captions") or {}))

    for kind, available in sources:
        language = _match_language(available.keys(), languages)
        if language is None:
            continue
        track = _preferred_format(available[language])
        if track:
            return language, kind, track
    return None


def _write_cookie_file(cookies_txt: str = None) -> str:
    """Materialise Netscape-format cookies for yt-dlp; returns a path or None.

    Falls back to the YTDLP_COOKIES env var (set it from a Modal secret) so
    sites that bot-check datacenter IPs can still be reached.
    """
    import os
    import tempfile

    cookies_txt = cookies_txt or os.environ.get("YTDLP_COOKIES")
    if not cookies_txt:
        return None

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(cookies_txt)
        return f.name


def _ytdl_options(cookies_path: str = None, extra: dict = None) -> dict:
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "http_headers": {"User-Agent": BROWSER_USER_AGENT},
    }
    if cookies_path:
        options["cookiefile"] = cookies_path
    if extra:
        options.update(extra)
    return options


def probe_media(url: str, cookies_txt: str = None) -> dict:
    """Return metadata and available caption languages without downloading."""
    import os
    import yt_dlp

    cookies_path = _write_cookie_file(cookies_txt)
    try:
        with yt_dlp.YoutubeDL(_ytdl_options(cookies_path, {"skip_download": True})) as ydl:
            info = ydl.extract_info(url, download=False)
    finally:
        if cookies_path:
            os.unlink(cookies_path)

    return {
        "title": info.get("title"),
        "extractor": info.get("extractor_key") or info.get("extractor"),
        "duration": info.get("duration"),
        "language": info.get("language"),
        "webpage_url": info.get("webpage_url") or url,
        "subtitles": sorted(info.get("subtitles") or {}),
        "automatic_captions": sorted(info.get("automatic_captions") or {}),
    }


def fetch_subtitles(
    url: str,
    languages: list = None,
    allow_auto: bool = True,
    cookies_txt: str = None,
) -> dict:
    """Download an existing transcript for a URL, or None if the site has none.

    Auto-captions are cheaper than WhisperX but noticeably worse: no reliable
    punctuation, no diarization, no per-language handling for bilingual audio.
    """
    import os
    import yt_dlp

    cookies_path = _write_cookie_file(cookies_txt)
    try:
        with yt_dlp.YoutubeDL(_ytdl_options(cookies_path, {"skip_download": True})) as ydl:
            info = ydl.extract_info(url, download=False)
            selection = select_subtitle_track(info, languages, allow_auto)
            if selection is None:
                print(f"No usable subtitles for {url}")
                return None
            language, kind, track = selection
            print(f"Using {kind} track '{language}' ({track.get('ext')})")
            raw = ydl.urlopen(track["url"]).read()
    finally:
        if cookies_path:
            os.unlink(cookies_path)

    transcript = parse_subtitle_data(raw, track.get("ext"), language)
    transcript["source"] = kind
    transcript["subtitle_format"] = track.get("ext")
    transcript["title"] = info.get("title")
    return transcript


def download_media(url: str, cookies_txt: str = None) -> dict:
    """Fetch audio bytes for any URL.

    Plain audio URLs are streamed with requests; everything else goes through
    yt-dlp, which covers ~1800 sites plus a generic extractor for pages with an
    embedded player, and transcodes whatever it finds to mp3.
    """
    import glob
    import os
    import shutil
    import tempfile
    from urllib.parse import urlparse

    import requests
    import yt_dlp

    try:
        response = requests.get(
            url, timeout=300, stream=True, headers={"User-Agent": BROWSER_USER_AGENT}
        )
        response.raise_for_status()
        if looks_like_direct_audio(url, response.headers.get("Content-Type")):
            filename = os.path.basename(urlparse(url).path) or "audio.mp3"
            print(f"Direct audio download: {filename}")
            return {
                "audio_bytes": response.content,
                "filename": filename,
                "title": filename,
                "extractor": None,
                "duration": None,
                "webpage_url": url,
                "source": "direct",
            }
        response.close()
    except requests.RequestException as e:
        print(f"Direct download failed ({e}); falling back to yt-dlp")

    print(f"Extracting audio with yt-dlp: {url}")
    cookies_path = _write_cookie_file(cookies_txt)
    tmpdir = tempfile.mkdtemp()
    try:
        options = _ytdl_options(cookies_path, {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
        })
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)

        produced = sorted(glob.glob(os.path.join(tmpdir, "*.mp3")))
        if not produced:
            raise RuntimeError(f"yt-dlp produced no audio file for {url}")

        with open(produced[0], "rb") as f:
            audio_bytes = f.read()

        title = info.get("title") or os.path.basename(produced[0])
        print(f"Extracted {len(audio_bytes) / 1e6:.1f} MB from '{title}'")
        return {
            "audio_bytes": audio_bytes,
            "filename": os.path.basename(produced[0]),
            "title": title,
            "extractor": info.get("extractor_key") or info.get("extractor"),
            "duration": info.get("duration"),
            "webpage_url": info.get("webpage_url") or url,
            "source": "yt-dlp",
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if cookies_path:
            os.unlink(cookies_path)


def _save_job_artifacts(job_id: str, transcript: dict, audio_bytes: bytes, metadata: dict):
    """Write transcript/audio/metadata for a job into the shared volume."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    job_dir = Path(f"{JOBS_PATH}/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False))
    if audio_bytes:
        (job_dir / "audio.mp3").write_bytes(audio_bytes)
    metadata = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    (job_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False))
    jobs_volume.commit()
    print(f"Artifacts saved to volume at /jobs/{job_id}/")


@app.function(image=light_image, timeout=300)
def probe_url(url: str, cookies_txt: str = None) -> dict:
    """Report title, duration and available caption languages for a URL."""
    return probe_media(url, cookies_txt=cookies_txt)


@app.function(image=light_image, timeout=1800, volumes={JOBS_PATH: jobs_volume})
def fetch_transcript(
    url: str,
    languages: list = None,
    allow_auto: bool = True,
    cookies_txt: str = None,
    job_id: str = None,
) -> dict:
    """Reuse a site's own subtitles instead of transcribing (no GPU).

    Raises if the URL has no usable captions — call transcribe_from_url when
    you want a WhisperX transcript regardless.
    """
    transcript = fetch_subtitles(
        url, languages=languages, allow_auto=allow_auto, cookies_txt=cookies_txt
    )
    if transcript is None:
        raise ValueError(f"No subtitles available for {url}")

    if job_id:
        # The player streams audio from the volume, so fetch it even though the
        # transcript came from the site.
        media = download_media(url, cookies_txt=cookies_txt)
        _save_job_artifacts(job_id, transcript, media["audio_bytes"], {
            "title": transcript.get("title") or media["title"],
            "language": transcript.get("language"),
            "type": "subtitles",
            "input": url,
            "source": transcript.get("source"),
            "extractor": media.get("extractor"),
        })

    return transcript


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={MODEL_CACHE_PATH: model_cache, JOBS_PATH: jobs_volume},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def transcribe_from_url(
    url: str,
    language: str = "zh",
    merge_words: bool = True,
    to_traditional: bool = False,
    hf_token: str = None,
    job_id: str = None,
    use_subtitles: bool = False,
    subtitle_languages: list = None,
    cookies_txt: str = None,
) -> dict:
    """Download and transcribe audio from any URL yt-dlp can resolve.

    With use_subtitles, an existing transcript on the source site is used when
    one exists and WhisperX only runs as a fallback. That short-circuit still
    happens inside this GPU container; use fetch_transcript directly when you
    know you only want captions.
    """
    result = None
    if use_subtitles:
        result = fetch_subtitles(
            url, languages=subtitle_languages or [language], cookies_txt=cookies_txt
        )

    # Only pay for the download when there is something to transcribe, or when
    # the job needs audio on the volume for the player to stream.
    media = download_media(url, cookies_txt=cookies_txt) if result is None or job_id else None

    if result is None:
        result = transcribe_audio.local(
            audio_bytes=media["audio_bytes"],
            filename=media["filename"],
            language=language,
            merge_words=merge_words,
            to_traditional=to_traditional,
            hf_token=hf_token,
        )
        result["source"] = "whisperx"

    if job_id:
        _save_job_artifacts(job_id, result, media["audio_bytes"], {
            "title": media["title"],
            "language": result.get("language") or language,
            "type": "url",
            "input": url,
            "source": result.get("source"),
            "extractor": media.get("extractor"),
        })

    return result


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={MODEL_CACHE_PATH: model_cache, JOBS_PATH: jobs_volume},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def transcribe_from_rss(
    rss_url: str,
    episode_index: int = 0,
    episode_title: str = None,
    language: str = "zh",
    merge_words: bool = True,
    to_traditional: bool = False,
    hf_token: str = None,
    job_id: str = None,
) -> dict:
    """Download and transcribe the latest (or specified) episode from an RSS feed."""
    import feedparser

    print(f"Fetching RSS feed: {rss_url}")
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        raise ValueError("No episodes found in feed")

    if episode_title:
        query = episode_title.lower()
        # Prefer substring match, fall back to best word-overlap score
        episode = next(
            (e for e in feed.entries if query in e.get("title", "").lower()),
            None,
        )
        if episode is None:
            query_words = set(query.split())
            episode = max(
                feed.entries,
                key=lambda e: len(query_words & set(e.get("title", "").lower().split())),
            )
        print(f"Title search '{episode_title}' matched: {episode.get('title')}")
    else:
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
    media = download_media(audio_url)

    result = transcribe_audio.local(
        audio_bytes=media["audio_bytes"],
        filename=f"{title}.mp3",
        language=language,
        merge_words=merge_words,
        to_traditional=to_traditional,
        hf_token=hf_token,
    )

    result["episode_title"] = title
    result["source"] = "whisperx"

    if job_id:
        _save_job_artifacts(job_id, result, media["audio_bytes"], {
            "title": title,
            "language": language,
            "type": "rss",
            "input": rss_url,
        })

    return result


# Web endpoint for API access
@app.function(
    gpu="T4",
    timeout=1800,
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
@modal.fastapi_endpoint(method="POST")
def transcribe_endpoint(request: dict) -> dict:
    """
    Web endpoint for transcription.

    POST body:
    {
        "url": "https://youtube.com/watch?v=...",  // any yt-dlp-supported URL,
                                                   // or "rss_url" for RSS feeds
        "language": "zh",
        "merge_words": true,
        "to_traditional": false,
        "use_subtitles": false                     // reuse the site's captions
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
            use_subtitles=request.get("use_subtitles", False),
            subtitle_languages=request.get("subtitle_languages"),
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
    hf_token: str = None,
    output: str = None,
    bilingual: bool = False,
    diarized: bool = False,
    num_speakers: int = None,
    output_dir: str = None,
    traditional: bool = True,
    subtitles: bool = False,
    subtitle_langs: str = None,
    cookies: str = None,
    list_subs: bool = False,
):
    """CLI entrypoint for running transcription.

    --audio-url accepts anything yt-dlp can resolve (YouTube, Vimeo,
    SoundCloud, a news page with an embedded player, a direct .mp3), not just
    direct audio files. Pass --list-subs to see which caption tracks a URL
    offers, or --subtitles to reuse them instead of running WhisperX.

    Sites that bot-check datacenter IPs need browser cookies: export them in
    Netscape format and pass --cookies cookies.txt.

    Pass --bilingual with --audio-path to auto-detect language per speech
    segment instead of forcing --language across the whole file (for
    recordings that alternate languages, e.g. a speaker + live interpreter).

    Pass --diarized with --audio-path to identify speakers first and then
    transcribe each speaker with their own language locked. Writes
    speakers.json, one speaker_<ID>.json per speaker, and combined.json into
    --output-dir; all three are playable in web/transcript-player.html.

    --diarized uses --traditional (default on, disable with --no-traditional)
    rather than --to-traditional: whisper emits a mix of simplified and
    traditional characters within one file, so normalising the script is the
    useful default here.
    """
    import json

    cookies_txt = open(cookies, encoding="utf-8").read() if cookies else None
    subtitle_languages = (
        [code.strip() for code in subtitle_langs.split(",") if code.strip()]
        if subtitle_langs
        else None
    )

    if list_subs:
        if not audio_url:
            print("--list-subs requires --audio-url")
            return
        info = probe_url.remote(url=audio_url, cookies_txt=cookies_txt)
        duration = f"{info['duration'] / 60:.1f} min" if info.get("duration") else "unknown"
        print(f"{info['title']} — {info['extractor']}, {duration}")
        print(f"  subtitles:          {', '.join(info['subtitles']) or 'none'}")
        autos = info["automatic_captions"]
        preview = ", ".join(autos[:12]) + (f", +{len(autos) - 12} more" if len(autos) > 12 else "")
        print(f"  automatic captions: {preview or 'none'}")
        return

    if diarized:
        if not audio_path:
            print("--diarized requires --audio-path")
            return
        from pathlib import Path

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        result = transcribe_diarized_bilingual.remote(
            audio_bytes=audio_bytes,
            filename=audio_path,
            hf_token=hf_token,
            num_speakers=num_speakers,
            merge_words=merge_words,
            to_traditional=traditional,
        )

        out_dir = Path(output_dir or "diarized_output")
        out_dir.mkdir(parents=True, exist_ok=True)

        def _write(name, payload):
            path = out_dir / name
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return path

        print(f"\nSaved to {out_dir}/")
        print(f"  {_write('speakers.json', result['speakers']).name}")
        for speaker, transcript in result["per_speaker"].items():
            path = _write(f"speaker_{speaker}.json", transcript)
            meta = result["speakers"][speaker]
            print(f"  {path.name} — {meta['language']}, {meta['segment_count']} segments, "
                  f"{meta['total_speech']:.1f}s speech")
        combined = result["combined"]
        _write("combined.json", combined)
        print(f"  combined.json — {len(combined['segments'])} segments ({combined['language']})")
        return

    if bilingual:
        if not audio_path:
            print("--bilingual currently requires --audio-path")
            return
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        result = transcribe_bilingual_audio.remote(audio_bytes=audio_bytes, filename=audio_path)
        output_path = output or "transcript.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Transcript saved to: {output_path}")
        return

    if rss_url:
        result = transcribe_from_rss.remote(
            rss_url=rss_url,
            language=language,
            merge_words=merge_words,
            to_traditional=to_traditional,
            hf_token=hf_token,
        )
    elif audio_url:
        if subtitles:
            # Captions only — no GPU, fails outright when the site has none.
            result = fetch_transcript.remote(
                url=audio_url,
                languages=subtitle_languages or [language],
                cookies_txt=cookies_txt,
            )
        else:
            result = transcribe_from_url.remote(
                url=audio_url,
                language=language,
                merge_words=merge_words,
                to_traditional=to_traditional,
                hf_token=hf_token,
                subtitle_languages=subtitle_languages,
                cookies_txt=cookies_txt,
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
            hf_token=hf_token,
        )
    else:
        print("Please provide --audio-url, --audio-path, or --rss-url")
        return

    # Save output
    output_path = output or "transcript.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to: {output_path}")

"""
Modal app for podcast transcription with WhisperX on GPU.

Usage:
    # Install modal and numpy
    pip install modal numpy

    # Authenticate (first time only)
    modal setup

    # Run transcription on a URL
    modal run scripts/modal/transcribe_modal.py --audio-url "https://example.com/podcast.mp3"

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
        "huggingface_hub<0.25.0",  # Pin to avoid use_auth_token deprecation error
        "whisperx @ git+https://github.com/m-bain/whisperx.git",
        "feedparser",
        "requests",
        "jieba",
        "opencc-python-reimplemented",
        "fastapi[standard]",
    )
)

app = modal.App("podcast-transcriber", image=image)

# Create a volume to cache models (saves download time on subsequent runs)
model_cache = modal.Volume.from_name("whisperx-models", create_if_missing=True)
MODEL_CACHE_PATH = "/cache/models"

# Persistent volume for completed job artifacts (transcript, audio, metadata)
jobs_volume = modal.Volume.from_name("podcast-jobs", create_if_missing=True)
JOBS_PATH = "/jobs"


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

        # Optional: Speaker diarization
        if hf_token:
            print("Running speaker diarization...")
            from whisperx.diarize import DiarizationPipeline, assign_word_speakers
            diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = assign_word_speakers(diarize_segments, result)

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

        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
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
) -> dict:
    """Download and transcribe audio from a URL."""
    import json
    import requests
    from datetime import datetime, timezone
    from pathlib import Path

    print(f"Downloading from {url}...")
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    filename = url.split("/")[-1].split("?")[0] or "audio.mp3"

    result = transcribe_audio.local(
        audio_bytes=response.content,
        filename=filename,
        language=language,
        merge_words=merge_words,
        to_traditional=to_traditional,
        hf_token=hf_token,
    )

    if job_id:
        job_dir = Path(f"{JOBS_PATH}/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "transcript.json").write_text(json.dumps(result, ensure_ascii=False))
        (job_dir / "audio.mp3").write_bytes(response.content)
        metadata = {
            "job_id": job_id,
            "title": filename,
            "language": language,
            "type": "url",
            "input": url,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (job_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False))
        jobs_volume.commit()
        print(f"Artifacts saved to volume at /jobs/{job_id}/")

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
    import json
    import feedparser
    import requests
    from datetime import datetime, timezone
    from pathlib import Path

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
    response = requests.get(audio_url, timeout=300)
    response.raise_for_status()

    result = transcribe_audio.local(
        audio_bytes=response.content,
        filename=f"{title}.mp3",
        language=language,
        merge_words=merge_words,
        to_traditional=to_traditional,
        hf_token=hf_token,
    )

    result["episode_title"] = title

    if job_id:
        job_dir = Path(f"{JOBS_PATH}/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "transcript.json").write_text(json.dumps(result, ensure_ascii=False))
        (job_dir / "audio.mp3").write_bytes(response.content)
        metadata = {
            "job_id": job_id,
            "title": title,
            "language": language,
            "type": "rss",
            "input": rss_url,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (job_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False))
        jobs_volume.commit()
        print(f"Artifacts saved to volume at /jobs/{job_id}/")

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
    hf_token: str = None,
    output: str = None,
    bilingual: bool = False,
    diarized: bool = False,
    num_speakers: int = None,
    output_dir: str = None,
    traditional: bool = True,
):
    """CLI entrypoint for running transcription.

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
        result = transcribe_from_url.remote(
            url=audio_url,
            language=language,
            merge_words=merge_words,
            to_traditional=to_traditional,
            hf_token=hf_token,
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

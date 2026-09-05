#!/usr/bin/env python3
"""
Transcribe audio/video with per-segment language auto-detection.

WhisperX (and OpenAI Whisper) normally detect the spoken language once for an
entire file and transcribe everything as that language. That breaks down for
recordings that alternate between languages within a single track -- e.g. a
speaker talking in Chinese with a live English interpreter, or vice versa.

This script splits the audio into speech chunks using silence detection, then
runs WhisperX's auto language-detection independently on each chunk, so each
segment is tagged and transcribed in whatever language was actually spoken.

Requires the `whisperx` conda environment (see CLAUDE.md):
    conda activate whisperx
    python scripts/local/transcribe_bilingual.py <input> [-o output.json]

Accepts any ffmpeg-readable input (mp3, m4a, mov, mp4, ...).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def probe_duration(path: str) -> float:
    """Return media duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def parse_silence_intervals(ffmpeg_stderr: str) -> list[tuple[float, float]]:
    """Parse silencedetect filter output into a list of (silence_start, silence_end)."""
    starts, ends = [], []
    for line in ffmpeg_stderr.splitlines():
        if "silence_start:" in line:
            starts.append(float(line.split("silence_start:")[1].strip()))
        elif "silence_end:" in line:
            ends.append(float(line.split("silence_end:")[1].split("|")[0].strip()))
    if len(starts) != len(ends):
        # Trailing silence with no logged end (runs to EOF) -- drop the dangling start.
        starts = starts[: len(ends)]
    return list(zip(starts, ends))


def speech_intervals_from_silence(
    silence_intervals: list[tuple[float, float]], total_duration: float, min_speech: float = 0.15
) -> list[tuple[float, float]]:
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


def merge_speech_intervals(
    speech_intervals: list[tuple[float, float]], merge_silence_max: float = 1.2, max_chunk_sec: float = 25.0
) -> list[tuple[float, float]]:
    """Bridge speech intervals separated by a short pause (same speaker breathing/pausing),
    capped at max_chunk_sec so language auto-detection stays granular enough to catch
    a real speaker/language switch."""
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


def detect_speech_chunks(
    audio_path: str, noise_db: int = -30, min_silence: float = 0.5,
    merge_silence_max: float = 1.2, max_chunk_sec: float = 25.0,
) -> list[tuple[float, float]]:
    """Run ffmpeg silencedetect and return merged (start, end) speech chunks."""
    proc = subprocess.run(
        ["ffmpeg", "-i", str(audio_path), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    silence = parse_silence_intervals(proc.stderr)
    duration = probe_duration(audio_path)
    speech = speech_intervals_from_silence(silence, duration)
    return merge_speech_intervals(speech, merge_silence_max, max_chunk_sec)


def transcribe_bilingual(
    input_path: str, model_size: str = "large-v3", device: str = "cpu",
    compute_type: str = "int8", batch_size: int = 8, vad_method: str = "silero",
    progress=lambda i, n, chunk, lang, text: None,
) -> dict:
    """Transcribe `input_path`, auto-detecting language independently per speech chunk.

    Returns {"segments": [{"start", "end", "language", "text"}, ...]} with global
    timestamps. Each segment's `language` reflects what was actually detected for
    that chunk, so a bilingual recording ends up with a mix of e.g. "zh" and "en"
    segments instead of one forced language.
    """
    import whisperx
    from whisperx.audio import SAMPLE_RATE

    chunks = detect_speech_chunks(input_path)
    audio = whisperx.load_audio(str(input_path))

    # language=None keeps the pipeline in auto-detect mode; WhisperX resets its
    # cached tokenizer after every transcribe() call when no language is preset,
    # so each chunk below gets a fresh language-detection pass instead of reusing
    # whatever was detected for the first chunk.
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language=None, vad_method=vad_method)

    segments = []
    for i, (start, end) in enumerate(chunks):
        s_idx, e_idx = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        result = model.transcribe(audio[s_idx:e_idx], batch_size=batch_size, language=None)
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
        progress(i + 1, len(chunks), (start, end), lang, result["segments"][0]["text"] if result["segments"] else "")

    return {"segments": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to audio or video file (any ffmpeg-readable format)")
    parser.add_argument("-o", "--output", help="Output JSON path (default: <input stem>.bilingual.json next to input)")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".bilingual.json")

    def report(i, n, chunk, lang, text):
        print(f"[{i}/{n}] {chunk[0]:.1f}-{chunk[1]:.1f}s lang={lang} :: {text[:60]}", file=sys.stderr, flush=True)

    result = transcribe_bilingual(
        str(input_path), model_size=args.model, device=args.device,
        compute_type=args.compute_type, progress=report,
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(result['segments'])} segments to {output_path}")


if __name__ == "__main__":
    main()

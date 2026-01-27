#!/usr/bin/env python3
"""Transcribe all audio files in the downloads folder using WhisperX."""

import subprocess
import sys
from pathlib import Path

DOWNLOADS_DIR = Path(__file__).parent / "downloads"
AUDIO_EXTENSIONS = {".mp3", ".m4a"}


def get_audio_files() -> list[Path]:
    """Get all audio files in the downloads directory."""
    if not DOWNLOADS_DIR.exists():
        return []
    return [f for f in DOWNLOADS_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]


def transcribe_file(audio_path: Path) -> bool:
    """Run WhisperX transcription on a single audio file."""
    cmd = [
        "whisperx",
        str(audio_path),
        "--language", "zh",
        "--device", "cpu",
        "--compute_type", "int8",
        "--vad_method", "silero",
        "--output_dir", str(audio_path.parent),
    ]

    print(f"Transcribing: {audio_path.name}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    audio_files = get_audio_files()

    if not audio_files:
        print(f"No audio files found in {DOWNLOADS_DIR}")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio file(s) to transcribe\n")

    success_count = 0
    for audio_file in audio_files:
        if transcribe_file(audio_file):
            success_count += 1
            print(f"✓ Completed: {audio_file.name}\n")
        else:
            print(f"✗ Failed: {audio_file.name}\n")

    print(f"\nTranscription complete: {success_count}/{len(audio_files)} files succeeded")


if __name__ == "__main__":
    main()

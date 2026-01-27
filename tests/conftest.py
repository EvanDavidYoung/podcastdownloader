"""Shared test fixtures."""

import pytest
import json
from pathlib import Path
import tempfile


@pytest.fixture
def sample_transcript():
    """Sample transcript with character-level Chinese words."""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "大家好",
                "words": [
                    {"word": "大", "start": 0.0, "end": 0.5, "score": 0.95},
                    {"word": "家", "start": 0.5, "end": 1.0, "score": 0.92},
                    {"word": "好", "start": 1.0, "end": 1.5, "score": 0.98},
                ],
            },
            {
                "start": 5.0,
                "end": 10.0,
                "text": "欢迎收听",
                "words": [
                    {"word": "欢", "start": 5.0, "end": 5.5, "score": 0.90},
                    {"word": "迎", "start": 5.5, "end": 6.0, "score": 0.91},
                    {"word": "收", "start": 6.0, "end": 6.5, "score": 0.89},
                    {"word": "听", "start": 6.5, "end": 7.0, "score": 0.93},
                ],
            },
        ],
        "word_segments": [
            {"word": "大", "start": 0.0, "end": 0.5, "score": 0.95},
            {"word": "家", "start": 0.5, "end": 1.0, "score": 0.92},
            {"word": "好", "start": 1.0, "end": 1.5, "score": 0.98},
        ],
        "language": "zh",
    }


@pytest.fixture
def sample_transcript_file(sample_transcript, tmp_path):
    """Write sample transcript to a temp file and return path."""
    file_path = tmp_path / "transcript.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_transcript, f, ensure_ascii=False)
    return file_path


@pytest.fixture
def simplified_transcript():
    """Sample transcript with simplified Chinese."""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "这个节目很有意思",
                "words": [
                    {"word": "这个", "start": 0.0, "end": 1.0, "score": 0.95},
                    {"word": "节目", "start": 1.0, "end": 2.0, "score": 0.92},
                    {"word": "很", "start": 2.0, "end": 2.5, "score": 0.98},
                    {"word": "有意思", "start": 2.5, "end": 4.0, "score": 0.90},
                ],
            },
        ],
        "language": "zh",
    }


@pytest.fixture
def simplified_transcript_file(simplified_transcript, tmp_path):
    """Write simplified transcript to a temp file and return path."""
    file_path = tmp_path / "simplified.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(simplified_transcript, f, ensure_ascii=False)
    return file_path

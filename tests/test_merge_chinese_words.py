"""Tests for merge_chinese_words.py."""

import pytest
import json
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "local"))

from merge_chinese_words import (
    is_chinese_char,
    is_punctuation,
    merge_words_in_segment,
    process_transcript,
    preview_merge,
)


class TestIsChinese:
    """Tests for is_chinese_char function."""

    def test_chinese_characters(self):
        assert is_chinese_char("中") is True
        assert is_chinese_char("国") is True
        assert is_chinese_char("好") is True

    def test_non_chinese_characters(self):
        assert is_chinese_char("a") is False
        assert is_chinese_char("1") is False
        assert is_chinese_char(" ") is False
        assert is_chinese_char("!") is False


class TestIsPunctuation:
    """Tests for is_punctuation function."""

    def test_chinese_punctuation(self):
        assert is_punctuation("，") is True
        assert is_punctuation("。") is True
        assert is_punctuation("！") is True

    def test_english_punctuation(self):
        assert is_punctuation(",") is True
        assert is_punctuation(".") is True
        assert is_punctuation("!") is True

    def test_non_punctuation(self):
        assert is_punctuation("a") is False
        assert is_punctuation("中") is False
        assert is_punctuation("1") is False


class TestMergeWordsInSegment:
    """Tests for merge_words_in_segment function."""

    def test_empty_input(self):
        assert merge_words_in_segment([]) == []
        assert merge_words_in_segment(None) is None

    def test_merge_simple_chinese(self):
        """Test merging '大家好' from characters to words."""
        words = [
            {"word": "大", "start": 0.0, "end": 0.5, "score": 0.95},
            {"word": "家", "start": 0.5, "end": 1.0, "score": 0.92},
            {"word": "好", "start": 1.0, "end": 1.5, "score": 0.98},
        ]
        result = merge_words_in_segment(words)

        # jieba should segment this as "大家" + "好"
        assert len(result) == 2
        assert result[0]["word"] == "大家"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.0
        assert result[1]["word"] == "好"
        assert result[1]["start"] == 1.0
        assert result[1]["end"] == 1.5

    def test_preserves_timestamps(self):
        """Test that merged words have correct start/end times."""
        words = [
            {"word": "欢", "start": 0.0, "end": 0.3, "score": 0.9},
            {"word": "迎", "start": 0.3, "end": 0.6, "score": 0.9},
            {"word": "收", "start": 0.6, "end": 0.9, "score": 0.9},
            {"word": "听", "start": 0.9, "end": 1.2, "score": 0.9},
        ]
        result = merge_words_in_segment(words)

        # Should merge to "欢迎" + "收听"
        assert len(result) == 2
        assert result[0]["word"] == "欢迎"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 0.6
        assert result[1]["word"] == "收听"
        assert result[1]["start"] == 0.6
        assert result[1]["end"] == 1.2

    def test_averages_scores(self):
        """Test that merged words have averaged scores."""
        words = [
            {"word": "大", "start": 0.0, "end": 0.5, "score": 0.8},
            {"word": "家", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        result = merge_words_in_segment(words)

        # "大家" should have average score of 0.9
        assert result[0]["word"] == "大家"
        assert result[0]["score"] == pytest.approx(0.9)


class TestProcessTranscript:
    """Tests for process_transcript function."""

    def test_process_transcript_creates_output(self, sample_transcript_file, tmp_path):
        """Test that process_transcript creates an output file."""
        output_path = tmp_path / "output.json"
        result = process_transcript(sample_transcript_file, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_process_transcript_default_output_name(self, sample_transcript_file):
        """Test default output naming (_merged suffix)."""
        result = process_transcript(sample_transcript_file)

        expected = sample_transcript_file.parent / "transcript_merged.json"
        assert result == expected
        assert expected.exists()

        # Cleanup
        expected.unlink()

    def test_process_transcript_merges_words(self, sample_transcript_file, tmp_path):
        """Test that words are actually merged in output."""
        output_path = tmp_path / "output.json"
        process_transcript(sample_transcript_file, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        # Check first segment - "大家好" should become ["大家", "好"]
        words = result["segments"][0]["words"]
        assert len(words) == 2
        assert words[0]["word"] == "大家"
        assert words[1]["word"] == "好"

    def test_process_transcript_merges_word_segments(self, sample_transcript_file, tmp_path):
        """Test that word_segments are also merged."""
        output_path = tmp_path / "output.json"
        process_transcript(sample_transcript_file, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        # word_segments should also be merged
        word_segments = result["word_segments"]
        assert len(word_segments) == 2  # "大家" + "好"


class TestPreviewMerge:
    """Tests for preview_merge function."""

    def test_preview_outputs_comparison(self, sample_transcript_file, capsys):
        """Test that preview shows original and merged words."""
        preview_merge(sample_transcript_file)

        captured = capsys.readouterr()
        # Should show both original and merged
        assert "ORIGINAL" in captured.out
        assert "MERGED" in captured.out
        # Should show character-level in original
        assert "大" in captured.out
        # Should show word-level in merged
        assert "大家" in captured.out

    def test_preview_empty_segments(self, tmp_path, capsys):
        """Test preview with no segments."""
        empty_transcript = {"segments": [], "language": "zh"}
        file_path = tmp_path / "empty.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(empty_transcript, f)

        preview_merge(file_path)

        captured = capsys.readouterr()
        assert "No segments found" in captured.out

    def test_preview_shows_token_count(self, sample_transcript_file, capsys):
        """Test that preview shows token count reduction."""
        preview_merge(sample_transcript_file)

        captured = capsys.readouterr()
        # Should show token count like "Original: X tokens → Merged: Y tokens"
        assert "tokens" in captured.out
        assert "→" in captured.out

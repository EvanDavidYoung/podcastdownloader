"""Tests for convert_to_traditional.py."""

import pytest
import json
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from convert_to_traditional import convert_transcript


class TestConvertTranscript:
    """Tests for convert_transcript function."""

    def test_converts_segment_text(self, simplified_transcript_file, tmp_path):
        """Test that segment text is converted to traditional."""
        output_path = tmp_path / "traditional.json"
        convert_transcript(simplified_transcript_file, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        # "这个节目很有意思" -> "這個節目很有意思"
        text = result["segments"][0]["text"]
        assert "這" in text  # 这 -> 這
        assert "個" in text  # 个 -> 個
        assert "節" in text  # 节 -> 節

    def test_converts_words(self, simplified_transcript_file, tmp_path):
        """Test that individual words are converted."""
        output_path = tmp_path / "traditional.json"
        convert_transcript(simplified_transcript_file, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        words = result["segments"][0]["words"]
        # "这个" -> "這個"
        assert words[0]["word"] == "這個"
        # "节目" -> "節目"
        assert words[1]["word"] == "節目"

    def test_preserves_timestamps(self, simplified_transcript_file, tmp_path):
        """Test that timestamps are preserved during conversion."""
        # Read original
        with open(simplified_transcript_file, "r", encoding="utf-8") as f:
            original = json.load(f)

        output_path = tmp_path / "traditional.json"
        convert_transcript(simplified_transcript_file, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        # Timestamps should be unchanged
        orig_words = original["segments"][0]["words"]
        result_words = result["segments"][0]["words"]

        for i in range(len(orig_words)):
            assert orig_words[i]["start"] == result_words[i]["start"]
            assert orig_words[i]["end"] == result_words[i]["end"]
            assert orig_words[i]["score"] == result_words[i]["score"]

    def test_default_output_name(self, simplified_transcript_file):
        """Test default output naming (_traditional suffix)."""
        result = convert_transcript(simplified_transcript_file)

        expected = simplified_transcript_file.parent / "simplified_traditional.json"
        assert result == expected
        assert expected.exists()

        # Cleanup
        expected.unlink()

    def test_taiwan_standard(self, simplified_transcript_file, tmp_path):
        """Test Taiwan standard conversion (s2tw)."""
        output_path = tmp_path / "taiwan.json"
        convert_transcript(simplified_transcript_file, output_path, config="s2tw")

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        # Should still convert to traditional
        text = result["segments"][0]["text"]
        assert "這" in text

    def test_reverse_conversion(self, tmp_path):
        """Test traditional to simplified conversion (t2s)."""
        # Create a traditional Chinese transcript
        traditional = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "這個節目很有意思",
                    "words": [
                        {"word": "這個", "start": 0.0, "end": 1.0, "score": 0.95},
                    ],
                },
            ],
            "language": "zh",
        }

        input_path = tmp_path / "traditional_input.json"
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(traditional, f, ensure_ascii=False)

        output_path = tmp_path / "simplified_output.json"
        convert_transcript(input_path, output_path, config="t2s")

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        # "這個節目很有意思" -> "这个节目很有意思"
        text = result["segments"][0]["text"]
        assert "这" in text
        assert "个" in text
        assert "节" in text


class TestConvertWordSegments:
    """Test that word_segments are also converted."""

    def test_converts_word_segments(self, tmp_path):
        """Test that top-level word_segments are converted."""
        transcript = {
            "segments": [],
            "word_segments": [
                {"word": "这个", "start": 0.0, "end": 1.0, "score": 0.95},
                {"word": "节目", "start": 1.0, "end": 2.0, "score": 0.92},
            ],
            "language": "zh",
        }

        input_path = tmp_path / "input.json"
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False)

        output_path = tmp_path / "output.json"
        convert_transcript(input_path, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        assert result["word_segments"][0]["word"] == "這個"
        assert result["word_segments"][1]["word"] == "節目"

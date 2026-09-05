"""Tests for the diarized bilingual helpers in scripts/modal/transcribe_modal.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock modal before import — the module builds an Image/App at import time and
# modal is not installed in the test env (same approach as test_app.py).
mock_modal = MagicMock()
mock_image = MagicMock()
mock_image.apt_install.return_value = mock_image
mock_image.pip_install.return_value = mock_image
mock_modal.Image.debian_slim.return_value = mock_image
mock_modal.App.return_value = MagicMock()
mock_modal.Secret.from_name.return_value = MagicMock()

sys.modules["modal"] = mock_modal
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "modal"))

from transcribe_modal import (  # noqa: E402
    build_diarization_pipeline,
    convert_to_traditional,
    merge_chinese_words,
    merge_speaker_turns,
    normalize_word_timings,
)


class TestBuildDiarizationPipeline:
    """build_diarization_pipeline picks the auth kwarg the installed whisperx accepts."""

    def test_uses_token_kwarg_on_new_signature(self):
        class NewPipeline:
            def __init__(self, token=None, device=None):
                self.kwargs = {"token": token, "device": device}

        pipe = build_diarization_pipeline(NewPipeline, "hf_abc", "cuda")
        assert pipe.kwargs == {"token": "hf_abc", "device": "cuda"}

    def test_uses_use_auth_token_kwarg_on_old_signature(self):
        class OldPipeline:
            def __init__(self, use_auth_token=None, device=None):
                self.kwargs = {"use_auth_token": use_auth_token, "device": device}

        pipe = build_diarization_pipeline(OldPipeline, "hf_abc", "cpu")
        assert pipe.kwargs == {"use_auth_token": "hf_abc", "device": "cpu"}

    def test_prefers_token_when_both_accepted(self):
        class BothPipeline:
            def __init__(self, token=None, use_auth_token=None, device=None):
                self.token = token
                self.use_auth_token = use_auth_token

        pipe = build_diarization_pipeline(BothPipeline, "hf_abc", "cpu")
        assert pipe.token == "hf_abc"
        assert pipe.use_auth_token is None


class TestMergeSpeakerTurns:
    def test_bridges_short_same_speaker_gaps(self):
        rows = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.3, "end": 4.0, "speaker": "SPEAKER_00"},
        ]
        assert merge_speaker_turns(rows, merge_gap=0.5) == [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}
        ]

    def test_never_bridges_across_speakers(self):
        rows = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.1, "end": 4.0, "speaker": "SPEAKER_01"},
        ]
        merged = merge_speaker_turns(rows, merge_gap=0.5)
        assert [t["speaker"] for t in merged] == ["SPEAKER_00", "SPEAKER_01"]

    def test_keeps_long_same_speaker_gaps_separate(self):
        rows = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_00"},
        ]
        assert len(merge_speaker_turns(rows, merge_gap=0.5)) == 2

    def test_drops_turns_below_min_turn(self):
        rows = [
            {"start": 0.0, "end": 0.2, "speaker": "SPEAKER_00"},  # 0.2s blip
            {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"},
        ]
        merged = merge_speaker_turns(rows, merge_gap=0.5, min_turn=0.4)
        assert merged == [{"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"}]

    def test_sorts_unordered_input(self):
        rows = [
            {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_01"},
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        ]
        assert [t["start"] for t in merge_speaker_turns(rows)] == [0.0, 5.0]

    def test_handles_nested_rows(self):
        # A row fully contained in the previous one must not shorten the turn.
        rows = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
        ]
        assert merge_speaker_turns(rows) == [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]

    def test_empty_input(self):
        assert merge_speaker_turns([]) == []


class TestNormalizeWordTimings:
    def test_redistributes_anomalous_first_word(self):
        # The real failure mode: alignment gives the first char the whole leading
        # silence (6.2s) while every other char gets ~0.02s.
        segment = {
            "start": 0.0,
            "end": 6.6,
            "words": [
                {"word": "请", "start": 0.258, "end": 6.479},
                {"word": "各", "start": 6.479, "end": 6.499},
                {"word": "位", "start": 6.499, "end": 6.519},
                {"word": "注", "start": 6.519, "end": 6.6},
            ],
        }
        normalize_word_timings(segment)
        words = segment["words"]
        assert words[0]["start"] == 0.0
        assert words[-1]["end"] == 6.6
        durations = [w["end"] - w["start"] for w in words]
        # Equal-length tokens get equal shares instead of one swallowing the segment.
        assert max(durations) - min(durations) < 0.01

    def test_leaves_healthy_segments_alone(self):
        segment = {
            "start": 0.0,
            "end": 1.0,
            "words": [
                {"word": "a", "start": 0.0, "end": 0.3},
                {"word": "b", "start": 0.3, "end": 0.6},
                {"word": "c", "start": 0.6, "end": 1.0},
            ],
        }
        before = [dict(w) for w in segment["words"]]
        normalize_word_timings(segment)
        assert segment["words"] == before

    def test_weights_by_character_count(self):
        segment = {
            "start": 0.0,
            "end": 3.0,
            "words": [
                {"word": "aaaa", "start": 0.0, "end": 2.9},
                {"word": "b", "start": 2.9, "end": 2.95},
                {"word": "c", "start": 2.95, "end": 3.0},
            ],
        }
        normalize_word_timings(segment)
        w = segment["words"]
        # 4:1:1 character split across 3.0s
        assert abs((w[0]["end"] - w[0]["start"]) - 2.0) < 0.01
        assert abs((w[1]["end"] - w[1]["start"]) - 0.5) < 0.01

    def test_ignores_too_short_segments(self):
        segment = {"start": 0.0, "end": 5.0, "words": [{"word": "a", "start": 0.0, "end": 4.9}]}
        normalize_word_timings(segment)
        assert segment["words"][0]["end"] == 4.9

    def test_tolerates_missing_word_timestamps(self):
        segment = {
            "start": 0.0,
            "end": 1.0,
            "words": [{"word": "a", "start": None, "end": None}, {"word": "b", "start": 0.5, "end": 1.0}],
        }
        normalize_word_timings(segment)  # must not raise


class TestPostProcessingPreservesSpeakerKeys:
    """The diarized output carries speaker/language on each segment; the existing
    zh post-processing must not strip them."""

    def _segments(self):
        return [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "我们可以",
                "speaker": "SPEAKER_01",
                "language": "zh",
                "words": [
                    {"word": "我", "start": 0.0, "end": 0.25, "score": 0.9},
                    {"word": "们", "start": 0.25, "end": 0.5, "score": 0.9},
                    {"word": "可", "start": 0.5, "end": 0.75, "score": 0.9},
                    {"word": "以", "start": 0.75, "end": 1.0, "score": 0.9},
                ],
            }
        ]

    def test_merge_chinese_words_keeps_segment_keys(self):
        data = merge_chinese_words({"segments": self._segments(), "language": "zh"})
        seg = data["segments"][0]
        assert seg["speaker"] == "SPEAKER_01"
        assert seg["language"] == "zh"
        # jieba merged the four chars into fewer, wider words
        assert len(seg["words"]) < 4
        assert seg["words"][0]["start"] == 0.0
        assert seg["words"][-1]["end"] == 1.0

    def test_convert_to_traditional_keeps_segment_keys(self):
        data = convert_to_traditional({"segments": self._segments(), "language": "zh"})
        seg = data["segments"][0]
        assert seg["speaker"] == "SPEAKER_01"
        assert seg["language"] == "zh"

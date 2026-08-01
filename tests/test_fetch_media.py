"""Tests for the yt-dlp media/subtitle helpers in scripts/modal/transcribe_modal.py."""

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
    looks_like_direct_audio,
    parse_json3_subtitles,
    parse_subtitle_data,
    parse_vtt_subtitles,
    select_subtitle_track,
)


class TestLooksLikeDirectAudio:
    def test_audio_content_type_wins_over_extension(self):
        assert looks_like_direct_audio("https://cdn.example.com/stream", "audio/mpeg")

    def test_html_page_goes_to_ytdlp(self):
        assert not looks_like_direct_audio(
            "https://example.com/ep.mp3", "text/html; charset=utf-8"
        )

    def test_video_goes_to_ytdlp_so_only_audio_is_downloaded(self):
        assert not looks_like_direct_audio("https://example.com/clip.mp4", "video/mp4")

    def test_falls_back_to_extension_without_content_type(self):
        assert looks_like_direct_audio("https://example.com/ep.m4a")
        assert not looks_like_direct_audio("https://youtube.com/watch?v=abc123")

    def test_ignores_query_string_when_reading_extension(self):
        assert looks_like_direct_audio("https://cdn.example.com/ep.mp3?token=xyz&t=1")

    def test_octet_stream_defers_to_extension(self):
        assert looks_like_direct_audio(
            "https://example.com/ep.flac", "application/octet-stream"
        )
        assert not looks_like_direct_audio(
            "https://example.com/download", "application/octet-stream"
        )


class TestParseJson3Subtitles:
    def test_word_offsets_become_word_timings(self):
        data = {
            "events": [
                {
                    "tStartMs": 1000,
                    "dDurationMs": 2000,
                    "segs": [
                        {"utf8": "hello", "tOffsetMs": 0},
                        {"utf8": " world", "tOffsetMs": 500},
                    ],
                }
            ]
        }
        result = parse_json3_subtitles(data, "en")

        assert result["language"] == "en"
        segment = result["segments"][0]
        assert segment["text"] == "hello world"
        assert segment["start"] == 1.0
        assert segment["end"] == 3.0
        assert segment["words"] == [
            {"word": "hello", "start": 1.0, "end": 1.5},
            {"word": "world", "start": 1.5, "end": 3.0},
        ]

    def test_word_segments_are_flattened_across_events(self):
        data = {
            "events": [
                {
                    "tStartMs": 0,
                    "dDurationMs": 1000,
                    "segs": [{"utf8": "one"}, {"utf8": " two", "tOffsetMs": 500}],
                },
                {
                    "tStartMs": 1000,
                    "dDurationMs": 1000,
                    "segs": [{"utf8": "three"}, {"utf8": " four", "tOffsetMs": 500}],
                },
            ]
        }
        result = parse_json3_subtitles(data)
        assert [w["word"] for w in result["word_segments"]] == ["one", "two", "three", "four"]

    def test_single_seg_events_carry_no_word_timings(self):
        """Manual subtitles are one seg per line — that line is not a word."""
        data = {
            "events": [
                {
                    "tStartMs": 1200,
                    "dDurationMs": 2160,
                    "segs": [{"utf8": "All right, so here we are"}],
                }
            ]
        }
        result = parse_json3_subtitles(data, "en")
        segment = result["segments"][0]
        assert segment["text"] == "All right, so here we are"
        assert segment["start"] == 1.2
        assert segment["words"] == []
        assert result["word_segments"] == []

    def test_skips_rolling_window_repeats_and_empty_events(self):
        data = {
            "events": [
                {"tStartMs": 0, "dDurationMs": 500, "segs": [{"utf8": "real"}]},
                {"tStartMs": 500, "aAppend": 1, "segs": [{"utf8": "real"}]},
                {"tStartMs": 600, "dDurationMs": 100},  # window definition, no segs
                {"tStartMs": 700, "dDurationMs": 100, "segs": [{"utf8": "\n"}]},
            ]
        }
        result = parse_json3_subtitles(data)
        assert [s["text"] for s in result["segments"]] == ["real"]

    def test_missing_duration_is_backfilled_from_next_segment(self):
        data = {
            "events": [
                {"tStartMs": 0, "segs": [{"utf8": "first"}, {"utf8": " one", "tOffsetMs": 200}]},
                {"tStartMs": 4000, "dDurationMs": 1000, "segs": [{"utf8": "second"}]},
            ]
        }
        result = parse_json3_subtitles(data)
        assert result["segments"][0]["end"] == 4.0
        assert result["segments"][0]["words"][-1]["end"] == 4.0

    def test_overlapping_rolling_windows_are_trimmed(self):
        """Auto-caption cues stay on screen into the next one; word times must not go backwards."""
        data = {
            "events": [
                {
                    "tStartMs": 4400,
                    "dDurationMs": 4160,  # runs to 8.56, past the next cue's start
                    "segs": [{"utf8": "sloppily"}, {"utf8": " written", "tOffsetMs": 560}],
                },
                {
                    "tStartMs": 6880,
                    "dDurationMs": 4560,
                    "segs": [{"utf8": "and"}, {"utf8": " rendered", "tOffsetMs": 240}],
                },
            ]
        }
        result = parse_json3_subtitles(data, "en")

        assert result["segments"][0]["end"] == 6.88
        words = result["word_segments"]
        assert [w["word"] for w in words] == ["sloppily", "written", "and", "rendered"]
        starts_and_ends = [t for w in words for t in (w["start"], w["end"])]
        assert starts_and_ends == sorted(starts_and_ends), starts_and_ends

    def test_empty_input_produces_empty_transcript(self):
        assert parse_json3_subtitles({}, "en") == {
            "segments": [],
            "word_segments": [],
            "language": "en",
        }


class TestParseVttSubtitles:
    def test_plain_vtt_has_segments_but_no_word_timings(self):
        vtt = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello there

00:00:04.000 --> 00:00:06.500
General Kenobi
"""
        result = parse_vtt_subtitles(vtt, "en")
        assert [s["text"] for s in result["segments"]] == ["Hello there", "General Kenobi"]
        assert result["segments"][0]["start"] == 1.0
        assert result["segments"][1]["end"] == 6.5
        assert result["word_segments"] == []

    def test_inline_timing_tags_produce_word_timings(self):
        vtt = """WEBVTT

00:00:00.030 --> 00:00:02.000
we're<00:00:00.389><c> going</c><00:00:00.629><c> to</c>
"""
        result = parse_vtt_subtitles(vtt, "en")
        words = result["segments"][0]["words"]
        assert [w["word"] for w in words] == ["we're", "going", "to"]
        assert words[0]["start"] == 0.03
        assert words[1]["start"] == 0.389
        assert words[2]["end"] == 2.0
        assert result["segments"][0]["text"] == "we're going to"

    def test_untimed_duplicate_cues_are_dropped_when_timed_cues_exist(self):
        # YouTube pairs every timed cue with a plain restatement of the same line.
        vtt = """WEBVTT

00:00:00.000 --> 00:00:02.000
we're going

00:00:02.000 --> 00:00:04.000
we're<00:00:00.389><c> going</c>
"""
        result = parse_vtt_subtitles(vtt)
        assert len(result["segments"]) == 1
        assert result["segments"][0]["start"] == 2.0

    def test_consecutive_identical_lines_are_deduplicated(self):
        vtt = """WEBVTT

00:00:01.000 --> 00:00:02.000
same line

00:00:02.000 --> 00:00:03.000
same line

00:00:03.000 --> 00:00:04.000
new line
"""
        result = parse_vtt_subtitles(vtt)
        assert [s["text"] for s in result["segments"]] == ["same line", "new line"]

    def test_srt_comma_decimals_and_index_lines_are_handled(self):
        srt = """1
00:00:01,000 --> 00:00:03,500
First cue

2
00:00:03,500 --> 00:00:05,000
Second cue
"""
        result = parse_vtt_subtitles(srt, "en")
        assert [s["text"] for s in result["segments"]] == ["First cue", "Second cue"]
        assert result["segments"][0]["end"] == 3.5

    def test_short_mm_ss_timestamps_are_supported(self):
        vtt = """WEBVTT

01:02.500 --> 01:04.000
Late cue
"""
        result = parse_vtt_subtitles(vtt)
        assert result["segments"][0]["start"] == 62.5


class TestParseSubtitleData:
    def test_dispatches_json3_from_bytes(self):
        raw = b'{"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hi"}]}]}'
        result = parse_subtitle_data(raw, "json3", "en")
        assert result["segments"][0]["text"] == "hi"

    def test_dispatches_vtt(self):
        raw = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n"
        assert parse_subtitle_data(raw, "vtt")["segments"][0]["text"] == "hi"

    def test_rejects_unknown_format(self):
        import pytest

        with pytest.raises(ValueError, match="Unsupported subtitle format"):
            parse_subtitle_data("<xml/>", "ttml")


class TestSelectSubtitleTrack:
    def test_manual_subtitles_beat_auto_captions(self):
        info = {
            "subtitles": {"en": [{"ext": "vtt", "url": "manual"}]},
            "automatic_captions": {"en": [{"ext": "json3", "url": "auto"}]},
        }
        language, kind, track = select_subtitle_track(info)
        assert (language, kind, track["url"]) == ("en", "subtitles", "manual")

    def test_json3_preferred_over_vtt_within_a_track(self):
        info = {
            "subtitles": {
                "en": [
                    {"ext": "vtt", "url": "v"},
                    {"ext": "json3", "url": "j"},
                ]
            }
        }
        _, _, track = select_subtitle_track(info)
        assert track["url"] == "j"

    def test_regional_variants_match_a_bare_language_code(self):
        info = {"subtitles": {"en-US": [{"ext": "vtt", "url": "u"}]}}
        language, _, _ = select_subtitle_track(info, languages=["en"])
        assert language == "en-US"

    def test_requested_language_wins_over_availability_order(self):
        info = {
            "subtitles": {
                "af": [{"ext": "vtt", "url": "af"}],
                "zh": [{"ext": "vtt", "url": "zh"}],
            }
        }
        language, _, _ = select_subtitle_track(info, languages=["zh"])
        assert language == "zh"

    def test_falls_back_to_the_media_language_when_none_requested(self):
        info = {
            "language": "zh",
            "subtitles": {
                "af": [{"ext": "vtt", "url": "af"}],
                "zh": [{"ext": "vtt", "url": "zh"}],
            },
        }
        language, _, _ = select_subtitle_track(info)
        assert language == "zh"

    def test_auto_captions_can_be_refused(self):
        info = {"automatic_captions": {"en": [{"ext": "json3", "url": "a"}]}}
        assert select_subtitle_track(info, allow_auto=False) is None

    def test_returns_none_when_nothing_is_available(self):
        assert select_subtitle_track({}) is None

    def test_returns_none_when_no_parsable_format_offered(self):
        info = {"subtitles": {"en": [{"ext": "ttml", "url": "t"}]}}
        assert select_subtitle_track(info) is None

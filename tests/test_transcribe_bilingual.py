"""Tests for transcribe_bilingual.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "local"))

from transcribe_bilingual import (
    parse_silence_intervals,
    speech_intervals_from_silence,
    merge_speech_intervals,
)


class TestParseSilenceIntervals:
    def test_parses_matched_pairs(self):
        stderr = (
            "[silencedetect @ 0x0] silence_start: 4.97525\n"
            "[silencedetect @ 0x0] silence_end: 5.553 | silence_duration: 0.57775\n"
            "[silencedetect @ 0x0] silence_start: 10.07825\n"
            "[silencedetect @ 0x0] silence_end: 10.699625 | silence_duration: 0.621375\n"
        )
        assert parse_silence_intervals(stderr) == [(4.97525, 5.553), (10.07825, 10.699625)]

    def test_no_silence(self):
        assert parse_silence_intervals("") == []

    def test_dangling_start_at_eof_is_dropped(self):
        stderr = (
            "[silencedetect @ 0x0] silence_start: 4.0\n"
            "[silencedetect @ 0x0] silence_end: 5.0 | silence_duration: 1.0\n"
            "[silencedetect @ 0x0] silence_start: 20.0\n"  # runs to EOF, no matching end logged
        )
        assert parse_silence_intervals(stderr) == [(4.0, 5.0)]


class TestSpeechIntervalsFromSilence:
    def test_inverts_silence_to_speech(self):
        silence = [(5.0, 6.0), (10.0, 11.0)]
        speech = speech_intervals_from_silence(silence, total_duration=15.0)
        assert speech == [(0.0, 5.0), (6.0, 10.0), (11.0, 15.0)]

    def test_drops_intervals_below_min_speech(self):
        silence = [(1.0, 1.05), (1.1, 5.0)]
        speech = speech_intervals_from_silence(silence, total_duration=5.0, min_speech=0.15)
        # (0.0, 1.0) survives, (1.05, 1.1) is a 0.05s blip and gets dropped
        assert speech == [(0.0, 1.0)]

    def test_no_silence_means_one_speech_interval(self):
        assert speech_intervals_from_silence([], total_duration=10.0) == [(0.0, 10.0)]


class TestMergeSpeechIntervals:
    def test_bridges_short_gaps(self):
        speech = [(0.0, 2.0), (2.5, 4.0), (4.3, 6.0)]
        merged = merge_speech_intervals(speech, merge_silence_max=1.0, max_chunk_sec=25.0)
        assert merged == [(0.0, 6.0)]

    def test_keeps_long_gaps_separate(self):
        speech = [(0.0, 2.0), (5.0, 7.0)]  # 3s gap, likely a real speaker/language switch
        merged = merge_speech_intervals(speech, merge_silence_max=1.2, max_chunk_sec=25.0)
        assert merged == [(0.0, 2.0), (5.0, 7.0)]

    def test_respects_max_chunk_cap_even_with_short_gaps(self):
        speech = [(0.0, 20.0), (20.5, 30.0)]
        merged = merge_speech_intervals(speech, merge_silence_max=1.0, max_chunk_sec=25.0)
        # would bridge on gap alone, but merged length would exceed the cap
        assert merged == [(0.0, 20.0), (20.5, 30.0)]

    def test_empty_input(self):
        assert merge_speech_intervals([]) == []

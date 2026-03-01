"""Tests for transcribe_local.py."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "local"))

from transcribe_local import get_audio_files, transcribe_file, AUDIO_EXTENSIONS


class TestGetAudioFiles:
    """Tests for get_audio_files function."""

    def test_returns_empty_when_directory_missing(self, tmp_path):
        """Test returns empty list when downloads directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"

        with patch("transcribe_local.DOWNLOADS_DIR", nonexistent):
            result = get_audio_files()

        assert result == []

    def test_returns_mp3_files(self, tmp_path):
        """Test that MP3 files are returned."""
        (tmp_path / "episode1.mp3").touch()
        (tmp_path / "episode2.mp3").touch()

        with patch("transcribe_local.DOWNLOADS_DIR", tmp_path):
            result = get_audio_files()

        assert len(result) == 2
        names = {f.name for f in result}
        assert "episode1.mp3" in names
        assert "episode2.mp3" in names

    def test_returns_m4a_files(self, tmp_path):
        """Test that M4A files are returned."""
        (tmp_path / "podcast.m4a").touch()

        with patch("transcribe_local.DOWNLOADS_DIR", tmp_path):
            result = get_audio_files()

        assert len(result) == 1
        assert result[0].name == "podcast.m4a"

    def test_ignores_non_audio_files(self, tmp_path):
        """Test that non-audio files are ignored."""
        (tmp_path / "episode.mp3").touch()
        (tmp_path / "transcript.json").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / ".gitkeep").touch()

        with patch("transcribe_local.DOWNLOADS_DIR", tmp_path):
            result = get_audio_files()

        assert len(result) == 1
        assert result[0].name == "episode.mp3"

    def test_case_insensitive_extension(self, tmp_path):
        """Test that file extension matching is case-insensitive."""
        (tmp_path / "lower.mp3").touch()
        (tmp_path / "upper.MP3").touch()
        (tmp_path / "mixed.Mp3").touch()

        with patch("transcribe_local.DOWNLOADS_DIR", tmp_path):
            result = get_audio_files()

        assert len(result) == 3


class TestTranscribeFile:
    """Tests for transcribe_file function."""

    def test_returns_true_on_success(self, tmp_path):
        """Test returns True when whisperx succeeds."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_result = Mock()
        mock_result.returncode = 0

        with patch("transcribe_local.subprocess.run", return_value=mock_result) as mock_run:
            result = transcribe_file(audio_file)

        assert result is True
        mock_run.assert_called_once()

    def test_returns_false_on_failure(self, tmp_path):
        """Test returns False when whisperx fails."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_result = Mock()
        mock_result.returncode = 1

        with patch("transcribe_local.subprocess.run", return_value=mock_result):
            result = transcribe_file(audio_file)

        assert result is False

    def test_calls_whisperx_with_correct_args(self, tmp_path):
        """Test that whisperx is called with the expected arguments."""
        audio_file = tmp_path / "podcast.mp3"
        audio_file.touch()

        mock_result = Mock()
        mock_result.returncode = 0

        with patch("transcribe_local.subprocess.run", return_value=mock_result) as mock_run:
            transcribe_file(audio_file)

        call_args = mock_run.call_args[0][0]

        # Check key arguments
        assert call_args[0] == "whisperx"
        assert str(audio_file) in call_args
        assert "--language" in call_args
        assert "zh" in call_args
        assert "--device" in call_args
        assert "cpu" in call_args
        assert "--compute_type" in call_args
        assert "int8" in call_args
        assert "--vad_method" in call_args
        assert "silero" in call_args
        assert "--output_dir" in call_args
        assert str(audio_file.parent) in call_args


class TestMain:
    """Tests for main function."""

    def test_exits_cleanly_when_no_files(self, tmp_path, capsys):
        """Test main exits with message when no audio files found."""
        with patch("transcribe_local.DOWNLOADS_DIR", tmp_path):
            with patch("transcribe_local.sys.exit") as mock_exit:
                from transcribe_local import main
                main()

        captured = capsys.readouterr()
        assert "No audio files found" in captured.out
        mock_exit.assert_called_once_with(0)

    def test_processes_all_files(self, tmp_path, capsys):
        """Test main processes all audio files."""
        (tmp_path / "ep1.mp3").touch()
        (tmp_path / "ep2.mp3").touch()

        mock_result = Mock()
        mock_result.returncode = 0

        with patch("transcribe_local.DOWNLOADS_DIR", tmp_path):
            with patch("transcribe_local.subprocess.run", return_value=mock_result) as mock_run:
                from transcribe_local import main
                main()

        # Should have called whisperx twice
        assert mock_run.call_count == 2

        captured = capsys.readouterr()
        assert "Found 2 audio file(s)" in captured.out
        assert "2/2 files succeeded" in captured.out

    def test_reports_failures(self, tmp_path, capsys):
        """Test main reports failed transcriptions."""
        (tmp_path / "good.mp3").touch()
        (tmp_path / "bad.mp3").touch()

        def side_effect(cmd):
            result = Mock()
            if "good.mp3" in str(cmd):
                result.returncode = 0
            else:
                result.returncode = 1
            return result

        with patch("transcribe_local.DOWNLOADS_DIR", tmp_path):
            with patch("transcribe_local.subprocess.run", side_effect=side_effect):
                from transcribe_local import main
                main()

        captured = capsys.readouterr()
        assert "1/2 files succeeded" in captured.out

"""Tests for download_podcast.py."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from download_podcast import download_podcast


class TestDownloadPodcast:
    """Tests for download_podcast function."""

    def test_no_episodes_in_feed(self, capsys):
        """Test handling of empty RSS feed."""
        mock_feed = Mock()
        mock_feed.entries = []

        with patch("download_podcast.feedparser.parse", return_value=mock_feed):
            result = download_podcast("https://example.com/rss")

        assert result is None
        captured = capsys.readouterr()
        assert "No episodes found" in captured.out

    def test_no_audio_url_in_episode(self, capsys):
        """Test handling when episode has no audio enclosure."""
        mock_episode = Mock()
        mock_episode.title = "Test Episode"
        mock_episode.links = [{"rel": "alternate", "type": "text/html", "href": "https://example.com"}]

        mock_feed = Mock()
        mock_feed.entries = [mock_episode]

        with patch("download_podcast.feedparser.parse", return_value=mock_feed):
            result = download_podcast("https://example.com/rss")

        assert result is None
        captured = capsys.readouterr()
        assert "Could not find an audio file" in captured.out

    def test_finds_audio_from_enclosure(self, tmp_path):
        """Test finding audio URL from enclosure link."""
        mock_episode = Mock()
        mock_episode.title = "Episode 123"
        mock_episode.links = [
            {"rel": "enclosure", "type": "audio/mpeg", "href": "https://example.com/audio.mp3"},
        ]

        mock_feed = Mock()
        mock_feed.entries = [mock_episode]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b"audio data chunk"])

        with patch("download_podcast.feedparser.parse", return_value=mock_feed):
            with patch("download_podcast.requests.get", return_value=mock_response) as mock_get:
                download_podcast("https://example.com/rss", save_dir=str(tmp_path))

        # Verify request was made with correct URL
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://example.com/audio.mp3"

        # Verify file was created
        files = list(tmp_path.glob("*.mp3"))
        assert len(files) == 1
        assert "Episode_123" in files[0].name

    def test_finds_audio_from_audio_mpeg_type(self, tmp_path):
        """Test finding audio URL from audio/mpeg type."""
        mock_episode = Mock()
        mock_episode.title = "My Podcast"
        mock_episode.links = [
            {"rel": "alternate", "type": "text/html", "href": "https://example.com/page"},
            {"rel": "alternate", "type": "audio/mpeg", "href": "https://cdn.example.com/episode.mp3"},
        ]

        mock_feed = Mock()
        mock_feed.entries = [mock_episode]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b"data"])

        with patch("download_podcast.feedparser.parse", return_value=mock_feed):
            with patch("download_podcast.requests.get", return_value=mock_response):
                download_podcast("https://example.com/rss", save_dir=str(tmp_path))

        files = list(tmp_path.glob("*.mp3"))
        assert len(files) == 1

    def test_failed_download(self, tmp_path, capsys):
        """Test handling of failed HTTP request."""
        mock_episode = Mock()
        mock_episode.title = "Test"
        mock_episode.links = [{"rel": "enclosure", "href": "https://example.com/audio.mp3"}]

        mock_feed = Mock()
        mock_feed.entries = [mock_episode]

        mock_response = Mock()
        mock_response.status_code = 404

        with patch("download_podcast.feedparser.parse", return_value=mock_feed):
            with patch("download_podcast.requests.get", return_value=mock_response):
                download_podcast("https://example.com/rss", save_dir=str(tmp_path))

        captured = capsys.readouterr()
        assert "Failed to download" in captured.out
        assert "404" in captured.out

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Test that download directory is created if it doesn't exist."""
        save_dir = tmp_path / "new_folder"
        assert not save_dir.exists()

        mock_episode = Mock()
        mock_episode.title = "Episode"
        mock_episode.links = [{"rel": "enclosure", "href": "https://example.com/audio.mp3"}]

        mock_feed = Mock()
        mock_feed.entries = [mock_episode]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b"data"])

        with patch("download_podcast.feedparser.parse", return_value=mock_feed):
            with patch("download_podcast.requests.get", return_value=mock_response):
                download_podcast("https://example.com/rss", save_dir=str(save_dir))

        assert save_dir.exists()

    def test_cleans_title_for_filename(self, tmp_path):
        """Test that special characters are removed from filename."""
        mock_episode = Mock()
        # Title with special characters that should be cleaned
        mock_episode.title = "Episode #1: Test/Episode (Part 1)"
        mock_episode.links = [{"rel": "enclosure", "href": "https://example.com/audio.mp3"}]

        mock_feed = Mock()
        mock_feed.entries = [mock_episode]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b"data"])

        with patch("download_podcast.feedparser.parse", return_value=mock_feed):
            with patch("download_podcast.requests.get", return_value=mock_response):
                download_podcast("https://example.com/rss", save_dir=str(tmp_path))

        files = list(tmp_path.glob("*.mp3"))
        assert len(files) == 1
        # Should not contain #, :, /, or ()
        filename = files[0].name
        assert "#" not in filename
        assert ":" not in filename
        assert "/" not in filename
        assert "(" not in filename

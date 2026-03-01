"""Tests for app.py FastAPI application."""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# We need to mock modal before importing app since modal isn't in whisperx env
# Create a comprehensive mock modal module
mock_modal = MagicMock()
mock_image = MagicMock()
mock_image.pip_install.return_value = mock_image
mock_image.add_local_dir.return_value = mock_image
mock_modal.Image.debian_slim.return_value = mock_image
mock_modal.App.return_value = MagicMock()
mock_modal.Secret.from_name.return_value = MagicMock()
mock_modal.concurrent.return_value = lambda f: f
mock_modal.asgi_app.return_value = lambda f: f

sys.modules["modal"] = mock_modal

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import specific items we want to test
from app import (
    cleanup_old_jobs,
    jobs,
    JOB_TTL_SECONDS,
    verify_api_key,
    TranscribeURLRequest,
    TranscribeRSSRequest,
    JobResponse,
    StatusResponse,
    web_app,
)
from fastapi import HTTPException


class TestCleanupOldJobs:
    """Tests for cleanup_old_jobs function."""

    def setup_method(self):
        """Clear jobs before each test."""
        jobs.clear()

    def test_removes_expired_jobs(self):
        """Test that jobs older than TTL are removed."""
        # Add an old job (older than TTL)
        old_time = time.time() - JOB_TTL_SECONDS - 100
        jobs["old_job"] = {"created_at": old_time, "status": "completed"}

        # Add a recent job
        jobs["new_job"] = {"created_at": time.time(), "status": "running"}

        cleanup_old_jobs()

        assert "old_job" not in jobs
        assert "new_job" in jobs

    def test_keeps_recent_jobs(self):
        """Test that recent jobs are not removed."""
        jobs["job1"] = {"created_at": time.time(), "status": "running"}
        jobs["job2"] = {"created_at": time.time() - 100, "status": "completed"}

        cleanup_old_jobs()

        assert "job1" in jobs
        assert "job2" in jobs

    def test_handles_empty_jobs(self):
        """Test cleanup with no jobs."""
        cleanup_old_jobs()
        assert len(jobs) == 0


class TestVerifyApiKey:
    """Tests for verify_api_key dependency."""

    def test_raises_500_when_api_key_not_configured(self):
        """Test error when API_KEY env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key("some-key")

            assert exc_info.value.status_code == 500
            assert "not configured" in exc_info.value.detail

    def test_raises_401_for_invalid_key(self):
        """Test error when wrong API key is provided."""
        with patch.dict("os.environ", {"API_KEY": "correct-key"}):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key("wrong-key")

            assert exc_info.value.status_code == 401
            assert "Invalid" in exc_info.value.detail

    def test_returns_key_when_valid(self):
        """Test success when correct API key is provided."""
        with patch.dict("os.environ", {"API_KEY": "my-secret-key"}):
            result = verify_api_key("my-secret-key")

        assert result == "my-secret-key"


class TestRequestModels:
    """Tests for Pydantic request models."""

    def test_transcribe_url_request_defaults(self):
        """Test TranscribeURLRequest default values."""
        req = TranscribeURLRequest(url="https://example.com/audio.mp3")

        assert req.url == "https://example.com/audio.mp3"
        assert req.language == "zh"
        assert req.merge_words is True
        assert req.to_traditional is False

    def test_transcribe_url_request_custom_values(self):
        """Test TranscribeURLRequest with custom values."""
        req = TranscribeURLRequest(
            url="https://example.com/audio.mp3",
            language="en",
            merge_words=False,
            to_traditional=True,
        )

        assert req.language == "en"
        assert req.merge_words is False
        assert req.to_traditional is True

    def test_transcribe_rss_request_defaults(self):
        """Test TranscribeRSSRequest default values."""
        req = TranscribeRSSRequest(rss_url="https://example.com/feed.rss")

        assert req.rss_url == "https://example.com/feed.rss"
        assert req.episode_index == 0
        assert req.language == "zh"

    def test_transcribe_rss_request_custom_episode(self):
        """Test TranscribeRSSRequest with custom episode index."""
        req = TranscribeRSSRequest(
            rss_url="https://example.com/feed.rss",
            episode_index=5,
        )

        assert req.episode_index == 5


class TestResponseModels:
    """Tests for Pydantic response models."""

    def test_job_response(self):
        """Test JobResponse model."""
        resp = JobResponse(job_id="abc123")

        assert resp.job_id == "abc123"
        assert resp.status == "pending"

    def test_job_response_with_status(self):
        """Test JobResponse with custom status."""
        resp = JobResponse(job_id="xyz", status="running")

        assert resp.status == "running"

    def test_status_response_completed(self):
        """Test StatusResponse for completed job."""
        resp = StatusResponse(
            status="completed",
            result={"segments": [], "language": "zh"},
        )

        assert resp.status == "completed"
        assert resp.result == {"segments": [], "language": "zh"}
        assert resp.error is None

    def test_status_response_error(self):
        """Test StatusResponse for failed job."""
        resp = StatusResponse(status="error", error="Transcription failed")

        assert resp.status == "error"
        assert resp.error == "Transcription failed"
        assert resp.result is None


class TestHealthEndpoint:
    """Tests for health check endpoint using TestClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        return TestClient(web_app)

    def test_health_returns_ok(self, client):
        """Test health endpoint returns OK status."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "podcast-transcriber"


class TestJobsTracking:
    """Tests for in-memory job tracking."""

    def setup_method(self):
        """Clear jobs before each test."""
        jobs.clear()

    def test_job_ttl_is_one_hour(self):
        """Test that JOB_TTL_SECONDS is 1 hour."""
        assert JOB_TTL_SECONDS == 3600

    def test_jobs_dict_stores_job_data(self):
        """Test that jobs can be stored and retrieved."""
        jobs["test-job"] = {
            "call": Mock(),
            "created_at": time.time(),
            "status": "running",
            "type": "url",
            "input": "https://example.com/audio.mp3",
        }

        assert "test-job" in jobs
        assert jobs["test-job"]["status"] == "running"
        assert jobs["test-job"]["type"] == "url"

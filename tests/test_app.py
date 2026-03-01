"""Tests for the FastAPI web application (src/app.py)."""

import io
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# ---------------------
# Mock modal before importing app — modal is not installed in the whisperx env
# ---------------------

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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import (  # noqa: E402
    JOB_TTL_SECONDS,
    JobResponse,
    StatusResponse,
    TranscribeRSSRequest,
    TranscribeURLRequest,
    _valid_api_keys,
    cleanup_old_jobs,
    jobs,
    verify_api_key,
    web_app,
)

# ---------------------
# Shared test constants
# ---------------------

API_KEY = "test-api-key"
SLACK_KEY = "test-slack-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


# ---------------------
# Shared fixtures
# ---------------------


@pytest.fixture(autouse=True)
def clear_jobs():
    """Wipe the in-memory jobs dict before and after every test."""
    jobs.clear()
    yield
    jobs.clear()


@pytest.fixture
def client():
    """TestClient with API_KEY configured in the environment."""
    with patch.dict("os.environ", {"API_KEY": API_KEY}):
        yield TestClient(web_app)


@pytest.fixture
def mock_fn():
    """
    A mock Modal function whose .spawn() returns a call with a fixed object_id.
    Saves and restores the global modal mock so tests don't bleed into each other.
    """
    call = Mock()
    call.object_id = "job-abc123"
    fn = Mock()
    fn.spawn.return_value = call

    original = mock_modal.Function.from_name.return_value
    mock_modal.Function.from_name.return_value = fn
    yield fn, call
    mock_modal.Function.from_name.return_value = original


# =====================
# Unit tests
# =====================


class TestValidApiKeys:
    """_valid_api_keys() collects valid keys from environment variables."""

    def test_empty_when_no_env_vars_set(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _valid_api_keys() == set()

    def test_returns_api_key(self):
        with patch.dict("os.environ", {"API_KEY": "k1"}, clear=True):
            assert _valid_api_keys() == {"k1"}

    def test_returns_slack_bot_key(self):
        with patch.dict("os.environ", {"SLACK_BOT_API_KEY": "s1"}, clear=True):
            assert _valid_api_keys() == {"s1"}

    def test_returns_both_keys_when_both_configured(self):
        with patch.dict("os.environ", {"API_KEY": "k1", "SLACK_BOT_API_KEY": "s1"}, clear=True):
            assert _valid_api_keys() == {"k1", "s1"}


class TestVerifyApiKey:
    """verify_api_key() enforces Authorization: Bearer authentication."""

    def test_raises_500_when_no_keys_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc:
                verify_api_key("Bearer anything")
            assert exc.value.status_code == 500

    def test_raises_401_when_bearer_prefix_missing(self):
        with patch.dict("os.environ", {"API_KEY": API_KEY}):
            with pytest.raises(HTTPException) as exc:
                verify_api_key(API_KEY)  # raw key, no "Bearer " prefix
            assert exc.value.status_code == 401
            assert "format" in exc.value.detail.lower()

    def test_raises_401_for_wrong_key(self):
        with patch.dict("os.environ", {"API_KEY": API_KEY}):
            with pytest.raises(HTTPException) as exc:
                verify_api_key("Bearer wrong-key")
            assert exc.value.status_code == 401

    def test_accepts_valid_api_key(self):
        with patch.dict("os.environ", {"API_KEY": API_KEY}):
            result = verify_api_key(f"Bearer {API_KEY}")
        assert result == f"Bearer {API_KEY}"

    def test_accepts_slack_bot_key(self):
        with patch.dict("os.environ", {"SLACK_BOT_API_KEY": SLACK_KEY}):
            result = verify_api_key(f"Bearer {SLACK_KEY}")
        assert result == f"Bearer {SLACK_KEY}"

    def test_accepts_either_key_when_both_configured(self):
        with patch.dict("os.environ", {"API_KEY": API_KEY, "SLACK_BOT_API_KEY": SLACK_KEY}):
            verify_api_key(f"Bearer {API_KEY}")
            verify_api_key(f"Bearer {SLACK_KEY}")


class TestCleanupOldJobs:
    """cleanup_old_jobs() removes expired entries from the jobs dict."""

    def test_removes_jobs_older_than_ttl(self):
        jobs["old"] = {"created_at": time.time() - JOB_TTL_SECONDS - 1, "status": "completed"}
        jobs["new"] = {"created_at": time.time(), "status": "running"}
        cleanup_old_jobs()
        assert "old" not in jobs
        assert "new" in jobs

    def test_keeps_jobs_within_ttl(self):
        jobs["j1"] = {"created_at": time.time(), "status": "running"}
        jobs["j2"] = {"created_at": time.time() - 60, "status": "completed"}
        cleanup_old_jobs()
        assert "j1" in jobs
        assert "j2" in jobs

    def test_handles_empty_jobs_dict(self):
        cleanup_old_jobs()
        assert len(jobs) == 0

    def test_ttl_is_one_hour(self):
        assert JOB_TTL_SECONDS == 3600


class TestRequestModels:
    """Pydantic request models have correct fields and defaults."""

    def test_transcribe_url_defaults(self):
        req = TranscribeURLRequest(url="https://example.com/ep.mp3")
        assert req.language == "zh"
        assert req.merge_words is True
        assert req.to_traditional is False

    def test_transcribe_url_custom_values(self):
        req = TranscribeURLRequest(
            url="https://example.com/ep.mp3",
            language="en",
            merge_words=False,
            to_traditional=True,
        )
        assert req.language == "en"
        assert req.merge_words is False
        assert req.to_traditional is True

    def test_transcribe_rss_defaults(self):
        req = TranscribeRSSRequest(rss_url="https://example.com/feed.xml")
        assert req.episode_index == 0
        assert req.language == "zh"

    def test_transcribe_rss_custom_episode(self):
        req = TranscribeRSSRequest(rss_url="https://example.com/feed.xml", episode_index=3)
        assert req.episode_index == 3


class TestResponseModels:
    """Pydantic response models have correct fields and defaults."""

    def test_job_response_defaults_to_pending(self):
        resp = JobResponse(job_id="abc")
        assert resp.job_id == "abc"
        assert resp.status == "pending"

    def test_job_response_custom_status(self):
        resp = JobResponse(job_id="abc", status="running")
        assert resp.status == "running"

    def test_status_response_completed(self):
        resp = StatusResponse(status="completed")
        assert resp.status == "completed"
        assert resp.error is None

    def test_status_response_error_includes_message(self):
        resp = StatusResponse(status="error", error="Something went wrong")
        assert resp.status == "error"
        assert resp.error == "Something went wrong"


# =====================
# E2E endpoint tests
# =====================


class TestHealthEndpoint:
    """GET /api/health — public endpoint, no authentication required."""

    def test_returns_ok_without_auth(self):
        """Health check is intentionally open so monitoring tools can reach it."""
        resp = TestClient(web_app).get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "service": "podcast-transcriber"}

    def test_still_returns_ok_when_auth_header_present(self, client):
        resp = client.get("/api/health", headers=AUTH)
        assert resp.status_code == 200


class TestAuthentication:
    """All protected endpoints enforce Authorization: Bearer authentication."""

    @pytest.fixture
    def c(self):
        with patch.dict("os.environ", {"API_KEY": API_KEY}):
            yield TestClient(web_app, raise_server_exceptions=False)

    # -- Header missing --

    def test_missing_auth_header_returns_422(self, c):
        """FastAPI returns 422 Unprocessable Entity when the required header is absent."""
        assert c.get("/api/jobs").status_code == 422
        assert c.get("/api/status/x").status_code == 422
        assert c.get("/api/result/x").status_code == 422

    # -- Wrong format / wrong key --

    def test_raw_key_without_bearer_prefix_returns_401(self, c):
        """Sending the raw key without the 'Bearer ' prefix is rejected."""
        bad = {"Authorization": API_KEY}
        assert c.get("/api/jobs", headers=bad).status_code == 401
        assert c.get("/api/status/x", headers=bad).status_code == 401

    def test_wrong_key_returns_401(self, c):
        bad = {"Authorization": "Bearer totally-wrong"}
        assert c.get("/api/jobs", headers=bad).status_code == 401

    # -- POST endpoints with valid bodies --

    def test_transcribe_url_rejects_wrong_key(self, c):
        resp = c.post(
            "/api/transcribe/url",
            json={"url": "https://example.com/ep.mp3"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    def test_transcribe_rss_rejects_wrong_key(self, c):
        resp = c.post(
            "/api/transcribe/rss",
            json={"rss_url": "https://example.com/feed.xml"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    def test_openai_endpoint_rejects_wrong_key(self, c):
        resp = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("ep.mp3", io.BytesIO(b"data"), "audio/mpeg")},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    # -- Server misconfiguration --

    def test_no_keys_configured_returns_500(self):
        """When no API keys are set in the environment the server returns 500."""
        with patch.dict("os.environ", {}, clear=True):
            c = TestClient(web_app, raise_server_exceptions=False)
            assert c.get("/api/jobs", headers=AUTH).status_code == 500

    # -- Alternative key --

    def test_slack_bot_key_accepted_on_all_get_endpoints(self):
        """SLACK_BOT_API_KEY is a valid credential on all protected routes."""
        slack_auth = {"Authorization": f"Bearer {SLACK_KEY}"}
        with patch.dict("os.environ", {"SLACK_BOT_API_KEY": SLACK_KEY}):
            c = TestClient(web_app, raise_server_exceptions=False)
            assert c.get("/api/jobs", headers=slack_auth).status_code == 200
            assert c.get("/api/status/x", headers=slack_auth).status_code == 404  # job not found, but auth passed


class TestTranscribeURLEndpoint:
    """POST /api/transcribe/url — kick off transcription from a direct audio URL."""

    def test_returns_job_id_with_running_status(self, client, mock_fn):
        """A valid request spawns a Modal job and immediately returns its ID."""
        resp = client.post(
            "/api/transcribe/url",
            json={"url": "https://example.com/ep.mp3"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-abc123"
        assert data["status"] == "running"

    def test_job_stored_in_memory(self, client, mock_fn):
        """After a successful request the job is tracked in the jobs dict."""
        client.post(
            "/api/transcribe/url",
            json={"url": "https://example.com/ep.mp3"},
            headers=AUTH,
        )
        assert "job-abc123" in jobs
        assert jobs["job-abc123"]["type"] == "url"
        assert jobs["job-abc123"]["input"] == "https://example.com/ep.mp3"
        assert jobs["job-abc123"]["status"] == "running"

    def test_all_params_forwarded_to_modal(self, client, mock_fn):
        """language, merge_words, and to_traditional are passed through to the Modal function."""
        fn, _ = mock_fn
        client.post(
            "/api/transcribe/url",
            json={
                "url": "https://example.com/ep.mp3",
                "language": "en",
                "merge_words": False,
                "to_traditional": True,
            },
            headers=AUTH,
        )
        fn.spawn.assert_called_once_with(
            url="https://example.com/ep.mp3",
            language="en",
            merge_words=False,
            to_traditional=True,
        )

    def test_defaults_to_chinese_with_merge_words(self, client, mock_fn):
        """Omitting optional fields uses Chinese-optimised defaults."""
        fn, _ = mock_fn
        client.post(
            "/api/transcribe/url",
            json={"url": "https://example.com/ep.mp3"},
            headers=AUTH,
        )
        fn.spawn.assert_called_once_with(
            url="https://example.com/ep.mp3",
            language="zh",
            merge_words=True,
            to_traditional=False,
        )


class TestTranscribeRSSEndpoint:
    """POST /api/transcribe/rss — kick off transcription from an RSS feed."""

    def test_returns_job_id_with_running_status(self, client, mock_fn):
        resp = client.post(
            "/api/transcribe/rss",
            json={"rss_url": "https://example.com/feed.xml"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-abc123"
        assert resp.json()["status"] == "running"

    def test_job_stored_with_rss_type(self, client, mock_fn):
        client.post(
            "/api/transcribe/rss",
            json={"rss_url": "https://example.com/feed.xml"},
            headers=AUTH,
        )
        assert jobs["job-abc123"]["type"] == "rss"
        assert jobs["job-abc123"]["input"] == "https://example.com/feed.xml"

    def test_episode_index_forwarded_to_modal(self, client, mock_fn):
        """Non-default episode_index is passed through to the Modal function."""
        fn, _ = mock_fn
        client.post(
            "/api/transcribe/rss",
            json={"rss_url": "https://example.com/feed.xml", "episode_index": 5},
            headers=AUTH,
        )
        fn.spawn.assert_called_once_with(
            rss_url="https://example.com/feed.xml",
            episode_index=5,
            language="zh",
            merge_words=True,
            to_traditional=False,
        )


class TestStatusEndpoint:
    """GET /api/status/{job_id} — poll a transcription job's current state."""

    def test_404_for_unknown_job(self, client):
        resp = client.get("/api/status/does-not-exist", headers=AUTH)
        assert resp.status_code == 404

    def test_running_while_modal_call_pending(self, client):
        """Returns 'running' when the Modal call raises TimeoutError (not yet done)."""
        call = Mock()
        call.get.side_effect = TimeoutError()
        jobs["j1"] = {"call": call, "created_at": time.time(), "status": "running"}

        resp = client.get("/api/status/j1", headers=AUTH)

        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_completed_when_modal_call_resolves(self, client):
        """Returns 'completed' and caches the result once the Modal call returns data."""
        call = Mock()
        call.get.return_value = {"segments": [], "language": "zh"}
        jobs["j2"] = {"call": call, "created_at": time.time(), "status": "running"}

        resp = client.get("/api/status/j2", headers=AUTH)

        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        assert jobs["j2"]["status"] == "completed"
        assert jobs["j2"]["result"] == {"segments": [], "language": "zh"}

    def test_error_when_modal_call_raises(self, client):
        """Returns 'error' with the exception message when the Modal call fails."""
        call = Mock()
        call.get.side_effect = RuntimeError("GPU out of memory")
        jobs["j3"] = {"call": call, "created_at": time.time(), "status": "running"}

        resp = client.get("/api/status/j3", headers=AUTH)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "GPU out of memory" in data["error"]
        assert jobs["j3"]["status"] == "error"

    def test_cached_completed_status_does_not_requery_modal(self, client):
        """Already-completed jobs return cached status without calling Modal again."""
        call = Mock()
        jobs["j4"] = {
            "call": call,
            "created_at": time.time(),
            "status": "completed",
            "result": {"segments": []},
        }

        resp = client.get("/api/status/j4", headers=AUTH)

        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        call.get.assert_not_called()

    def test_cached_error_status_does_not_requery_modal(self, client):
        """Already-failed jobs return cached error without calling Modal again."""
        call = Mock()
        jobs["j5"] = {
            "call": call,
            "created_at": time.time(),
            "status": "error",
            "error": "Something broke",
        }

        resp = client.get("/api/status/j5", headers=AUTH)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["error"] == "Something broke"
        call.get.assert_not_called()


class TestResultEndpoint:
    """GET /api/result/{job_id} — download the full transcript JSON for a completed job."""

    def test_404_for_unknown_job(self, client):
        resp = client.get("/api/result/does-not-exist", headers=AUTH)
        assert resp.status_code == 404

    def test_202_while_job_still_running(self, client):
        """Returns 202 Accepted when the job hasn't finished yet."""
        jobs["j1"] = {"call": Mock(), "created_at": time.time(), "status": "running"}
        resp = client.get("/api/result/j1", headers=AUTH)
        assert resp.status_code == 202

    def test_400_when_job_errored(self, client):
        """Returns 400 Bad Request with the error detail when the job failed."""
        jobs["j2"] = {
            "call": Mock(),
            "created_at": time.time(),
            "status": "error",
            "error": "Transcription failed",
        }
        resp = client.get("/api/result/j2", headers=AUTH)
        assert resp.status_code == 400
        assert "Transcription failed" in resp.json()["detail"]

    def test_returns_transcript_json_for_completed_job(self, client):
        """Returns the full transcript as a downloadable JSON response."""
        transcript = {"segments": [{"text": "你好", "start": 0.0, "end": 1.0}], "language": "zh"}
        jobs["j3"] = {
            "call": Mock(),
            "created_at": time.time(),
            "status": "completed",
            "result": transcript,
            "input": "https://example.com/episode.mp3",
        }

        resp = client.get("/api/result/j3", headers=AUTH)

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        assert "attachment" in resp.headers["content-disposition"]
        assert resp.json() == transcript

    def test_filename_derived_from_input_url(self, client):
        """The Content-Disposition filename is based on the original audio file stem."""
        jobs["j4"] = {
            "call": Mock(),
            "created_at": time.time(),
            "status": "completed",
            "result": {"segments": []},
            "input": "https://example.com/my-episode.mp3",
        }

        resp = client.get("/api/result/j4", headers=AUTH)

        assert "my-episode" in resp.headers["content-disposition"]


class TestJobsEndpoint:
    """GET /api/jobs — list all currently tracked jobs."""

    def test_empty_list_when_no_jobs(self, client):
        resp = client.get("/api/jobs", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json() == {"jobs": []}

    def test_lists_all_tracked_jobs(self, client):
        jobs["ja"] = {
            "call": Mock(), "created_at": time.time(),
            "status": "running", "type": "url", "input": "https://example.com/a.mp3",
        }
        jobs["jb"] = {
            "call": Mock(), "created_at": time.time(),
            "status": "completed", "type": "rss", "input": "https://example.com/feed.xml",
        }

        resp = client.get("/api/jobs", headers=AUTH)

        assert resp.status_code == 200
        by_id = {j["job_id"]: j for j in resp.json()["jobs"]}
        assert by_id["ja"]["status"] == "running"
        assert by_id["ja"]["type"] == "url"
        assert by_id["jb"]["status"] == "completed"
        assert by_id["jb"]["type"] == "rss"

    def test_expired_jobs_excluded_from_listing(self, client):
        """Jobs older than the TTL are purged before the list is returned."""
        jobs["old"] = {
            "call": Mock(),
            "created_at": time.time() - JOB_TTL_SECONDS - 1,
            "status": "completed",
        }

        resp = client.get("/api/jobs", headers=AUTH)

        job_ids = [j["job_id"] for j in resp.json()["jobs"]]
        assert "old" not in job_ids

    def test_age_seconds_reflects_job_age(self, client):
        """age_seconds in the response reflects how long ago the job was created."""
        jobs["j1"] = {
            "call": Mock(), "created_at": time.time() - 30,
            "status": "running", "type": "url", "input": "https://example.com/ep.mp3",
        }

        resp = client.get("/api/jobs", headers=AUTH)

        age = resp.json()["jobs"][0]["age_seconds"]
        assert age >= 30


class TestOpenAITranscribeEndpoint:
    """POST /v1/audio/transcriptions — OpenAI Whisper-compatible audio upload."""

    def _audio_file(self, content=b"fake-audio", name="episode.mp3"):
        return {"file": (name, io.BytesIO(content), "audio/mpeg")}

    def test_returns_segments_and_language(self, client, mock_fn):
        """A valid upload returns a JSON body with segments and detected language."""
        _, call = mock_fn
        call.get.return_value = {
            "segments": [{"text": "你好", "start": 0.0, "end": 1.0}],
            "language": "zh",
        }

        resp = client.post(
            "/v1/audio/transcriptions",
            files=self._audio_file(),
            data={"language": "zh"},
            headers=AUTH,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "segments" in data
        assert data["language"] == "zh"

    def test_language_forwarded_to_modal(self, client, mock_fn):
        """The language form field is passed through to the Modal transcription function."""
        fn, call = mock_fn
        call.get.return_value = {"segments": [], "language": "en"}

        client.post(
            "/v1/audio/transcriptions",
            files=self._audio_file(),
            data={"language": "en"},
            headers=AUTH,
        )

        assert fn.spawn.call_args.kwargs["language"] == "en"

    def test_audio_bytes_and_filename_forwarded_to_modal(self, client, mock_fn):
        """The raw audio bytes and filename are passed to the Modal function."""
        fn, call = mock_fn
        call.get.return_value = {"segments": [], "language": "zh"}

        client.post(
            "/v1/audio/transcriptions",
            files=self._audio_file(content=b"my-audio-data", name="ep.mp3"),
            headers=AUTH,
        )

        assert fn.spawn.call_args.kwargs["audio_bytes"] == b"my-audio-data"
        assert fn.spawn.call_args.kwargs["filename"] == "ep.mp3"

    def test_hf_token_included_when_diarize_true(self, client, mock_fn):
        """When diarize=True, the HF_TOKEN from the environment is forwarded."""
        fn, call = mock_fn
        call.get.return_value = {"segments": [], "language": "zh"}

        with patch.dict("os.environ", {"HF_TOKEN": "hf-secret"}):
            client.post(
                "/v1/audio/transcriptions",
                files=self._audio_file(),
                data={"diarize": "true"},
                headers=AUTH,
            )

        assert fn.spawn.call_args.kwargs["hf_token"] == "hf-secret"

    def test_hf_token_is_none_when_diarize_false(self, client, mock_fn):
        """When diarize=False, hf_token is passed as None regardless of HF_TOKEN env."""
        fn, call = mock_fn
        call.get.return_value = {"segments": [], "language": "zh"}

        with patch.dict("os.environ", {"HF_TOKEN": "hf-secret"}):
            client.post(
                "/v1/audio/transcriptions",
                files=self._audio_file(),
                data={"diarize": "false"},
                headers=AUTH,
            )

        assert fn.spawn.call_args.kwargs["hf_token"] is None

    def test_slack_bot_key_accepted(self, mock_fn):
        """SLACK_BOT_API_KEY is a valid credential for this endpoint."""
        _, call = mock_fn
        call.get.return_value = {"segments": [], "language": "zh"}

        with patch.dict("os.environ", {"SLACK_BOT_API_KEY": SLACK_KEY}):
            resp = TestClient(web_app).post(
                "/v1/audio/transcriptions",
                files=self._audio_file(),
                headers={"Authorization": f"Bearer {SLACK_KEY}"},
            )

        assert resp.status_code == 200

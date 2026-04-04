"""Tests for the FastAPI web application (src/app.py)."""

import io
import json
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

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
from fastapi.security import HTTPAuthorizationCredentials  # noqa: E402
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
    """TestClient with FASTAPI_APIKEY configured in the environment."""
    with patch.dict("os.environ", {"FASTAPI_APIKEY": API_KEY}):
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
        with patch.dict("os.environ", {"FASTAPI_APIKEY": "k1"}, clear=True):
            assert _valid_api_keys() == {"k1"}

    def test_returns_slack_bot_key(self):
        with patch.dict("os.environ", {"SLACK_BOT_API_KEY": "s1"}, clear=True):
            assert _valid_api_keys() == {"s1"}

    def test_returns_both_keys_when_both_configured(self):
        with patch.dict("os.environ", {"FASTAPI_APIKEY": "k1", "SLACK_BOT_API_KEY": "s1"}, clear=True):
            assert _valid_api_keys() == {"k1", "s1"}


class TestVerifyApiKey:
    """verify_api_key() enforces Authorization: Bearer authentication."""

    def _creds(self, token: str) -> HTTPAuthorizationCredentials:
        return HTTPAuthorizationCredentials(scheme="bearer", credentials=token)

    def test_raises_500_when_no_keys_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc:
                verify_api_key(self._creds("anything"))
            assert exc.value.status_code == 500

    def test_raises_401_for_wrong_key(self):
        with patch.dict("os.environ", {"FASTAPI_APIKEY": API_KEY}):
            with pytest.raises(HTTPException) as exc:
                verify_api_key(self._creds("wrong-key"))
            assert exc.value.status_code == 401

    def test_accepts_valid_api_key(self):
        with patch.dict("os.environ", {"FASTAPI_APIKEY": API_KEY}):
            result = verify_api_key(self._creds(API_KEY))
        assert result == API_KEY

    def test_accepts_slack_bot_key(self):
        with patch.dict("os.environ", {"SLACK_BOT_API_KEY": SLACK_KEY}):
            result = verify_api_key(self._creds(SLACK_KEY))
        assert result == SLACK_KEY

    def test_accepts_either_key_when_both_configured(self):
        with patch.dict("os.environ", {"FASTAPI_APIKEY": API_KEY, "SLACK_BOT_API_KEY": SLACK_KEY}):
            verify_api_key(self._creds(API_KEY))
            verify_api_key(self._creds(SLACK_KEY))


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
        with patch.dict("os.environ", {"FASTAPI_APIKEY": API_KEY}):
            yield TestClient(web_app, raise_server_exceptions=False)

    # -- Header missing --

    def test_missing_auth_header_returns_4xx(self, c):
        """HTTPBearer returns 4xx when the Authorization header is absent."""
        assert c.get("/api/jobs").status_code in (401, 403)
        assert c.get("/api/status/x").status_code in (401, 403)
        assert c.get("/api/result/x").status_code in (401, 403)

    # -- Wrong format / wrong key --

    def test_raw_key_without_bearer_prefix_returns_4xx(self, c):
        """HTTPBearer returns 4xx when the Authorization header isn't in Bearer format."""
        bad = {"Authorization": API_KEY}
        assert c.get("/api/jobs", headers=bad).status_code in (401, 403)
        assert c.get("/api/status/x", headers=bad).status_code in (401, 403)

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
        """A valid request spawns a Modal job and immediately returns a UUID job ID."""
        resp = client.post(
            "/api/transcribe/url",
            json={"url": "https://example.com/ep.mp3"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        # job_id is a pre-generated UUID, not Modal's call.object_id
        assert uuid.UUID(data["job_id"])  # valid UUID
        assert data["status"] == "running"

    def test_job_stored_in_memory(self, client, mock_fn):
        """After a successful request the job is tracked in the jobs dict."""
        resp = client.post(
            "/api/transcribe/url",
            json={"url": "https://example.com/ep.mp3"},
            headers=AUTH,
        )
        job_id = resp.json()["job_id"]
        assert job_id in jobs
        assert jobs[job_id]["type"] == "url"
        assert jobs[job_id]["input"] == "https://example.com/ep.mp3"
        assert jobs[job_id]["status"] == "running"

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
            job_id=ANY,
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
            job_id=ANY,
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
        assert uuid.UUID(resp.json()["job_id"])  # valid UUID
        assert resp.json()["status"] == "running"

    def test_job_stored_with_rss_type(self, client, mock_fn):
        resp = client.post(
            "/api/transcribe/rss",
            json={"rss_url": "https://example.com/feed.xml"},
            headers=AUTH,
        )
        job_id = resp.json()["job_id"]
        assert jobs[job_id]["type"] == "rss"
        assert jobs[job_id]["input"] == "https://example.com/feed.xml"

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
            job_id=ANY,
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


class TestPlayerJobsEndpoint:
    """GET /api/player/jobs — list jobs from persistent volume (no auth)."""

    def test_returns_empty_list_when_jobs_dir_missing(self):
        """Returns an empty list when the /jobs directory doesn't exist."""
        with patch("app.jobs_volume"):
            with patch("app.Path") as mock_path_cls:
                mock_jobs_dir = Mock()
                mock_jobs_dir.exists.return_value = False
                mock_path_cls.return_value = mock_jobs_dir
                resp = TestClient(web_app).get("/api/player/jobs")
        assert resp.status_code == 200
        assert resp.json() == {"jobs": []}

    def test_no_auth_required(self):
        """Player listing endpoint is public (no Authorization header needed)."""
        with patch("app.jobs_volume"):
            with patch("app.Path") as mock_path_cls:
                mock_jobs_dir = Mock()
                mock_jobs_dir.exists.return_value = False
                mock_path_cls.return_value = mock_jobs_dir
                resp = TestClient(web_app).get("/api/player/jobs")
        # 200 (not 403/401) confirms no auth requirement
        assert resp.status_code == 200

    def test_returns_sorted_jobs_from_volume(self, tmp_path):
        """Returns jobs from metadata.json files, sorted newest first."""
        job1_dir = tmp_path / "job-1"
        job1_dir.mkdir()
        meta1 = {"job_id": "job-1", "title": "Older Episode", "created_at": "2024-01-01T00:00:00+00:00", "type": "url", "language": "zh"}
        (job1_dir / "metadata.json").write_text(json.dumps(meta1))

        job2_dir = tmp_path / "job-2"
        job2_dir.mkdir()
        meta2 = {"job_id": "job-2", "title": "Newer Episode", "created_at": "2024-06-01T00:00:00+00:00", "type": "rss", "language": "zh"}
        (job2_dir / "metadata.json").write_text(json.dumps(meta2))

        with patch("app.jobs_volume"):
            with patch("app.Path", return_value=tmp_path):
                resp = TestClient(web_app).get("/api/player/jobs")

        assert resp.status_code == 200
        jobs_list = resp.json()["jobs"]
        assert len(jobs_list) == 2
        # Newer episode should be first
        assert jobs_list[0]["job_id"] == "job-2"
        assert jobs_list[1]["job_id"] == "job-1"


class TestPlayerTranscriptEndpoint:
    """GET /api/player/transcript/{job_id} — serve transcript JSON (no auth)."""

    def test_404_when_transcript_missing(self):
        """Returns 404 when the transcript file doesn't exist in the volume."""
        with patch("app.jobs_volume"):
            resp = TestClient(web_app).get("/api/player/transcript/nonexistent-id")
        assert resp.status_code == 404

    def test_no_auth_required(self):
        """Transcript endpoint is public."""
        with patch("app.jobs_volume"):
            resp = TestClient(web_app).get("/api/player/transcript/some-id")
        assert resp.status_code == 404  # 404 not found, not 403 forbidden

    def test_returns_transcript_json(self, tmp_path):
        """Returns the transcript JSON content when the file exists."""
        transcript = {"segments": [{"text": "你好", "start": 0.0, "end": 1.0}], "language": "zh"}
        job_dir = tmp_path / "my-job-id"
        job_dir.mkdir()
        (job_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False))

        def fake_path(path_str):
            # Map /jobs/my-job-id/transcript.json → tmp_path / my-job-id / transcript.json
            return tmp_path / "/".join(str(path_str).split("/")[2:])

        with patch("app.jobs_volume"):
            with patch("app.Path", side_effect=fake_path):
                resp = TestClient(web_app).get("/api/player/transcript/my-job-id")

        assert resp.status_code == 200
        assert resp.json() == transcript


class TestPlayerAudioEndpoint:
    """GET /api/player/audio/{job_id} — stream audio (no auth)."""

    def test_404_when_audio_missing(self):
        """Returns 404 when the audio file doesn't exist in the volume."""
        with patch("app.jobs_volume"):
            resp = TestClient(web_app).get("/api/player/audio/nonexistent-id")
        assert resp.status_code == 404

    def test_no_auth_required(self):
        """Audio endpoint is public."""
        with patch("app.jobs_volume"):
            resp = TestClient(web_app).get("/api/player/audio/some-id")
        assert resp.status_code == 404  # 404 not found, not 403 forbidden

    def test_returns_audio_bytes_with_correct_content_type(self, tmp_path):
        """Streams audio bytes with audio/mpeg content type."""
        job_dir = tmp_path / "my-job-id"
        job_dir.mkdir()
        audio_bytes = b"fake-mp3-data"
        (job_dir / "audio.mp3").write_bytes(audio_bytes)

        def fake_path(path_str):
            return tmp_path / "/".join(str(path_str).split("/")[2:])

        with patch("app.jobs_volume"):
            with patch("app.Path", side_effect=fake_path):
                resp = TestClient(web_app).get("/api/player/audio/my-job-id")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"
        assert resp.content == audio_bytes


class TestPlayerHTMLRoutes:
    """GET /player and /player/{job_id} — serve HTML pages (no auth)."""

    def test_player_listing_no_auth_required(self):
        """/player is accessible without auth (returns 200 or error for missing file, never 403)."""
        resp = TestClient(web_app, raise_server_exceptions=False).get("/player")
        assert resp.status_code != 403

    def test_player_detail_no_auth_required(self):
        """/player/{job_id} is accessible without auth."""
        resp = TestClient(web_app, raise_server_exceptions=False).get("/player/some-uuid")
        assert resp.status_code != 403


class TestWebhookCallback:
    """callback_url — fire a POST when the job finishes."""

    def test_callback_url_stored_in_job(self, client, mock_fn):
        """callback_url is persisted in the in-memory job entry."""
        resp = client.post(
            "/api/transcribe/url",
            json={"url": "https://example.com/ep.mp3", "callback_url": "https://hook.example.com/done"},
            headers=AUTH,
        )
        job_id = resp.json()["job_id"]
        assert jobs[job_id]["callback_url"] == "https://hook.example.com/done"

    def test_no_callback_url_stored_as_none(self, client, mock_fn):
        """Omitting callback_url stores None (no webhook will fire)."""
        resp = client.post(
            "/api/transcribe/url",
            json={"url": "https://example.com/ep.mp3"},
            headers=AUTH,
        )
        job_id = resp.json()["job_id"]
        assert jobs[job_id]["callback_url"] is None

    def test_rss_callback_url_stored_in_job(self, client, mock_fn):
        """callback_url also works for the RSS endpoint."""
        resp = client.post(
            "/api/transcribe/rss",
            json={"rss_url": "https://example.com/feed.xml", "callback_url": "https://hook.example.com/done"},
            headers=AUTH,
        )
        job_id = resp.json()["job_id"]
        assert jobs[job_id]["callback_url"] == "https://hook.example.com/done"


class TestWatchAndCallback:
    """_watch_and_callback() — unit tests for the webhook delivery coroutine."""

    @pytest.mark.asyncio
    async def test_posts_completed_payload_on_success(self):
        """On a successful job, POSTs status=completed with the result."""
        import httpx
        from app import _watch_and_callback

        result = {"segments": [], "language": "zh"}
        mock_call = Mock()
        mock_call.get.return_value = result

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "running"}

        posted = []

        async def fake_post(url, json=None, **kwargs):
            posted.append((url, json))
            return Mock(status_code=200)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = fake_post

        with patch("app.asyncio.to_thread", new=AsyncMock(return_value=result)):
            with patch("httpx.AsyncClient", return_value=mock_client):
                await _watch_and_callback(job_id, mock_call, "https://hook.example.com/done")

        assert jobs[job_id]["status"] == "completed"
        assert len(posted) == 1
        url, payload = posted[0]
        assert url == "https://hook.example.com/done"
        assert payload["job_id"] == job_id
        assert payload["status"] == "completed"
        assert payload["result"] == result

    @pytest.mark.asyncio
    async def test_posts_error_payload_on_failure(self):
        """On a failed job, POSTs status=error with the error message."""
        from app import _watch_and_callback

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "running"}

        posted = []

        async def fake_post(url, json=None, **kwargs):
            posted.append((url, json))
            return Mock(status_code=200)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = fake_post

        with patch("app.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("GPU OOM"))):
            with patch("httpx.AsyncClient", return_value=mock_client):
                await _watch_and_callback(job_id, Mock(), "https://hook.example.com/done")

        assert jobs[job_id]["status"] == "error"
        assert jobs[job_id]["error"] == "GPU OOM"
        url, payload = posted[0]
        assert payload["status"] == "error"
        assert payload["error"] == "GPU OOM"

    @pytest.mark.asyncio
    async def test_webhook_delivery_failure_does_not_raise(self):
        """A network error sending the webhook is swallowed (logged, not raised)."""
        from app import _watch_and_callback

        result = {"segments": [], "language": "zh"}
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "running"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch("app.asyncio.to_thread", new=AsyncMock(return_value=result)):
            with patch("httpx.AsyncClient", return_value=mock_client):
                # Should not raise
                await _watch_and_callback(job_id, Mock(), "https://hook.example.com/done")

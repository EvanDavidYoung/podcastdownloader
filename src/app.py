"""
Modal web app for podcast transcription.

This app provides a FastAPI server that:
1. Serves the React frontend (when built)
2. Exposes REST API for transcription jobs
3. Calls the transcription functions from transcribe_modal.py

Usage:
    # Development (hot reload)
    modal serve src/app.py

    # Production deployment
    cd frontend && npm run build
    modal deploy src/app.py
"""

import modal
from pathlib import Path

# ---------------------
# Modal App Setup
# ---------------------

# Image for the web server (lightweight, no GPU needed)
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]",
    "pydantic",
    "numpy",
    "python-multipart",
    "httpx",
)

app = modal.App("podcast-web")

# Check if frontend build exists
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    web_image = web_image.add_local_dir(str(frontend_dist), remote_path="/assets")

# Include web/ directory for player pages
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    web_image = web_image.add_local_dir(str(web_dir), remote_path="/web")

# Persistent volume for completed job artifacts (transcript, audio, metadata)
jobs_volume = modal.Volume.from_name("podcast-jobs", create_if_missing=True)

# ---------------------
# FastAPI Application
# ---------------------

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, Response, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import time
import os
import uuid

web_app = FastAPI(title="Podcast Transcriber API")
_bearer = HTTPBearer()


# ---------------------
# Authentication
# ---------------------

def _valid_api_keys() -> set[str]:
    """Collect all configured API keys from environment variables."""
    keys = set()
    for var in ("FASTAPI_APIKEY", "SLACK_BOT_API_KEY"):
        val = os.environ.get(var)
        if val:
            keys.add(val)
    return keys


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    """Verify the API key from the Authorization: Bearer header."""
    valid_keys = _valid_api_keys()
    if not valid_keys:
        raise HTTPException(status_code=500, detail="No API keys configured")
    if credentials.credentials not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Allow CORS for local development
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Request/Response Models
# ---------------------


class TranscribeURLRequest(BaseModel):
    url: str
    language: str = "zh"
    merge_words: bool = True
    to_traditional: bool = False
    callback_url: Optional[str] = None


class TranscribeRSSRequest(BaseModel):
    rss_url: str
    episode_index: int = 0
    language: str = "zh"
    merge_words: bool = True
    to_traditional: bool = False
    callback_url: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str = "pending"


class StatusResponse(BaseModel):
    status: str  # pending, running, completed, error
    error: Optional[str] = None


# ---------------------
# In-memory job tracking
# ---------------------

# Maps job_id -> {"call": FunctionCall, "created_at": timestamp, "status": str}
jobs: dict = {}

# Cleanup jobs older than 1 hour
JOB_TTL_SECONDS = 3600


def cleanup_old_jobs():
    """Remove expired jobs from memory."""
    now = time.time()
    expired = [jid for jid, job in jobs.items() if now - job["created_at"] > JOB_TTL_SECONDS]
    for jid in expired:
        del jobs[jid]


# ---------------------
# Webhook delivery
# ---------------------


async def _watch_and_callback(job_id: str, call, callback_url: str):
    """Wait for a Modal job to finish, then POST the result to the callback URL."""
    import httpx

    try:
        result = await asyncio.to_thread(call.get, timeout=JOB_TTL_SECONDS)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        payload = {"job_id": job_id, "status": "completed", "result": result}
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        payload = {"job_id": job_id, "status": "error", "error": str(e)}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(callback_url, json=payload)
        print(f"Webhook delivered for job {job_id} to {callback_url}")
    except Exception as e:
        print(f"Webhook delivery failed for job {job_id}: {e}")


# ---------------------
# API Endpoints
# ---------------------


@web_app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "podcast-transcriber"}


@web_app.post("/api/transcribe/url", response_model=JobResponse)
async def transcribe_from_url(
    req: TranscribeURLRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Start transcription from a direct audio URL.

    Returns a job_id to poll for status. If callback_url is provided, a POST
    request will be sent to it when the job completes or fails.
    """
    cleanup_old_jobs()

    # Pre-generate job ID so the transcription function can self-identify for artifact storage
    job_id = str(uuid.uuid4())

    # Look up the deployed transcription function
    transcribe_fn = modal.Function.from_name("podcast-transcriber", "transcribe_from_url")

    # Spawn async job
    call = transcribe_fn.spawn(
        url=req.url,
        language=req.language,
        merge_words=req.merge_words,
        to_traditional=req.to_traditional,
        job_id=job_id,
    )

    jobs[job_id] = {
        "call": call,
        "created_at": time.time(),
        "status": "running",
        "type": "url",
        "input": req.url,
        "callback_url": req.callback_url,
    }

    if req.callback_url:
        background_tasks.add_task(_watch_and_callback, job_id, call, req.callback_url)

    return JobResponse(job_id=job_id, status="running")


@web_app.post("/api/transcribe/rss", response_model=JobResponse)
async def transcribe_from_rss(
    req: TranscribeRSSRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Start transcription from an RSS feed (latest or specified episode).

    Returns a job_id to poll for status. If callback_url is provided, a POST
    request will be sent to it when the job completes or fails.
    """
    cleanup_old_jobs()

    # Pre-generate job ID so the transcription function can self-identify for artifact storage
    job_id = str(uuid.uuid4())

    # Look up the deployed transcription function
    transcribe_fn = modal.Function.from_name("podcast-transcriber", "transcribe_from_rss")

    # Spawn async job
    call = transcribe_fn.spawn(
        rss_url=req.rss_url,
        episode_index=req.episode_index,
        language=req.language,
        merge_words=req.merge_words,
        to_traditional=req.to_traditional,
        job_id=job_id,
    )

    jobs[job_id] = {
        "call": call,
        "created_at": time.time(),
        "status": "running",
        "type": "rss",
        "input": req.rss_url,
        "callback_url": req.callback_url,
    }

    if req.callback_url:
        background_tasks.add_task(_watch_and_callback, job_id, call, req.callback_url)

    return JobResponse(job_id=job_id, status="running")


@web_app.get("/api/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str, api_key: str = Depends(verify_api_key)):
    """
    Poll the status of a transcription job.

    Returns:
    - status: pending, running, completed, or error
    - error: error message (if failed)

    Use GET /api/result/{job_id} to fetch the full transcript once completed.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # If already completed/errored, return cached status
    if job["status"] in ("completed", "error"):
        return StatusResponse(
            status=job["status"],
            error=job.get("error"),
        )

    # Check if the async call has finished
    call = job["call"]
    try:
        # Non-blocking check (timeout=0)
        result = call.get(timeout=0)
        job["status"] = "completed"
        job["result"] = result
        return StatusResponse(status="completed")
    except TimeoutError:
        # Still running
        return StatusResponse(status="running")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        return StatusResponse(status="error", error=str(e))


@web_app.get("/api/result/{job_id}")
async def get_job_result(job_id: str, api_key: str = Depends(verify_api_key)):
    """
    Fetch the full transcript result for a completed job.

    Downloads as a JSON file. Only available once status is 'completed'.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] == "error":
        raise HTTPException(status_code=400, detail=job.get("error", "Job failed"))
    if job["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Job is {job['status']}, not yet completed")

    import json
    from urllib.parse import urlparse
    input_url = job.get("input", "")
    base = Path(urlparse(input_url).path).stem or job_id
    filename = f"transcript-{base}.json"

    content = json.dumps(job["result"], ensure_ascii=False, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )


@web_app.get("/api/jobs")
async def list_jobs(api_key: str = Depends(verify_api_key)):
    """List all active jobs (for debugging)."""
    cleanup_old_jobs()
    return {
        "jobs": [
            {
                "job_id": jid,
                "status": job["status"],
                "type": job.get("type"),
                "input": job.get("input"),
                "age_seconds": int(time.time() - job["created_at"]),
            }
            for jid, job in jobs.items()
        ]
    }


# ---------------------
# Player endpoints (no auth — job IDs are UUIDs, functionally unguessable)
# ---------------------


@web_app.get("/api/player/jobs")
async def list_player_jobs():
    """List all completed jobs from persistent volume, sorted newest first."""
    jobs_volume.reload()
    jobs_dir = Path("/jobs")
    result = []
    if jobs_dir.exists():
        for job_dir in sorted(jobs_dir.iterdir()):
            meta_path = job_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    result.append(json.load(f))
    result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"jobs": result}


@web_app.get("/api/player/transcript/{job_id}")
async def get_player_transcript(job_id: str):
    """Serve the transcript JSON for a completed job."""
    jobs_volume.reload()
    transcript_path = Path(f"/jobs/{job_id}/transcript.json")
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")
    return Response(content=transcript_path.read_text(encoding="utf-8"), media_type="application/json")


@web_app.get("/api/player/audio/{job_id}")
async def get_player_audio(job_id: str):
    """Stream the audio file for a completed job."""
    jobs_volume.reload()
    audio_path = Path(f"/jobs/{job_id}/audio.mp3")
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    def iterfile():
        with open(audio_path, "rb") as f:
            yield from f

    return StreamingResponse(iterfile(), media_type="audio/mpeg")


@web_app.get("/player")
async def player_listing():
    """Job listing page — lists all completed transcription jobs."""
    return FileResponse("/web/player.html")


@web_app.get("/player/{job_id}")
async def player_detail(job_id: str):
    """Transcript player for a specific job."""
    return FileResponse("/web/player-detail.html")


@web_app.post("/v1/audio/transcriptions")
async def openai_transcribe(
    file: UploadFile,
    model: str = Form("whisper-1"),
    language: str = Form("en"),
    response_format: str = Form("verbose_json"),
    timestamp_granularities: list[str] = Form(["segment"]),
    diarize: bool = Form(True),
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
):
    """
    OpenAI Whisper-compatible transcription endpoint.

    Accepts multipart/form-data with an audio file and returns a synchronous
    JSON response compatible with OpenAI's /v1/audio/transcriptions API.

    Uses StreamingResponse with keep-alive whitespace heartbeats to prevent
    proxy/gateway timeouts during long transcription jobs.
    """
    valid_keys = _valid_api_keys()
    if not valid_keys:
        raise HTTPException(status_code=500, detail="No API keys configured")
    if credentials.credentials not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    audio_bytes = await file.read()
    filename = file.filename or "audio.mp3"
    hf_token = os.environ.get("HF_TOKEN") if diarize else None

    transcribe_fn = modal.Function.from_name("podcast-transcriber", "transcribe_audio")
    call = transcribe_fn.spawn(
        audio_bytes=audio_bytes,
        filename=filename,
        language=language,
        hf_token=hf_token,
    )

    async def stream_result():
        while True:
            try:
                result = call.get(timeout=5)
                yield json.dumps({
                    "segments": result["segments"],
                    "language": result.get("language", language),
                })
                return
            except TimeoutError:
                yield " "  # keep-alive heartbeat

    return StreamingResponse(stream_result(), media_type="application/json")


# ---------------------
# Modal Function
# ---------------------


@app.function(
    image=web_image,
    scaledown_window=300,
    timeout=1800,
    secrets=[modal.Secret.from_name("api-auth")],
    volumes={"/jobs": jobs_volume},
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    """
    Serve the FastAPI application.

    In production, this also serves the React frontend from /assets.
    """
    # Mount static frontend if available
    if Path("/assets").exists():
        from fastapi.staticfiles import StaticFiles

        # Serve API routes first, then fall back to static files
        web_app.mount("/", StaticFiles(directory="/assets", html=True))

    return web_app


# ---------------------
# Local entrypoint for testing
# ---------------------


@app.local_entrypoint()
def main():
    """Print the URL when running locally."""
    print("Starting web server...")
    print("API docs available at: <modal-url>/docs")

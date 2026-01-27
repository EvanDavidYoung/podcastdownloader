"""
Modal web app for podcast transcription.

This app provides a FastAPI server that:
1. Serves the React frontend (when built)
2. Exposes REST API for transcription jobs
3. Calls the transcription functions from transcribe_modal.py

Usage:
    # Development (hot reload)
    modal serve scripts/app.py

    # Production deployment
    cd frontend && npm run build
    modal deploy scripts/app.py
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
)

app = modal.App("podcast-web")

# Check if frontend build exists
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    web_image = web_image.add_local_dir(str(frontend_dist), remote_path="/assets")

# ---------------------
# FastAPI Application
# ---------------------

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

web_app = FastAPI(title="Podcast Transcriber API")

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


class TranscribeRSSRequest(BaseModel):
    rss_url: str
    episode_index: int = 0
    language: str = "zh"
    merge_words: bool = True
    to_traditional: bool = False


class JobResponse(BaseModel):
    job_id: str
    status: str = "pending"


class StatusResponse(BaseModel):
    status: str  # pending, running, completed, error
    result: Optional[dict] = None
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
# API Endpoints
# ---------------------


@web_app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "podcast-transcriber"}


@web_app.post("/api/transcribe/url", response_model=JobResponse)
async def transcribe_from_url(req: TranscribeURLRequest):
    """
    Start transcription from a direct audio URL.

    Returns a job_id to poll for status.
    """
    cleanup_old_jobs()

    # Look up the deployed transcription function
    transcribe_fn = modal.Function.lookup("podcast-transcriber", "transcribe_from_url")

    # Spawn async job
    call = transcribe_fn.spawn(
        url=req.url,
        language=req.language,
        merge_words=req.merge_words,
        to_traditional=req.to_traditional,
    )

    job_id = call.object_id
    jobs[job_id] = {
        "call": call,
        "created_at": time.time(),
        "status": "running",
        "type": "url",
        "input": req.url,
    }

    return JobResponse(job_id=job_id, status="running")


@web_app.post("/api/transcribe/rss", response_model=JobResponse)
async def transcribe_from_rss(req: TranscribeRSSRequest):
    """
    Start transcription from an RSS feed (latest or specified episode).

    Returns a job_id to poll for status.
    """
    cleanup_old_jobs()

    # Look up the deployed transcription function
    transcribe_fn = modal.Function.lookup("podcast-transcriber", "transcribe_from_rss")

    # Spawn async job
    call = transcribe_fn.spawn(
        rss_url=req.rss_url,
        episode_index=req.episode_index,
        language=req.language,
        merge_words=req.merge_words,
        to_traditional=req.to_traditional,
    )

    job_id = call.object_id
    jobs[job_id] = {
        "call": call,
        "created_at": time.time(),
        "status": "running",
        "type": "rss",
        "input": req.rss_url,
    }

    return JobResponse(job_id=job_id, status="running")


@web_app.get("/api/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """
    Poll the status of a transcription job.

    Returns:
    - status: pending, running, completed, or error
    - result: transcript data (if completed)
    - error: error message (if failed)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # If already completed/errored, return cached result
    if job["status"] in ("completed", "error"):
        return StatusResponse(
            status=job["status"],
            result=job.get("result"),
            error=job.get("error"),
        )

    # Check if the async call has finished
    call = job["call"]
    try:
        # Non-blocking check (timeout=0)
        result = call.get(timeout=0)
        job["status"] = "completed"
        job["result"] = result
        return StatusResponse(status="completed", result=result)
    except TimeoutError:
        # Still running
        return StatusResponse(status="running")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        return StatusResponse(status="error", error=str(e))


@web_app.get("/api/jobs")
async def list_jobs():
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
# Modal Function
# ---------------------


@app.function(
    image=web_image,
    allow_concurrent_inputs=100,
    container_idle_timeout=300,  # Keep warm for 5 minutes
)
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

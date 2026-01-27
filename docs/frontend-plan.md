# Frontend Architecture Plan

## Proposed Project Structure

```
PodcastDownloader/
├── frontend/                    # React + Vite SPA
│   ├── src/
│   │   ├── components/
│   │   │   ├── TranscriptPlayer.tsx    # Port of transcript-player.html
│   │   │   ├── PodcastSearch.tsx       # Search/browse podcasts
│   │   │   ├── TranscribeForm.tsx      # Upload/URL input for transcription
│   │   │   ├── TTSPanel.tsx            # Text-to-speech controls
│   │   │   └── JobStatus.tsx           # Transcription progress indicator
│   │   ├── api/
│   │   │   └── client.ts               # API client for backend calls
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── scripts/
│   ├── app.py                   # NEW: Main Modal app (FastAPI + workers)
│   ├── transcribe_modal.py      # Existing transcription functions
│   ├── tts_modal.py             # NEW: ElevenLabs TTS functions
│   ├── download_podcast.py
│   ├── transcribe_local.py
│   ├── merge_chinese_words.py
│   └── convert_to_traditional.py
│
├── web/                         # Keep for standalone HTML player
│   └── transcript-player.html
│
├── docs/
├── downloads/
└── ...
```

## Backend: scripts/app.py

Main Modal application that:
1. Serves the React frontend (static files)
2. Exposes REST API endpoints
3. Orchestrates transcription and TTS workers

```python
"""
Modal app serving the podcast transcriber frontend and API.

Deploy: modal deploy scripts/app.py
Dev:    modal serve scripts/app.py
"""

import modal
from pathlib import Path

# Import existing transcription functions
from transcribe_modal import transcribe_audio, transcribe_from_url, transcribe_from_rss

app = modal.App("podcast-transcriber-web")

# Build frontend into the image
frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]")
    .add_local_dir(frontend_path, remote_path="/assets")
)

# ---------------------
# API Endpoints
# ---------------------

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

web_app = FastAPI()

class TranscribeRequest(BaseModel):
    url: str | None = None
    rss_url: str | None = None
    language: str = "zh"
    to_traditional: bool = False

class TTSRequest(BaseModel):
    text: str
    voice_id: str = "default"

# In-progress jobs cache
jobs: dict[str, dict] = {}

@web_app.post("/api/transcribe")
async def start_transcription(req: TranscribeRequest):
    """Start a transcription job, returns job_id for polling."""
    if req.rss_url:
        call = transcribe_from_rss.spawn(
            rss_url=req.rss_url,
            language=req.language,
            to_traditional=req.to_traditional,
        )
    elif req.url:
        call = transcribe_from_url.spawn(
            url=req.url,
            language=req.language,
            to_traditional=req.to_traditional,
        )
    else:
        raise HTTPException(400, "Provide url or rss_url")

    job_id = call.object_id
    jobs[job_id] = {"status": "processing", "call": call}
    return {"job_id": job_id}

@web_app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    call = job["call"]

    try:
        result = call.get(timeout=0)  # Non-blocking check
        jobs[job_id] = {"status": "completed", "result": result}
        return {"status": "completed", "result": result}
    except TimeoutError:
        return {"status": "processing"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@web_app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """Generate speech from text using ElevenLabs."""
    from tts_modal import generate_speech
    audio_bytes = generate_speech.remote(req.text, req.voice_id)

    from fastapi.responses import Response
    return Response(content=audio_bytes, media_type="audio/mpeg")

@web_app.get("/api/health")
async def health():
    return {"status": "ok"}

# ---------------------
# Main App
# ---------------------

@app.function(image=image, concurrency_limit=10)
@modal.asgi_app()
def serve():
    """Serve frontend and API."""
    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app
```

## Backend: scripts/tts_modal.py

```python
"""ElevenLabs TTS functions for Modal."""

import modal

app = modal.App("podcast-tts")

image = modal.Image.debian_slim().pip_install("requests")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("elevenlabs")],
)
def generate_speech(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> bytes:
    """Generate speech using ElevenLabs API."""
    import os
    import requests

    api_key = os.environ["ELEVENLABS_API_KEY"]

    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_monolingual_v1",
        },
    )
    response.raise_for_status()
    return response.content
```

## Frontend API Client: frontend/src/api/client.ts

```typescript
const API_BASE = '/api';

export interface TranscribeRequest {
  url?: string;
  rss_url?: string;
  language?: string;
  to_traditional?: boolean;
}

export interface JobStatus {
  status: 'processing' | 'completed' | 'error';
  result?: TranscriptResult;
  error?: string;
}

export interface TranscriptResult {
  segments: Segment[];
  word_segments: Word[];
  language: string;
}

export interface Segment {
  start: number;
  end: number;
  text: string;
  speaker?: string;
  words?: Word[];
}

export interface Word {
  word: string;
  start: number;
  end: number;
  score?: number;
}

export async function startTranscription(req: TranscribeRequest): Promise<{ job_id: string }> {
  const res = await fetch(`${API_BASE}/transcribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function pollStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/status/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function textToSpeech(text: string, voiceId?: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, voice_id: voiceId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

// Polling helper
export async function waitForTranscription(
  jobId: string,
  onProgress?: (status: JobStatus) => void,
  intervalMs = 2000
): Promise<TranscriptResult> {
  while (true) {
    const status = await pollStatus(jobId);
    onProgress?.(status);

    if (status.status === 'completed') {
      return status.result!;
    }
    if (status.status === 'error') {
      throw new Error(status.error);
    }

    await new Promise(resolve => setTimeout(resolve, intervalMs));
  }
}
```

## Development Workflow

```bash
# Terminal 1: Frontend dev server (with hot reload)
cd frontend
npm install
npm run dev  # Runs on localhost:5173

# Terminal 2: Modal backend (with hot reload)
modal serve scripts/app.py  # Runs on Modal, prints URL

# For frontend dev, proxy API calls to Modal:
# In vite.config.ts, add proxy config
```

## Production Deployment

```bash
# 1. Build frontend
cd frontend && npm run build

# 2. Deploy everything to Modal
modal deploy scripts/app.py

# Output: https://your-app--podcast-transcriber-web-serve.modal.run
```

## ElevenLabs Setup

```bash
# Create Modal secret for ElevenLabs
modal secret create elevenlabs ELEVENLABS_API_KEY=your_key_here
```

## Key Features to Build

### Phase 1: Core
- [ ] Transcribe from URL input
- [ ] Transcribe from RSS feed
- [ ] Job status polling with progress UI
- [ ] Transcript viewer (port transcript-player.html)

### Phase 2: Enhanced
- [ ] Podcast search (via Podchaser or iTunes API)
- [ ] Speaker labels in transcript
- [ ] Download transcript (JSON, SRT, VTT)
- [ ] Audio playback with word highlighting

### Phase 3: TTS
- [ ] Select text → generate speech
- [ ] Voice selection
- [ ] Download generated audio

## Questions to Decide

1. **Podcast discovery**: Use Podchaser API (requires key) or iTunes Search API (free)?
2. **Storage**: Store transcripts in Modal Volume or return to client only?
3. **Authentication**: Add user auth, or keep it open?
4. **Rate limiting**: Limit transcription requests per IP/user?

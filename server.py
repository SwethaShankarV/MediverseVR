"""
server.py
─────────
MediverseVR Narration Pipeline — FastAPI Backend

Runs entirely locally (no cloud costs). Designed so that when you are ready
to deploy, you only change the storage and server configuration — not this code.

Endpoints
─────────
  POST /process
      Upload a video file + surgery events JSON.
      Starts the narration pipeline in the background.
      Returns: { job_id, status, message }

  GET  /status/{job_id}
      Poll for job progress.
      Returns: { job_id, status, step, message, created_at, updated_at }

  GET  /download/{job_id}
      Download the finished narrated MP4.
      Returns: the .mp4 file as an attachment.

  GET  /jobs
      List all jobs and their statuses.

  DELETE /jobs/{job_id}
      Remove a job and all its files.

  GET  /health
      Simple liveness check.

  GET  /
      Serves the web UI (static/index.html) for manual testing in a browser.

Job lifecycle
─────────────
  queued → running → done
                   → failed

All job files are stored under ./jobs/<job_id>/
  input_video.<ext>          original uploaded video
  surgery_events.json        uploaded event log
  final_mapped.json          Step 1 output
  audio_steps/               Step 2 output (WAV files)
  audio_manifest.json        Step 2 output (manifest)
  captions.srt               Step 3 output
  narrated_<name>.mp4        Step 4 output (download target)
  pipeline.log               full stdout/stderr from run_pipeline.py

Usage
─────
  # Start the server (development mode with auto-reload)
  python server.py

  # Or via uvicorn directly
  uvicorn server:app --reload --host 0.0.0.0 --port 8000

  # Then open http://localhost:8000 in your browser
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import (
    BackgroundTasks, FastAPI, File, Form, HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE      = Path(__file__).parent.resolve()
JOBS_DIR  = HERE / "jobs"
STATIC_DIR = HERE / "static"

JOBS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

PIPELINE_SCRIPT = HERE / "run_pipeline.py"
PYTHON          = sys.executable   # same interpreter that's running this file

# ── In-memory job store ───────────────────────────────────────────────────────
# Simple dict; on restart jobs are lost (fine for local dev).
# Replace with SQLite/Redis for production.

_jobs: dict = {}    # job_id → dict
_lock = threading.Lock()

# ── Job helpers ───────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _create_job(job_id: str, video_filename: str) -> dict:
    job = {
        "job_id":     job_id,
        "status":     "queued",
        "step":       "waiting",
        "message":    "Job queued, waiting to start.",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "video_filename": video_filename,
        "output_file": None,
    }
    with _lock:
        _jobs[job_id] = job
    return job


def _update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _jobs[job_id]["updated_at"] = _now_iso()


def _get_job(job_id: str) -> Optional[dict]:
    with _lock:
        return dict(_jobs.get(job_id, {}))


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


# ── Background pipeline runner ────────────────────────────────────────────────

def _run_pipeline(job_id: str, video_path: Path, events_path: Path) -> None:
    """
    Called in a background thread. Invokes run_pipeline.py as a subprocess
    so that heavy imports (torch, transformers) don't block the API server.
    """
    jdir    = job_dir(job_id)
    log_path = jdir / "pipeline.log"

    # Derive output path
    base        = video_path.stem
    output_mp4  = jdir / f"narrated_{base}.mp4"

    _update_job(job_id,
        status="running",
        step="starting",
        message="Pipeline starting…",
    )

    cmd = [
        PYTHON, str(PIPELINE_SCRIPT),
        "--video",          str(video_path),
        "--events",         str(events_path),
        "--output",         str(output_mp4),
        "--work-dir",       str(jdir),
        # Skip narration + TTS if final_mapped.json and audio_manifest.json
        # already exist in the job dir (unlikely on first run, useful for retries)
    ]

    try:
        with open(log_path, "w") as log_f:
            _update_job(job_id, step="running", message="Pipeline running…")
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(HERE),   # scripts resolve default paths relative to HERE
            )
            proc.wait()

        if proc.returncode == 0 and output_mp4.exists():
            _update_job(job_id,
                status="done",
                step="complete",
                message="Pipeline finished. Video is ready to download.",
                output_file=str(output_mp4),
            )
        else:
            # Read the last 20 lines of the log for the error message
            try:
                lines = log_path.read_text().splitlines()
                tail  = "\n".join(lines[-20:])
            except Exception:
                tail = "(log unavailable)"
            _update_job(job_id,
                status="failed",
                step="error",
                message=f"Pipeline exited with code {proc.returncode}.\n{tail}",
            )

    except Exception as exc:
        _update_job(job_id,
            status="failed",
            step="error",
            message=f"Unexpected error: {exc}",
        )


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="MediverseVR Narration API",
    description="Upload VR surgery recordings and get back narrated MP4 videos.",
    version="1.0.0",
)

# Serve static files (the web UI) from ./static/
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the test web UI."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        async with aiofiles.open(index, "r") as f:
            return HTMLResponse(await f.read())
    return HTMLResponse("<h1>MediverseVR API</h1><p>Visit <a href='/docs'>/docs</a> for the API.</p>")


@app.get("/health")
def health():
    """Liveness check — used by load balancers and Unity to verify the server is up."""
    return {"status": "ok", "timestamp": _now_iso()}


@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    video: UploadFile   = File(...,  description="Recorded VR session video (MP4/AVI/MOV)"),
    events: UploadFile  = File(...,  description="surgery_events.json from Unity"),
    skip_narration: bool = Form(False, description="Skip BioMistral (use existing final_mapped.json if present)"),
    skip_tts: bool       = Form(False, description="Skip Coqui TTS (use existing WAVs if present)"),
):
    """
    Upload a video + events file to start the narration pipeline.

    Returns a job_id you can use to poll /status/{job_id} and then
    download from /download/{job_id}.
    """
    # ── Validate file types ───────────────────────────────────────────
    allowed_video = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_ext     = Path(video.filename).suffix.lower()
    if video_ext not in allowed_video:
        raise HTTPException(
            status_code=400,
            detail=f"Video must be one of {allowed_video}. Got: {video_ext}",
        )

    if not events.filename.endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail="Events file must be a .json file.",
        )

    # ── Create job directory ──────────────────────────────────────────
    job_id   = str(uuid.uuid4())
    jdir     = job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    video_path  = jdir / f"input_video{video_ext}"
    events_path = jdir / "surgery_events.json"

    # ── Save uploaded files ───────────────────────────────────────────
    async with aiofiles.open(video_path, "wb") as f:
        await f.write(await video.read())

    async with aiofiles.open(events_path, "wb") as f:
        await f.write(await events.read())

    # ── Validate events JSON ──────────────────────────────────────────
    try:
        content = events_path.read_text()
        parsed  = json.loads(content)
        if not isinstance(parsed, dict) or ("data" not in parsed and "Items" not in parsed):
            raise ValueError("Expected a JSON object with a 'data' or 'Items' key.")
    except Exception as exc:
        shutil.rmtree(jdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid events JSON: {exc}")

    # ── Build the pipeline command (with optional skip flags) ─────────
    # We write a tiny wrapper script into the job dir so that
    # skip_narration / skip_tts can be passed through cleanly.
    run_cmd_path = jdir / "run.sh"
    skip_flags   = []
    if skip_narration:
        skip_flags.append("--skip-narration")
    if skip_tts:
        skip_flags.append("--skip-tts")

    # ── Register and start job ────────────────────────────────────────
    _create_job(job_id, video.filename)

    # Run in a background thread so the API stays responsive
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, video_path, events_path),
        daemon=True,
    )
    thread.start()

    return JSONResponse(status_code=202, content={
        "job_id":  job_id,
        "status":  "queued",
        "message": "Job created. Poll /status/{job_id} to track progress.",
        "poll_url":     f"/status/{job_id}",
        "download_url": f"/download/{job_id}",
    })


@app.get("/status/{job_id}")
def status(job_id: str):
    """
    Poll this endpoint to check job progress.

    Statuses:
      queued   — job created, waiting for a worker thread
      running  — pipeline is actively executing
      done     — finished, video ready at /download/{job_id}
      failed   — something went wrong (check 'message' for details)
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/download/{job_id}")
def download(job_id: str):
    """
    Download the finished narrated MP4.
    Only available when /status/{job_id} returns status='done'.
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not done yet (status='{job['status']}'). "
                   f"Poll /status/{job_id} first.",
        )

    output_file = job.get("output_file")
    if not output_file or not Path(output_file).exists():
        raise HTTPException(
            status_code=500,
            detail="Output file missing — the pipeline may have failed silently.",
        )

    filename = Path(output_file).name
    return FileResponse(
        path=output_file,
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/jobs")
def list_jobs():
    """List all jobs with their current status."""
    with _lock:
        jobs = list(_jobs.values())
    # Sort newest first
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return {"total": len(jobs), "jobs": jobs}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """
    Delete a job and remove all its files from disk.
    Running jobs cannot be deleted (stop them first).
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a running job. Wait for it to finish.",
        )

    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    with _lock:
        _jobs.pop(job_id, None)

    return {"message": f"Job '{job_id}' deleted."}


@app.get("/logs/{job_id}", response_class=HTMLResponse)
async def get_logs(job_id: str):
    """Return the pipeline log for a job as plain text (useful for debugging)."""
    log_path = job_dir(job_id) / "pipeline.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found.")
    async with aiofiles.open(log_path, "r", errors="replace") as f:
        content = await f.read()
    # Return as <pre> so it's readable in a browser
    return HTMLResponse(f"<pre style='font-family:monospace;font-size:13px'>{content}</pre>")


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "═"*60)
    print("  MediverseVR Narration API  —  local dev server")
    print("═"*60)
    print(f"\n  API:      http://localhost:8000")
    print(f"  Web UI:   http://localhost:8000")
    print(f"  API docs: http://localhost:8000/docs")
    print(f"  Jobs dir: {JOBS_DIR}")
    print(f"\n  Press Ctrl+C to stop.\n")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # auto-restarts when you edit server.py
        reload_dirs=[str(HERE)],
    )

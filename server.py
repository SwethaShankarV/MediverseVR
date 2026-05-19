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

# Load .env so env vars (Supabase, etc.) are visible to the server process.
# python-dotenv is optional — if it's not installed, env vars must be set
# in the shell before launching the server.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Firebase uploader — optional; gracefully skipped when not configured
try:
    from firebase_uploader import firebase_configured, upload_job as _firebase_upload_job
    _FIREBASE_AVAILABLE = True
except ImportError:
    _FIREBASE_AVAILABLE = False
    def firebase_configured(): return False  # type: ignore[misc]

# Supabase uploader — optional; gracefully skipped when credentials aren't set
try:
    from supabase_uploader import upload_job as _supabase_upload_job
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False


def _supabase_configured() -> bool:
    """Supabase is considered enabled if URL + any usable key is in the env."""
    if not _SUPABASE_AVAILABLE:
        return False
    if not os.environ.get("SUPABASE_URL"):
        return False
    return any(os.environ.get(k) for k in (
        "SUPABASE_SECRET_KEY",
        "SUPABASE_SERVICE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    ))

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

def _run_pipeline(
    job_id: str,
    video_path: Path,
    events_path: Path,
    skip_narration: bool = False,
    skip_tts: bool = False,
    vr_assets_only: bool = False,
) -> None:
    """
    Called in a background thread. Invokes run_pipeline.py as a subprocess
    so that heavy imports (torch, transformers) don't block the API server.

    vr_assets_only  — passes --no-video; job succeeds after Steps 1–3 so
                      Unity can immediately download narration text + audio.
    """
    jdir     = job_dir(job_id)
    log_path = jdir / "pipeline.log"

    # Derive output path (only used when vr_assets_only=False)
    base       = video_path.stem
    output_mp4 = jdir / f"narrated_{base}.mp4"

    _update_job(job_id,
        status="running",
        step="starting",
        message="Pipeline starting…",
    )

    # ── Seed skip files into job dir ─────────────────────────────────────────
    # run_pipeline.py requires that skipped-step outputs already exist IN the
    # work dir (jdir). If the caller requests skipping but the file isn't there
    # yet, copy it from the pipeline root as a fallback so the skip succeeds.

    if skip_narration:
        job_fm = jdir / "final_mapped.json"
        if not job_fm.exists():
            src_fm = HERE / "final_mapped.json"
            if src_fm.exists():
                shutil.copy(src_fm, job_fm)
                print(f"[server] Seeded final_mapped.json from pipeline root into {jdir.name}")
            else:
                # Nothing to copy — let the pipeline fail with a clear error
                print("[server] Warning: skip_narration=True but no final_mapped.json found.")

    if skip_tts:
        job_manifest = jdir / "audio_manifest.json"
        if not job_manifest.exists():
            src_manifest = HERE / "audio_manifest.json"
            if src_manifest.exists():
                shutil.copy(src_manifest, job_manifest)
                # Also copy the WAV files
                src_audio = HERE / "audio_steps"
                dst_audio = jdir / "audio_steps"
                if src_audio.exists():
                    shutil.copytree(src_audio, dst_audio, dirs_exist_ok=True)
                print(f"[server] Seeded audio_manifest.json + audio_steps/ from pipeline root.")

    # ── Demo video fallback ──────────────────────────────────────────────────
    # Unity sends a 0-byte placeholder when Upload Video is unticked.
    # If that happens, substitute demo_recording.mp4 from the pipeline root
    # so the video composition step (Step 4) still produces a real MP4.
    if not vr_assets_only and video_path.stat().st_size == 0:
        demo_video = HERE / "demo_recording.mp4"
        if demo_video.exists():
            real_video = jdir / "input_video.mp4"
            shutil.copy(demo_video, real_video)
            video_path = real_video
            # Recalculate output path now that we know the stem
            base       = video_path.stem
            output_mp4 = jdir / f"narrated_{base}.mp4"
            print(f"[server] No video uploaded — using demo_recording.mp4 as video source")
        else:
            print("[server] Warning: 0-byte video received and no demo_recording.mp4 found. "
                  "Video composition will likely fail.")

    cmd = [
        PYTHON, str(PIPELINE_SCRIPT),
        "--events",   str(events_path),
        "--work-dir", str(jdir),
    ]

    if vr_assets_only:
        cmd.append("--no-video")
    else:
        cmd += ["--video", str(video_path), "--output", str(output_mp4)]

    if skip_narration:
        cmd.append("--skip-narration")
    if skip_tts:
        cmd.append("--skip-tts")

    # Always strip original video audio — the demo_recording.mp4 has its own
    # audio track which overlaps with the AI narration. Narration-only is cleaner.
    if not vr_assets_only:
        cmd.append("--no-original-audio")

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

        # Success criteria:
        #   vr_assets_only → Steps 1–3 produced final_mapped.json
        #   normal         → Step 4 produced narrated MP4
        vr_success    = vr_assets_only and (jdir / "final_mapped.json").exists()
        video_success = (not vr_assets_only) and proc.returncode == 0 and output_mp4.exists()

        if proc.returncode == 0 and (vr_success or video_success):
            # Collect audio asset paths for Unity to download
            audio_dir   = jdir / "audio_steps"
            audio_files = sorted(f.name for f in audio_dir.glob("*.wav")) if audio_dir.exists() else []

            done_msg = (
                "VR narration assets ready. Open the Explanation panel to play."
                if vr_assets_only else
                "Pipeline finished. Video is ready to download."
            )

            _update_job(job_id,
                status="done",
                step="complete",
                message=done_msg,
                output_file=str(output_mp4) if video_success else None,
                # Asset URLs for direct download by Unity
                final_mapped_url=f"/assets/{job_id}/final_mapped.json"
                    if (jdir / "final_mapped.json").exists() else None,
                audio_manifest_url=f"/assets/{job_id}/audio_manifest.json"
                    if (jdir / "audio_manifest.json").exists() else None,
                audio_urls=[f"/assets/{job_id}/audio_steps/{f}" for f in audio_files],
                srt_url=f"/assets/{job_id}/captions.srt"
                    if (jdir / "captions.srt").exists() else None,
            )

            # ── Upload to Firebase (non-blocking, non-fatal) ───────────────
            if _FIREBASE_AVAILABLE and firebase_configured():
                try:
                    firebase_urls = _firebase_upload_job(job_id, jdir)
                    _update_job(job_id, firebase_urls=firebase_urls)
                    _update_job(job_id,
                        message=done_msg + " Replay also available on Firebase.")
                    print(f"[server] Firebase upload complete for job {job_id}")
                except Exception as fb_err:
                    print(f"[server] Firebase upload failed (non-fatal): {fb_err}")
            else:
                print("[server] Firebase not configured — skipping upload. "
                      "Set env vars from .env.example to enable.")

            # ── Upload to Supabase (non-blocking, non-fatal) ───────────────
            # Storage + replays table row. Same pattern as Firebase — if
            # anything goes wrong, the local pipeline still succeeds.
            if _supabase_configured():
                try:
                    sb_result = _supabase_upload_job(
                        job_dir  = jdir,
                        language = "en",     # TODO: pipe through from /process when we add a language picker
                        title    = f"Surgery session {job_id[:8]}",
                    )
                    _update_job(job_id,
                        supabase_row_id  = sb_result.get("id"),
                        supabase_urls    = {
                            k: v for k, v in sb_result.items() if k != "id"
                        },
                        message = done_msg + " Replay also uploaded to Supabase.",
                    )
                    print(f"[server] Supabase upload complete for job {job_id} "
                          f"(row id: {sb_result.get('id')})")
                except Exception as sb_err:
                    print(f"[server] Supabase upload failed (non-fatal): {sb_err}")
            else:
                print("[server] Supabase not configured — skipping upload. "
                      "Set SUPABASE_URL + SUPABASE_SERVICE_KEY in .env to enable.")

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
    video: UploadFile   = File(...,  description="Recorded VR session video (MP4/AVI/MOV). Send an empty file with a .mp4 name when vr_assets_only=true."),
    events: UploadFile  = File(...,  description="surgery_events.json from Unity"),
    skip_narration: bool = Form(False, description="Skip BioMistral (use existing final_mapped.json if present)"),
    skip_tts: bool       = Form(False, description="Skip Coqui TTS (use existing WAVs if present)"),
    vr_assets_only: bool = Form(False, description="Only generate VR narration assets (Steps 1–3). Skips video composition. Use this from standalone Quest headsets."),
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

    # ── Register and start job ────────────────────────────────────────
    _create_job(job_id, video.filename)

    # Run in a background thread so the API stays responsive
    thread = threading.Thread(
        target=_run_pipeline,
        kwargs={
            "job_id":         job_id,
            "video_path":     video_path,
            "events_path":    events_path,
            "skip_narration": skip_narration,
            "skip_tts":       skip_tts,
            "vr_assets_only": vr_assets_only,
        },
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


@app.get("/assets/{job_id}/{file_path:path}")
def get_asset(job_id: str, file_path: str):
    """
    Serve a specific output file from a job directory.

    Used by NarrationPipelineTrigger.cs to download individual assets:
      /assets/{job_id}/final_mapped.json
      /assets/{job_id}/audio_manifest.json
      /assets/{job_id}/audio_steps/step_0000_<name>.wav
      /assets/{job_id}/captions.srt

    Returns 404 if the job or file does not exist.
    """
    jdir_path  = job_dir(job_id).resolve()
    asset_path = (jdir_path / file_path).resolve()

    # Security: prevent path traversal outside the job directory
    if not str(asset_path).startswith(str(jdir_path)):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not asset_path.exists():
        raise HTTPException(status_code=404, detail=f"Asset not found: {file_path}")

    # Pick a sensible media type
    suffix = asset_path.suffix.lower()
    media_type = {
        ".json": "application/json",
        ".wav":  "audio/wav",
        ".srt":  "text/plain",
        ".mp4":  "video/mp4",
    }.get(suffix, "application/octet-stream")

    return FileResponse(str(asset_path), media_type=media_type)


@app.get("/replays")
def list_replays():
    """
    List all available replays with their Firebase download URLs.

    When Firebase is configured: queries Firestore for all replay documents.
    When Firebase is not configured: falls back to returning locally finished jobs.

    Unity's FirebaseReplayBrowser.cs calls this endpoint to populate
    the in-headset replay browser.
    """
    if _FIREBASE_AVAILABLE and firebase_configured():
        try:
            import firebase_admin  # noqa: PLC0415
            from firebase_admin import firestore  # noqa: PLC0415
            from firebase_uploader import _init_firebase  # noqa: PLC0415
            _init_firebase()
            db     = firestore.client()
            docs   = db.collection("replays").order_by(
                "uploaded_at", direction=firestore.Query.DESCENDING
            ).limit(50).get()
            replays = [doc.to_dict() for doc in docs]
            return {"source": "firebase", "total": len(replays), "replays": replays}
        except Exception as exc:
            # Fall through to local fallback
            print(f"[server] Firestore query failed, falling back to local: {exc}")

    # Local fallback — done jobs only
    with _lock:
        done = [j for j in _jobs.values() if j["status"] == "done"]
    done.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return {"source": "local", "total": len(done), "replays": done}


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
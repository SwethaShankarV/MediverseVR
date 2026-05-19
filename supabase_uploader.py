"""
supabase_uploader.py
────────────────────
Uploads a finished MediverseVR job (narrated MP4 + audio WAVs + JSON files +
SRT captions) to Supabase Storage and inserts a metadata row into the
`replays` database table so the files can be browsed and downloaded by
other users.

Reads credentials from environment variables:
  SUPABASE_URL          e.g. https://xxxxxxxxxxxx.supabase.co
  SUPABASE_SECRET_KEY   sb_secret_xxx  (server-side use only — bypasses RLS)

Also supports the legacy variable names if a project hasn't migrated yet:
  SUPABASE_SERVICE_KEY  (treated as alias for SUPABASE_SECRET_KEY)
  SUPABASE_SERVICE_ROLE_KEY  (treated as alias)

Usage from the command line (for manual upload of an existing job):
  python supabase_uploader.py --job-dir jobs/<job_id> --language en --title "Foot surgery demo"

Usage from server.py (programmatic — see upload_job() at the bottom).

Output
──────
Returns a dict with public URLs for each uploaded file plus the database
row ID. The URLs are immediately accessible because the `replays` bucket
is configured as public. Example:
  {
    "id": "5e3f...",
    "video_url":          "https://xxx.supabase.co/storage/v1/object/public/replays/<job_id>/narrated_session.mp4",
    "audio_manifest_url": "https://xxx.supabase.co/...",
    "events_url":         "https://xxx.supabase.co/...",
    "srt_url":            "https://xxx.supabase.co/...",
  }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ── Lazy SDK import so this file can be imported without supabase installed ──

def _get_supabase_client():
    """
    Return an initialized Supabase client using credentials from env vars.
    Raises a clear error if anything is missing.
    """
    try:
        from supabase import create_client
    except ImportError:
        raise RuntimeError(
            "supabase package not installed. Install with: "
            ".venv/bin/pip install supabase python-dotenv"
        )

    # Load .env automatically if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass   # not fatal — env vars may be set in the shell directly

    url = os.environ.get("SUPABASE_URL")
    # Accept new and legacy key names so we work across migration states
    secret = (
        os.environ.get("SUPABASE_SECRET_KEY")
        or os.environ.get("SUPABASE_SERVICE_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    )

    if not url:
        raise RuntimeError("SUPABASE_URL is not set. Add it to .env or your shell.")
    if not secret:
        raise RuntimeError(
            "No Supabase secret key found. Set one of: "
            "SUPABASE_SECRET_KEY, SUPABASE_SERVICE_KEY, SUPABASE_SERVICE_ROLE_KEY"
        )

    return create_client(url, secret), url


# ── Core upload logic ────────────────────────────────────────────────────────

BUCKET_NAME = "replays"


def _upload_file(client, local_path: Path, remote_path: str,
                 content_type: str | None = None) -> str:
    """
    Upload a single file to the Supabase storage bucket and return its public URL.
    Uses upsert=True so re-running the same job overwrites cleanly.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"File to upload does not exist: {local_path}")

    with open(local_path, "rb") as f:
        file_bytes = f.read()

    # The Python SDK signature is: storage.from_(bucket).upload(path, file, file_options)
    file_options = {"upsert": "true"}     # string "true" required by SDK quirks
    if content_type:
        file_options["content-type"] = content_type

    response = client.storage.from_(BUCKET_NAME).upload(
        path         = remote_path,
        file         = file_bytes,
        file_options = file_options,
    )

    # In newer SDK versions, upload returns an object with a .path attribute;
    # in older ones it returns a dict. Public URL is built from the path.
    public_url = client.storage.from_(BUCKET_NAME).get_public_url(remote_path)
    return public_url


def upload_job(job_dir: Path | str,
               language: str = "en",
               title: str | None = None,
               video_filename: str | None = None) -> dict:
    """
    Upload all artifacts in a job directory to Supabase and insert a metadata row.

    Args:
      job_dir         — path to the finished job folder containing:
                        narrated_*.mp4, audio_manifest.json, events.json,
                        captions.srt (any of these may be missing — only
                        present files are uploaded)
      language        — language code stored in the row ("en", "zh", "hi", …)
      title           — human-readable title; auto-generated from job dir name if None
      video_filename  — name of the MP4 to upload. If None, searches for a
                        file matching narrated_*.mp4 in job_dir.

    Returns: dict with the row ID and public URLs for each uploaded file.
    """
    job_dir = Path(job_dir).resolve()
    if not job_dir.is_dir():
        raise NotADirectoryError(f"Job directory not found: {job_dir}")

    client, url_root = _get_supabase_client()
    job_id = job_dir.name

    print(f"[supabase] Uploading job {job_id} → {url_root}/{BUCKET_NAME}/{job_id}/")

    # ── Locate the narrated video ──
    if video_filename is None:
        candidates = list(job_dir.glob("narrated_*.mp4"))
        if not candidates:
            # Some demos use a fixed name — try that too
            for fallback in ["narrated_session.mp4", "narrated_recording.mp4"]:
                if (job_dir / fallback).exists():
                    candidates = [job_dir / fallback]
                    break
        if not candidates:
            raise FileNotFoundError(
                f"No narrated_*.mp4 found in {job_dir}. "
                f"Pass --video-filename explicitly if the MP4 has a custom name."
            )
        video_path = candidates[0]
    else:
        video_path = job_dir / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

    # ── Upload files (only those that exist) ──
    urls = {}

    # Required: the video
    urls["video_url"] = _upload_file(
        client, video_path, f"{job_id}/{video_path.name}",
        content_type = "video/mp4",
    )
    print(f"  ✔  video         → {video_path.name}")

    # Optional artifacts — upload whichever ones we find
    optional_files = [
        ("audio_manifest.json", "audio_manifest_url", "application/json"),
        ("final_mapped.json",   "final_mapped_url",   "application/json"),
        ("events.json",         "events_url",         "application/json"),
        ("surgery_events.json", "events_url",         "application/json"),
        ("CombinedToolLog.json", "events_url",        "application/json"),
        ("captions.srt",        "srt_url",            "text/plain"),
    ]

    for filename, url_key, mime in optional_files:
        local = job_dir / filename
        if local.exists() and url_key not in urls:    # don't overwrite if already set
            urls[url_key] = _upload_file(
                client, local, f"{job_id}/{filename}",
                content_type = mime,
            )
            print(f"  ✔  {url_key:<22} → {filename}")

    # Upload WAV files inside audio_steps/ if present
    audio_steps_dir = job_dir / "audio_steps"
    if audio_steps_dir.is_dir():
        wav_count = 0
        for wav in sorted(audio_steps_dir.glob("*.wav")):
            _upload_file(
                client, wav, f"{job_id}/audio_steps/{wav.name}",
                content_type = "audio/wav",
            )
            wav_count += 1
        if wav_count:
            print(f"  ✔  audio_steps/   → {wav_count} WAV file(s)")

    # ── Probe video duration via ffprobe (optional but nice to have) ──
    duration_seconds = None
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        duration_seconds = float(result.stdout.strip()) if result.returncode == 0 else None
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass

    # ── Insert the metadata row ──
    if title is None:
        title = f"Surgery session {job_id[:8]} ({language})"

    row = {
        "title":               title,
        "language":            language,
        "duration_seconds":    duration_seconds,
        "video_url":           urls.get("video_url"),
        "audio_manifest_url":  urls.get("audio_manifest_url"),
        "events_url":          urls.get("events_url"),
        "srt_url":             urls.get("srt_url"),
    }

    print(f"[supabase] Inserting row into 'replays' table…")
    response = client.table("replays").insert(row).execute()

    if not response.data:
        raise RuntimeError(f"Failed to insert replay row: {response}")

    row_id = response.data[0]["id"]
    print(f"[supabase] Done. Row ID: {row_id}")

    return {"id": row_id, **urls}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a finished MediverseVR job to Supabase.",
    )
    parser.add_argument("--job-dir", "-j", required=True,
        help="Path to the job directory containing the MP4 and other artifacts.")
    parser.add_argument("--language", "-l", default="en",
        help="Language code (en, zh, hi, ...). Default: en")
    parser.add_argument("--title", "-t", default=None,
        help="Human-readable title for this replay. Auto-generated if omitted.")
    parser.add_argument("--video-filename", default=None,
        help="MP4 filename inside --job-dir. Auto-detects narrated_*.mp4 if omitted.")
    args = parser.parse_args()

    try:
        result = upload_job(
            job_dir        = args.job_dir,
            language       = args.language,
            title          = args.title,
            video_filename = args.video_filename,
        )
    except Exception as exc:
        print(f"\n✖  Upload failed: {exc}", file=sys.stderr)
        return 1

    print("\n✅ Replay uploaded:")
    print(f"   Row ID:    {result['id']}")
    print(f"   Video URL: {result.get('video_url')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

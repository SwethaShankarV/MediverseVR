"""
firebase_uploader.py
────────────────────
Uploads a completed pipeline job to Firebase Storage and records
replay metadata in Firestore.

Called automatically by server.py after the pipeline exits with code 0.
Safe no-op when Firebase env vars are not set — local dev still works fine.

Requirements
────────────
  pip install firebase-admin

Environment variables (set in Render dashboard, never commit to GitHub)
────────────────────────────────────────────────────────────────────────
  FIREBASE_PROJECT_ID
      e.g.  mediversevr-12345

  FIREBASE_STORAGE_BUCKET
      e.g.  mediversevr-12345.appspot.com

  FIREBASE_SERVICE_ACCOUNT_JSON   (recommended for Render)
      The entire contents of your service account JSON file, pasted as one
      long string into the Render env-var field.
      Get it from: Firebase Console → Project Settings → Service Accounts →
      Generate new private key.

  GOOGLE_APPLICATION_CREDENTIALS  (alternative for local dev)
      Path to the service account JSON file on disk.
      e.g.  /Users/yourname/firebase-admin-key.json

Firebase layout after upload
────────────────────────────
  Storage:
    replays/{job_id}/surgery_events.json
    replays/{job_id}/final_mapped.json
    replays/{job_id}/audio_manifest.json
    replays/{job_id}/audio_steps/step_0000_*.wav  (one per narration sentence)
    replays/{job_id}/captions.srt
    replays/{job_id}/narrated.mp4                 (if video was composed)

  Firestore document  replays/{job_id}:
    {
      job_id, uploaded_at, title, duration_seconds,
      events_url, final_mapped_url, audio_manifest_url,
      audio_urls: [...],  srt_url,  video_url
    }

CLI (manual test)
─────────────────
  python firebase_uploader.py --job-id <id> --job-dir ./jobs/<id>
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ── Firebase availability ──────────────────────────────────────────────────────

def firebase_configured() -> bool:
    """Return True if the minimum Firebase env vars are present."""
    has_project = bool(os.environ.get("FIREBASE_PROJECT_ID"))
    has_bucket  = bool(os.environ.get("FIREBASE_STORAGE_BUCKET"))
    has_creds   = bool(
        os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    return has_project and has_bucket and has_creds


def _init_firebase() -> None:
    """Initialise the firebase-admin SDK app (idempotent — safe to call multiple times)."""
    import firebase_admin  # noqa: PLC0415
    from firebase_admin import credentials  # noqa: PLC0415

    if firebase_admin._apps:
        return  # already initialised

    sa_json_str = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if sa_json_str:
        # Render-friendly path: env var contains the raw JSON string
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        tmp.write(sa_json_str)
        tmp.close()
        cred = credentials.Certificate(tmp.name)
    else:
        # Local dev path: env var points to a JSON file on disk
        cred = credentials.Certificate(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )

    firebase_admin.initialize_app(cred, {
        "storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"],
    })


# ── Main upload function ───────────────────────────────────────────────────────

def upload_job(job_id: str, job_dir: Path) -> Dict:
    """
    Upload all output files from job_dir to Firebase Storage and write a
    Firestore document with download URLs.

    Parameters
    ----------
    job_id  : The UUID for this pipeline job.
    job_dir : Local directory containing all pipeline outputs.

    Returns
    -------
    dict    : Field name → download URL (e.g. {"video_url": "https://..."}).
              Returns {} (empty) if Firebase is not configured or upload fails.
    """
    if not firebase_configured():
        print("[firebase] Not configured — skipping upload. "
              "Set FIREBASE_PROJECT_ID, FIREBASE_STORAGE_BUCKET, and credentials.")
        return {}

    try:
        _init_firebase()
    except ImportError:
        print("[firebase] firebase-admin package not installed — skipping upload. "
              "Run: pip install firebase-admin")
        return {}
    except Exception as exc:
        print(f"[firebase] Initialisation failed — skipping upload. Error: {exc}")
        return {}

    from firebase_admin import firestore, storage  # noqa: PLC0415

    bucket = storage.bucket()
    prefix = f"replays/{job_id}"
    urls: Dict = {}
    audio_urls: List[str] = []

    def _upload(local_path: Path, remote_path: str) -> str:
        """Upload one file and return its public download URL."""
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(str(local_path))
        blob.make_public()
        print(f"[firebase]   ✔  {remote_path}")
        return blob.public_url

    print(f"[firebase] Uploading job {job_id} → gs://{os.environ['FIREBASE_STORAGE_BUCKET']}/{prefix}/")

    # surgery_events.json
    _try_upload(job_dir / "surgery_events.json",
                f"{prefix}/surgery_events.json", _upload, urls, "events_url")

    # final_mapped.json (narration text + timestamps — needed for VR replay)
    _try_upload(job_dir / "final_mapped.json",
                f"{prefix}/final_mapped.json", _upload, urls, "final_mapped_url")

    # audio_manifest.json (links WAVs to timestamps)
    _try_upload(job_dir / "audio_manifest.json",
                f"{prefix}/audio_manifest.json", _upload, urls, "audio_manifest_url")

    # audio_steps/*.wav — sorted so indices are stable
    audio_dir = job_dir / "audio_steps"
    if audio_dir.exists():
        for wav in sorted(audio_dir.glob("*.wav")):
            url = _upload(wav, f"{prefix}/audio_steps/{wav.name}")
            audio_urls.append(url)
    urls["audio_urls"] = audio_urls

    # captions.srt
    _try_upload(job_dir / "captions.srt",
                f"{prefix}/captions.srt", _upload, urls, "srt_url")

    # narrated MP4 — find whichever narrated_*.mp4 exists
    mp4_files = sorted(job_dir.glob("narrated_*.mp4"))
    if mp4_files:
        urls["video_url"] = _upload(mp4_files[0], f"{prefix}/narrated.mp4")
    else:
        print("[firebase]   (no narrated MP4 found — video step may have been skipped)")

    # ── Write Firestore metadata document ─────────────────────────────────────
    doc = {
        "job_id":           job_id,
        "uploaded_at":      datetime.now(timezone.utc).isoformat(),
        "title":            _extract_title(job_dir),
        "duration_seconds": _extract_duration(job_dir),
        **{k: v for k, v in urls.items() if k != "audio_urls"},
        "audio_urls":       audio_urls,
    }

    db = firestore.client()
    db.collection("replays").document(job_id).set(doc)
    print(f"[firebase] Firestore document written: replays/{job_id}")
    print(f"[firebase] Upload complete — {len(audio_urls)} audio files + "
          f"{sum(1 for k in urls if k not in ('audio_urls',) and urls[k])} other files.")

    return urls


def _try_upload(local_path: Path, remote_path: str, upload_fn, urls: dict, key: str) -> None:
    """Upload a file if it exists; silently skip if not."""
    if local_path.exists():
        urls[key] = upload_fn(local_path, remote_path)
    else:
        print(f"[firebase]   (skipping {local_path.name} — not found)")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_title(job_dir: Path) -> str:
    """Derive a human-readable session title from surgery_events.json."""
    try:
        data  = json.loads((job_dir / "surgery_events.json").read_text())
        items = data.get("data") or data.get("Items") or []
        if items:
            first  = items[0]
            tool   = first.get("tool", "")
            target = first.get("target", "")
            if tool and target:
                return f"{tool} on {target}"
    except Exception:
        pass
    return "Surgery Session"


def _extract_duration(job_dir: Path) -> float:
    """Return total narration duration in seconds from audio_manifest.json."""
    try:
        manifest = json.loads((job_dir / "audio_manifest.json").read_text())
        steps    = manifest.get("steps", [])
        if steps:
            last = steps[-1]
            ts   = float(last.get("source_times", {}).get("timestamp", 0))
            dur  = float(last.get("duration_seconds", 0))
            return round(ts + dur, 2)
    except Exception:
        pass
    return 0.0


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Manually upload a completed pipeline job to Firebase."
    )
    parser.add_argument("--job-id",  required=True, help="Job UUID")
    parser.add_argument("--job-dir", required=True, help="Path to the job directory")
    args = parser.parse_args()

    result = upload_job(args.job_id, Path(args.job_dir))
    print("\nDownload URLs:")
    print(json.dumps(result, indent=2))

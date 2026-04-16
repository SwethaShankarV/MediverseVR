"""
run_pipeline.py
───────────────
Single-command orchestrator for the MediverseVR narration pipeline.

Runs all four steps in order for a new VR session recording:

  Step 1 ── generate_narration.py
            Events JSON → BioMistral-7B → final_mapped.json
            (skipped if final_mapped.json already exists and --force not set)

  Step 2 ── generate_step_audio.py
            final_mapped.json → Coqui TTS → audio_steps/*.wav + audio_manifest.json
            (skipped if audio_manifest.json already exists and --force not set)

  Step 3 ── generate_srt.py
            audio_manifest.json → captions.srt
            (skipped if captions.srt already exists and --force not set)

  Step 4 ── compose_video.py
            video + audio_steps/ + captions.srt → narrated_<name>.mp4
            (always runs; re-compose is fast compared to LLM/TTS)

Usage examples
──────────────
  # Full pipeline from scratch
  python run_pipeline.py --video recording.mp4 --events surgery_events.json

  # Skip narration generation (use existing final_mapped.json)
  python run_pipeline.py --video recording.mp4 --skip-narration

  # Force every step to re-run even if outputs exist
  python run_pipeline.py --video recording.mp4 --events surgery_events.json --force

  # Run only up to SRT generation (no video composition)
  python run_pipeline.py --events surgery_events.json --no-video

  # Quick demo: skip LLM and TTS (use existing data), just compose the video
  python run_pipeline.py --video recording.mp4 --skip-narration --skip-tts

  # Pass extra flags through to individual steps
  python run_pipeline.py --video recording.mp4 --narration-volume 0.8 --soft-subs

Output
──────
  final_mapped.json         — narration sentences + timestamps
  audio_steps/              — one WAV per sentence
  audio_manifest.json       — metadata linking WAVs to timestamps
  captions.srt              — subtitle file
  narrated_<video>.mp4      — final deliverable with audio + captions
"""

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional


# ── Colour helpers (no dependencies) ─────────────────────────────────────────

USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOUR else text

def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def red(t):    return _c("31", t)


# ── Step runner ───────────────────────────────────────────────────────────────

def run_step(
    step_num:   int,
    step_name:  str,
    cmd:        List[str],
    skip:       bool = False,
    skip_reason: str = "",
) -> bool:
    """
    Print a header, optionally skip, or run the command.

    Returns True on success, False on failure.
    Prints elapsed time on completion.
    """
    header = bold(f"\n{'─'*60}\n  Step {step_num}: {step_name}\n{'─'*60}")
    print(header)

    if skip:
        print(yellow(f"  ⏭  Skipped — {skip_reason}"))
        return True

    print(cyan(f"  $ {' '.join(cmd)}\n"))
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(red(f"\n  ✖  Step {step_num} failed (exit code {result.returncode})."))
        return False

    print(green(f"\n  ✔  Step {step_num} complete  ({elapsed:.1f}s)"))
    return True


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="MediverseVR end-to-end narration pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Core I/O ─────────────────────────────────────────────────────
    parser.add_argument("--video", "-v",
        default=None,
        help="Path to the recorded VR session video. Required unless --no-video is set.")
    parser.add_argument("--events", "-e",
        default=os.path.join(here, "surgery_events.json"),
        help="Path to the VR event log JSON (default: %(default)s)")
    parser.add_argument("--output", "-o",
        default=None,
        help="Output MP4 path (default: narrated_<video_name>.mp4 beside the video).")
    parser.add_argument("--work-dir", "-w",
        default=here,
        help="Working directory for all intermediate files (default: script folder).")

    # ── Step skipping ─────────────────────────────────────────────────
    skip_group = parser.add_argument_group("step skipping")
    skip_group.add_argument("--skip-narration", action="store_true",
        help="Skip Step 1 (narration generation). Uses existing final_mapped.json.")
    skip_group.add_argument("--skip-tts", action="store_true",
        help="Skip Step 2 (TTS audio generation). Uses existing audio_manifest.json + WAVs.")
    skip_group.add_argument("--skip-srt", action="store_true",
        help="Skip Step 3 (SRT generation). Uses existing captions.srt.")
    skip_group.add_argument("--no-video", action="store_true",
        help="Stop after Step 3 — do not run Step 4 (video composition).")
    skip_group.add_argument("--force", "-f", action="store_true",
        help="Re-run every step even if its output already exists.")

    # ── Step 1 passthrough flags ──────────────────────────────────────
    narr_group = parser.add_argument_group("narration (Step 1)")
    narr_group.add_argument("--model",
        default="BioMistral/BioMistral-7B",
        help="BioMistral model name or local path (default: %(default)s)")
    narr_group.add_argument("--offload-folder",
        default=os.path.join(here, "offload"),
        help="CPU offload folder for BioMistral weights (default: %(default)s)")
    narr_group.add_argument("--max-new-tokens", type=int, default=300,
        help="Max new tokens for BioMistral generation (default: 300)")
    narr_group.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature for BioMistral (default: 0.7)")
    narr_group.add_argument("--skip-hallucination-check", action="store_true",
        help="Skip hallucination detection in Step 1.")

    # ── Step 2 passthrough flags ──────────────────────────────────────
    tts_group = parser.add_argument_group("TTS audio (Step 2)")
    tts_group.add_argument("--tts-model",
        default="tts_models/en/ljspeech/glow-tts",
        help="Coqui TTS model name (default: %(default)s)")
    tts_group.add_argument("--tts-gpu", action="store_true",
        help="Use GPU for Coqui TTS if available.")

    # ── Step 4 passthrough flags ──────────────────────────────────────
    vid_group = parser.add_argument_group("video composition (Step 4)")
    vid_group.add_argument("--narration-volume", type=float, default=1.0,
        metavar="0.0-2.0",
        help="Volume multiplier for narration audio in the output video (default: 1.0).")
    vid_group.add_argument("--soft-subs", action="store_true",
        help="Attach captions as a soft subtitle track instead of burning them in.")

    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def py() -> str:
    """Return the Python interpreter that is currently running this script."""
    return sys.executable


def exists_and_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def step_output_exists(path: str, label: str, force: bool) -> Optional[str]:
    """
    Return a skip reason string if the output file already exists and --force
    was not requested, otherwise return None (meaning: do run the step).
    """
    if not force and exists_and_nonempty(path):
        return f"{label} already exists. Use --force to regenerate."
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    wd   = os.path.abspath(args.work_dir)

    # ── Resolve intermediate file paths ──────────────────────────────
    final_mapped   = os.path.join(wd, "final_mapped.json")
    audio_manifest = os.path.join(wd, "audio_manifest.json")
    audio_dir      = os.path.join(wd, "audio_steps")
    captions_srt   = os.path.join(wd, "captions.srt")

    # ── Banner ────────────────────────────────────────────────────────
    print(bold("\n" + "═"*60))
    print(bold("  MediverseVR — Full Narration Pipeline"))
    print(bold("═"*60))

    print(f"\n  Work dir:  {wd}")
    print(f"  Events:    {args.events}")
    if args.video:
        print(f"  Video:     {args.video}")
    if args.output:
        print(f"  Output:    {args.output}")
    print(f"  Force:     {'yes' if args.force else 'no (skip steps whose output exists)'}")

    # ── Pre-flight checks ─────────────────────────────────────────────
    print(bold("\n  Pre-flight checks"))

    errors = []

    if not args.skip_narration and not os.path.exists(args.events):
        errors.append(f"Events file not found: {args.events}")

    if not args.no_video:
        if not args.video:
            errors.append("--video is required unless --no-video is set.")
        elif not os.path.exists(args.video):
            errors.append(f"Video file not found: {args.video}")

    if errors:
        for e in errors:
            print(red(f"  ✖  {e}"))
        sys.exit(1)

    print(green("  ✔  All inputs present."))

    # ── Track outcomes ────────────────────────────────────────────────
    results = {}   # step_name → True/False

    pipeline_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — Generate narration text (BioMistral)
    # ══════════════════════════════════════════════════════════════════
    skip_reason = None
    if args.skip_narration:
        skip_reason = "--skip-narration flag set"
        if not exists_and_nonempty(final_mapped):
            print(red(f"\n  ✖  --skip-narration set but {final_mapped} not found."))
            sys.exit(1)
    else:
        skip_reason = step_output_exists(final_mapped, "final_mapped.json", args.force)

    cmd1 = [
        py(), os.path.join(here, "generate_narration.py"),
        "--events",          args.events,
        "--output",          final_mapped,
        "--model",           args.model,
        "--offload-folder",  args.offload_folder,
        "--max-new-tokens",  str(args.max_new_tokens),
        "--temperature",     str(args.temperature),
    ]
    if args.skip_hallucination_check:
        cmd1.append("--skip-hallucination-check")

    ok = run_step(1, "Generate narration text  (BioMistral-7B)", cmd1,
                  skip=skip_reason is not None, skip_reason=skip_reason or "")
    results["narration"] = ok
    if not ok:
        print(red("\n  Pipeline aborted at Step 1."))
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 — Text-to-speech  (Coqui TTS)
    # ══════════════════════════════════════════════════════════════════
    skip_reason = None
    if args.skip_tts:
        skip_reason = "--skip-tts flag set"
        if not exists_and_nonempty(audio_manifest):
            print(red(f"\n  ✖  --skip-tts set but {audio_manifest} not found."))
            sys.exit(1)
    else:
        skip_reason = step_output_exists(audio_manifest, "audio_manifest.json", args.force)

    cmd2 = [
        py(), os.path.join(here, "generate_step_audio.py"),
        "--input",            final_mapped,
        "--output-manifest",  audio_manifest,
        "--audio-dir",        audio_dir,
        "--model",            args.tts_model,
    ]
    if args.tts_gpu:
        cmd2.append("--gpu")
    if args.force:
        cmd2.append("--force")

    ok = run_step(2, "Text-to-speech audio generation  (Coqui TTS)", cmd2,
                  skip=skip_reason is not None, skip_reason=skip_reason or "")
    results["tts"] = ok
    if not ok:
        print(red("\n  Pipeline aborted at Step 2."))
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 — Generate SRT subtitles
    # ══════════════════════════════════════════════════════════════════
    skip_reason = None
    if args.skip_srt:
        skip_reason = "--skip-srt flag set"
    else:
        skip_reason = step_output_exists(captions_srt, "captions.srt", args.force)

    cmd3 = [
        py(), os.path.join(here, "generate_srt.py"),
        "--manifest", audio_manifest,
        "--fallback", final_mapped,
        "--output",   captions_srt,
    ]

    ok = run_step(3, "Generate SRT subtitle file", cmd3,
                  skip=skip_reason is not None, skip_reason=skip_reason or "")
    results["srt"] = ok
    if not ok:
        print(red("\n  Pipeline aborted at Step 3."))
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 — Compose final video
    # ══════════════════════════════════════════════════════════════════
    if args.no_video:
        print(bold(f"\n{'─'*60}\n  Step 4: Video composition\n{'─'*60}"))
        print(yellow("  ⏭  Skipped — --no-video flag set"))
        results["video"] = True
    else:
        cmd4 = [
            py(), os.path.join(here, "compose_video.py"),
            "--video",    args.video,
            "--manifest", audio_manifest,
            "--srt",      captions_srt,
            "--narration-volume", str(args.narration_volume),
        ]
        if args.output:
            cmd4 += ["--output", args.output]
        if args.soft_subs:
            cmd4.append("--soft-subs")

        ok = run_step(4, "Compose narrated video  (ffmpeg)", cmd4)
        results["video"] = ok
        if not ok:
            print(red("\n  Pipeline aborted at Step 4."))
            sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - pipeline_start

    print(bold(f"\n{'═'*60}"))
    print(bold("  Pipeline Summary"))
    print(bold(f"{'═'*60}"))

    step_labels = [
        ("narration", "Step 1 — Narration generation"),
        ("tts",       "Step 2 — TTS audio"),
        ("srt",       "Step 3 — SRT subtitles"),
        ("video",     "Step 4 — Video composition"),
    ]
    for key, label in step_labels:
        outcome = results.get(key)
        if outcome is None:
            icon = yellow("  ⏭  skipped")
        elif outcome:
            icon = green("  ✔  done   ")
        else:
            icon = red("  ✖  failed ")
        print(f"{icon}   {label}")

    # Resolve final output path for display
    if not args.no_video and args.video:
        if args.output:
            out_path = args.output
        else:
            base    = os.path.splitext(os.path.basename(args.video))[0]
            out_dir = os.path.dirname(os.path.abspath(args.video))
            out_path = os.path.join(out_dir, f"narrated_{base}.mp4")

        if exists_and_nonempty(out_path):
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(green(f"\n  🎬 Output video: {out_path}  ({size_mb:.1f} MB)"))
        else:
            print(yellow(f"\n  🎬 Expected output: {out_path}  (not found)"))

    print(f"\n  Total time: {total_elapsed:.1f}s")
    print(bold("═"*60 + "\n"))


if __name__ == "__main__":
    main()

"""
generate_srt.py
───────────────
Converts MediverseVR narration data into a standard SRT subtitle file.

Two input modes (tried in this order):
  1. audio_manifest.json  — preferred; uses actual audio duration for precise
                            end-times so captions disappear when speech stops.
  2. final_mapped.json    — fallback; estimates duration from text length when
                            no manifest is available.

Usage examples
──────────────
  # Default paths (manifest + srt in the same folder as this script)
  python generate_srt.py

  # Explicit paths
  python generate_srt.py --manifest audio_manifest.json --output captions.srt

  # Force fallback to final_mapped.json even if manifest exists
  python generate_srt.py --no-manifest --fallback final_mapped.json --output captions.srt

Output
──────
  captions.srt  — standard subtitle file readable by VLC, ffmpeg, browsers, etc.

Overlap prevention
──────────────────
If a TTS clip takes longer to speak than the real-time gap between surgical
events, captions would visually overlap. This script automatically shifts
later captions so no two ever appear simultaneously. Mirrors the same
overlap prevention in compose_video.py so audio and captions stay in sync.
"""

import argparse
import json
import os
import sys


# ── Helpers ──────────────────────────────────────────────────────────────────

def seconds_to_srt_time(seconds: float) -> str:
    """Convert a float number of seconds to SRT timestamp format HH:MM:SS,mmm."""
    total_ms   = int(round(seconds * 1000))
    ms         = total_ms % 1000
    total_secs = total_ms // 1000
    secs       = total_secs % 60
    total_mins = total_secs // 60
    mins       = total_mins % 60
    hours      = total_mins // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def estimate_duration(text: str) -> float:
    """
    Rough speech duration estimate when no audio file is available.
    Average English speech rate ≈ 150 words/min → ~0.4 s/word.
    Minimum 2 s, maximum 15 s.
    """
    word_count = len(text.split())
    estimated  = word_count * 0.4
    return max(2.0, min(estimated, 15.0))


def repack_to_prevent_overlap(entries: list) -> list:
    """
    If a caption is still displayed when the next one starts, push the next
    caption (and all subsequent ones) back so no two captions ever overlap.

    Mirrors the audio-overlap prevention in compose_video.py (GAP_MS=300)
    so captions stay in sync with the repacked audio clips.
    """
    GAP_S = 3  # matches GAP_MS=300 in compose_video.py
    for i in range(len(entries) - 1):
        min_next_start = entries[i]["end"] + GAP_S
        if entries[i + 1]["start"] < min_next_start:
            original_start = entries[i + 1]["start"]
            duration       = entries[i + 1]["end"] - entries[i + 1]["start"]
            entries[i + 1]["start"] = min_next_start
            entries[i + 1]["end"]   = min_next_start + duration
            print(f"  ⚠  Caption {i+2} shifted from {original_start:.2f}s "
                  f"to {min_next_start:.2f}s to avoid overlap with caption {i+1}")
    return entries


def build_srt_block(index: int, start_s: float, end_s: float, text: str) -> str:
    """Return one complete SRT block as a string (no trailing newline)."""
    return (
        f"{index}\n"
        f"{seconds_to_srt_time(start_s)} --> {seconds_to_srt_time(end_s)}\n"
        f"{text.strip()}"
    )


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_from_manifest(manifest_path: str) -> list:
    """
    Parse audio_manifest.json and return a list of subtitle entries:
      [{"start": float, "end": float, "text": str}, ...]

    Uses source_times.timestamp as the start and
    (timestamp + duration_seconds) as the end — the most accurate option
    because it reflects exactly how long the TTS clip actually plays.

    Applies overlap prevention so captions stay synced with the audio
    composed by compose_video.py.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    steps = manifest.get("steps", [])
    if not steps:
        raise ValueError(f"No steps found in {manifest_path}")

    entries = []
    for step in steps:
        # Skip steps that failed during TTS generation
        if step.get("error"):
            print(f"  ⚠  Skipping step {step.get('step_id', '?')} — TTS error: {step['error']}")
            continue

        source_times = step.get("source_times", {})
        start        = source_times.get("timestamp", source_times.get("start_time", 0.0))
        duration     = step.get("duration_seconds", 0.0)
        end          = start + duration if duration > 0 else start + estimate_duration(step.get("text", ""))
        text         = step.get("text", "").strip()

        if not text:
            print(f"  ⚠  Skipping step {step.get('step_id', '?')} — empty text.")
            continue

        entries.append({"start": start, "end": end, "text": text})

    # Prevent captions from overlapping — mirrors compose_video.py audio repack
    entries = repack_to_prevent_overlap(entries)

    return entries


def load_from_final_mapped(fallback_path: str) -> list:
    """
    Parse final_mapped.json and return subtitle entries.

    Uses start_time as the caption start.
    End time = start_time + word-count estimate (final_mapped.json end_time
    is the end of the *surgical action*, not the end of the spoken sentence,
    so it can be much longer than the actual audio clip).
    """
    with open(fallback_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {fallback_path}, got {type(data).__name__}")

    entries = []
    for item in data:
        text  = (item.get("sentence") or item.get("narration") or "").strip()
        start = item.get("start_time", item.get("timestamp", 0.0))
        end   = start + estimate_duration(text)

        if not text:
            continue

        entries.append({"start": start, "end": end, "text": text})

    # Prevent captions from overlapping — mirrors compose_video.py audio repack
    entries = repack_to_prevent_overlap(entries)

    return entries


# ── Writer ────────────────────────────────────────────────────────────────────

def write_srt(entries: list, output_path: str) -> None:
    """Write a list of subtitle entries to an SRT file."""
    blocks = []
    for i, entry in enumerate(entries, start=1):
        blocks.append(build_srt_block(i, entry["start"], entry["end"], entry["text"]))

    # SRT blocks are separated by a blank line; file ends with a trailing newline
    content = "\n\n".join(blocks) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Generate a .srt subtitle file from MediverseVR narration data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest", "-m",
        default=os.path.join(here, "audio_manifest.json"),
        help="Path to audio_manifest.json (default: %(default)s)",
    )
    parser.add_argument(
        "--fallback", "-f",
        default=os.path.join(here, "final_mapped.json"),
        help="Path to final_mapped.json used when manifest is unavailable (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(here, "captions.srt"),
        help="Output .srt file path (default: %(default)s)",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip manifest even if it exists and use final_mapped.json directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  MediverseVR — SRT Subtitle Generator")
    print("=" * 60)

    entries = []

    # ── Try manifest first (most accurate) ──────────────────────────
    if not args.no_manifest and os.path.exists(args.manifest):
        print(f"\n📄 Loading from manifest:  {args.manifest}")
        try:
            entries = load_from_manifest(args.manifest)
            print(f"   ✔  {len(entries)} step(s) loaded from manifest.")
        except Exception as exc:
            print(f"   ✖  Failed to read manifest ({exc}) — falling back to final_mapped.json.")
            entries = []

    # ── Fallback to final_mapped.json ───────────────────────────────
    if not entries:
        if not os.path.exists(args.fallback):
            print(f"\n✖  Neither manifest nor fallback file found.")
            print(f"   Looked for: {args.manifest}")
            print(f"              {args.fallback}")
            sys.exit(1)

        print(f"\n📄 Loading from fallback:  {args.fallback}")
        try:
            entries = load_from_final_mapped(args.fallback)
            print(f"   ✔  {len(entries)} step(s) loaded from final_mapped.json.")
            print(f"   ℹ  Caption end-times are estimated (no audio durations available).")
        except Exception as exc:
            print(f"\n✖  Failed to parse fallback file: {exc}")
            sys.exit(1)

    if not entries:
        print("\n✖  No subtitle entries found. Nothing to write.")
        sys.exit(1)

    # ── Preview table ────────────────────────────────────────────────
    print("\n┌─────┬──────────────────┬──────────────────┬─────────────────────────────────────────────────┐")
    print("│  #  │    Start time    │     End time     │  Text                                           │")
    print("├─────┼──────────────────┼──────────────────┼─────────────────────────────────────────────────┤")
    for i, e in enumerate(entries, start=1):
        text_preview = e["text"][:47] + "…" if len(e["text"]) > 48 else e["text"]
        print(f"│ {i:<3} │ {seconds_to_srt_time(e['start']):<16} │ {seconds_to_srt_time(e['end']):<16} │ {text_preview:<47} │")
    print("└─────┴──────────────────┴──────────────────┴─────────────────────────────────────────────────┘")

    # ── Write SRT ────────────────────────────────────────────────────
    print(f"\n💾 Writing SRT → {args.output}")
    write_srt(entries, args.output)
    print(f"   ✔  captions.srt written ({len(entries)} subtitle block(s)).")
    print("\n✅ Done! Open the .srt file in any text editor to verify, or")
    print("   test it with:  ffplay -vf subtitles=captions.srt <your_video.mp4>")
    print()


if __name__ == "__main__":
    main()
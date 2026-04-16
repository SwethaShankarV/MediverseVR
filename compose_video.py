"""
compose_video.py
────────────────
Merges a recorded VR session video with MediverseVR narration audio and
SRT captions into a single, self-contained MP4 file.

What it does
────────────
  1. Reads audio_manifest.json to find each WAV clip and its timestamp offset.
  2. Builds an ffmpeg filter_complex that:
       • Places every WAV clip at the correct millisecond offset via adelay
       • Mixes all delayed clips together with amix
       • Burns the SRT captions into the video using the subtitles filter
         (requires libass). If libass is not available, falls back to
         drawtext filters (no external library needed).
  3. Re-encodes to H.264 video + AAC audio → standard MP4 readable everywhere.

If the input video has no audio track the narration becomes the only audio.
If it already has audio the narration is mixed on top at a controllable volume.

Usage examples
──────────────
  # All defaults (looks for recording.mp4 in the same folder)
  python compose_video.py --video recording.mp4

  # Explicit paths
  python compose_video.py \\
      --video      session.mp4 \\
      --manifest   audio_manifest.json \\
      --srt        captions.srt \\
      --output     narrated_session.mp4

  # Lower the narration volume relative to original video audio
  python compose_video.py --video session.mp4 --narration-volume 0.8

  # Attach captions as a soft subtitle track (toggle on/off in VLC)
  python compose_video.py --video session.mp4 --soft-subs

Output
──────
  narrated_<input_video_name>.mp4  (default)
  or whatever path you pass to --output
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import List, Optional


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    """Exit with a helpful message if ffmpeg is not on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("✖  ffmpeg not found. Install it with:  brew install ffmpeg")
        sys.exit(1)


def ffmpeg_caption_mode() -> str:
    """
    Detect the best available caption-burning method for this ffmpeg build.

    Returns one of:
      "libass"    — subtitles filter available (best quality, needs libass)
      "drawtext"  — drawtext filter available (no libass, needs freetype)
      "soft"      — neither hard-sub filter available; will use mov_text track
      "none"      — no caption support at all
    """
    result = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
    filters = result.stdout
    if "subtitles" in filters:
        return "libass"
    if "drawtext" in filters:
        return "drawtext"
    # Check for mov_text codec (soft subs)
    codecs = subprocess.run(["ffmpeg", "-codecs"], capture_output=True, text=True)
    if "mov_text" in codecs.stdout:
        return "soft"
    return "none"


def video_has_audio(video_path: str) -> bool:
    """Return True if the video file contains at least one audio stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def get_video_duration(video_path: str) -> float:
    """Return the video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# ── Manifest loader ───────────────────────────────────────────────────────────

def load_audio_steps(manifest_path: str, base_dir: str) -> List[dict]:
    """
    Parse audio_manifest.json and return a list of audio step dicts:
      [{"path": str, "delay_ms": int, "text": str}, ...]

    Resolves relative audio_path values against base_dir (the folder that
    contains the manifest).
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    steps = manifest.get("steps", [])
    if not steps:
        raise ValueError(f"No steps found in {manifest_path}")

    result = []
    for step in steps:
        if step.get("error"):
            print(f"  ⚠  Skipping step {step.get('step_id','?')} (TTS error: {step['error']})")
            continue

        audio_path = step.get("audio_path", "")
        if not audio_path:
            print(f"  ⚠  Skipping step {step.get('step_id','?')} — no audio_path.")
            continue

        # Resolve relative paths against the manifest's own folder
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(base_dir, audio_path)

        if not os.path.exists(audio_path):
            print(f"  ⚠  Audio file not found, skipping: {audio_path}")
            continue

        source_times = step.get("source_times", {})
        timestamp_s  = source_times.get("timestamp", source_times.get("start_time", 0.0))
        delay_ms     = max(0, int(round(timestamp_s * 1000)))

        result.append({
            "path":     audio_path,
            "delay_ms": delay_ms,
            "text":     step.get("text", ""),
        })

    return result


# ── SRT parser (for drawtext fallback) ───────────────────────────────────────

def srt_time_to_seconds(ts: str) -> float:
    """Convert SRT timestamp string '00:00:05,561' to float seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def parse_srt(srt_path: str) -> List[dict]:
    """
    Parse an SRT file into a list of caption dicts:
      [{"start": float, "end": float, "text": str}, ...]
    """
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on blank lines to get individual blocks
    blocks = re.split(r"\n\s*\n", content.strip())
    captions = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        # Line 0: index number, Line 1: timecodes, Line 2+: text
        timecode_line = lines[1]
        match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            timecode_line,
        )
        if not match:
            continue
        start = srt_time_to_seconds(match.group(1))
        end   = srt_time_to_seconds(match.group(2))
        text  = " ".join(lines[2:]).strip()
        captions.append({"start": start, "end": end, "text": text})

    return captions


def escape_drawtext(text: str) -> str:
    """
    Escape special characters for ffmpeg's drawtext filter.
    The filter uses ':' as option separator and '\\' for escaping.
    Single quotes also need escaping inside the filter string.
    """
    text = text.replace("\\", "\\\\")  # backslash first
    text = text.replace("'",  "\u2019")  # replace smart-quote to avoid shell issues
    text = text.replace(":",  "\\:")
    text = text.replace("[",  "\\[")
    text = text.replace("]",  "\\]")
    return text


def build_drawtext_filters(captions: List[dict]) -> str:
    """
    Build a chained drawtext filter string for all captions.
    Each caption is shown only during its time window via enable='between(t,s,e)'.
    Captions are white text with a semi-transparent black shadow, centred
    near the bottom of the frame.
    """
    parts = []
    for cap in captions:
        safe_text = escape_drawtext(cap["text"])
        # fontsize=28, white text, black outline for readability on any background
        part = (
            f"drawtext=text='{safe_text}'"
            f":enable='between(t,{cap['start']:.3f},{cap['end']:.3f})'"
            f":fontsize=28"
            f":fontcolor=white"
            f":borderw=2"
            f":bordercolor=black"
            f":x=(w-text_w)/2"
            f":y=h-60"
        )
        parts.append(part)

    # Chain all drawtext filters on [0:v]
    return "[0:v]" + ",".join(parts) + "[vout]"


# ── ffmpeg command builder ────────────────────────────────────────────────────

def build_ffmpeg_command(
    video_path:       str,
    audio_steps:      List[dict],
    srt_path:         Optional[str],
    output_path:      str,
    narration_volume: float,
    soft_subs:        bool,
    has_video_audio:  bool,
    caption_mode:     str,          # "libass" | "drawtext" | "soft" | "none"
) -> List[str]:
    """
    Construct the full ffmpeg command as a list of strings.

    filter_complex design
    ─────────────────────
    Input indices:
      [0]        → video file  (video stream [0:v], optional audio [0:a])
      [1]..[N]   → one WAV per narration step

    Audio chain:
      Each WAV [k:a] is delayed by delay_ms ms using adelay,
      then all delayed streams are mixed with amix. If the video already
      has audio it is mixed in too at its natural volume.

    Video caption strategy (in order of preference):
      libass     →  subtitles=filename=... filter (best quality, requires libass)
      drawtext   →  chained drawtext filters (requires freetype, no libass needed)
      soft       →  no video filter; SRT attached as selectable mov_text track
      none       →  video passes through unmodified (captions omitted entirely)
    """
    n = len(audio_steps)

    # Effective caption mode (--soft-subs flag overrides auto-detected mode)
    effective_mode = "soft" if (soft_subs and srt_path) else caption_mode

    # ── Collect ALL inputs first (ffmpeg requires all -i before any -map) ──
    # Input 0     : video
    # Inputs 1..N : WAV clips
    # Input N+1   : SRT (only when using soft subs — must be declared up front)
    cmd = ["ffmpeg", "-y"]
    cmd += ["-i", video_path]
    for step in audio_steps:
        cmd += ["-i", step["path"]]

    srt_input_idx = None
    if srt_path and effective_mode == "soft":
        srt_input_idx = n + 1
        cmd += ["-i", srt_path]

    # ── filter_complex ───────────────────────────────────────────────
    filters = []

    # 1. Delay + volume each narration clip
    delayed_labels = []
    for k, step in enumerate(audio_steps):
        input_idx = k + 1
        label     = f"a{k}"
        delay_ms  = step["delay_ms"]
        filters.append(
            f"[{input_idx}:a]adelay={delay_ms}|{delay_ms},"
            f"volume={narration_volume}[{label}]"
        )
        delayed_labels.append(f"[{label}]")

    # 2. Mix narration + original audio (if present)
    if has_video_audio:
        mix_inputs = "[0:a]" + "".join(delayed_labels)
        n_mix      = n + 1
    else:
        mix_inputs = "".join(delayed_labels)
        n_mix      = n

    filters.append(f"{mix_inputs}amix=inputs={n_mix}:normalize=0[aout]")

    # 3. Video caption strategy
    if srt_path and effective_mode == "libass":
        escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        filters.append(f"[0:v]subtitles=filename={escaped}[vout]")
        video_map = "[vout]"

    elif srt_path and effective_mode == "drawtext":
        captions = parse_srt(srt_path)
        if captions:
            filters.append(build_drawtext_filters(captions))
            video_map = "[vout]"
        else:
            video_map = "0:v"

    else:
        # soft or none — video passes straight through
        video_map = "0:v"

    cmd += ["-filter_complex", "; ".join(filters)]
    cmd += ["-map", video_map]
    cmd += ["-map", "[aout]"]

    # 4. Map soft subtitle stream (input already declared above)
    if srt_input_idx is not None:
        cmd += ["-map", f"{srt_input_idx}:s"]
        cmd += ["-c:s", "mov_text"]

    # ── Output encoding ──────────────────────────────────────────────
    cmd += [
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ]

    return cmd


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Compose a narrated VR surgery video with audio and captions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video",  "-v", required=True,
        help="Path to the recorded VR session video.")
    parser.add_argument("--manifest", "-m",
        default=os.path.join(here, "audio_manifest.json"),
        help="Path to audio_manifest.json (default: %(default)s)")
    parser.add_argument("--srt", "-s",
        default=os.path.join(here, "captions.srt"),
        help="Path to captions.srt (default: %(default)s). Skipped if absent.")
    parser.add_argument("--output", "-o", default=None,
        help="Output MP4 path. Default: narrated_<input_name>.mp4 beside the input.")
    parser.add_argument("--narration-volume", type=float, default=1.0, metavar="0.0-2.0",
        help="Volume multiplier for narration audio (default: 1.0).")
    parser.add_argument("--soft-subs", action="store_true",
        help="Attach captions as a soft (selectable) subtitle track instead of burning them in.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_ffmpeg()

    here = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  MediverseVR — Video Composer")
    print("=" * 60)

    # ── Validate inputs ──────────────────────────────────────────────
    if not os.path.exists(args.video):
        print(f"\n✖  Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.manifest):
        print(f"\n✖  Manifest not found: {args.manifest}")
        print("   Run generate_step_audio.py first to create it.")
        sys.exit(1)

    srt_path = args.srt if os.path.exists(args.srt) else None
    if not srt_path:
        print(f"\n⚠  SRT file not found ({args.srt}) — captions will be skipped.")
        print("   Run generate_srt.py first to add captions.")

    # ── Resolve output path ──────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        base        = os.path.splitext(os.path.basename(args.video))[0]
        outdir      = os.path.dirname(os.path.abspath(args.video))
        output_path = os.path.join(outdir, f"narrated_{base}.mp4")

    # ── Load audio steps ─────────────────────────────────────────────
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    print(f"\n📄 Manifest:    {args.manifest}")
    try:
        audio_steps = load_audio_steps(args.manifest, manifest_dir)
    except Exception as exc:
        print(f"\n✖  Failed to load manifest: {exc}")
        sys.exit(1)

    if not audio_steps:
        print("\n✖  No valid audio clips found in manifest.")
        sys.exit(1)

    print(f"   ✔  {len(audio_steps)} audio clip(s) loaded.")

    # ── System info ──────────────────────────────────────────────────
    video_duration  = get_video_duration(args.video)
    has_video_audio = video_has_audio(args.video)
    caption_mode    = ffmpeg_caption_mode()

    print(f"\n🎬 Video:       {args.video}")
    print(f"   Duration:   {video_duration:.2f}s")
    print(f"   Has audio:  {'yes' if has_video_audio else 'no'}")

    if srt_path:
        effective_mode = "soft" if args.soft_subs else caption_mode
        mode_labels = {
            "libass":    "hard-burned via libass (best quality)",
            "drawtext":  "hard-burned via drawtext (freetype, no libass needed)",
            "soft":      "soft subtitle track (selectable in VLC / most players)",
            "none":      "⚠  skipped — no caption support in this ffmpeg build",
        }
        print(f"\n📝 Captions:    {srt_path}")
        print(f"   Mode:       {mode_labels.get(effective_mode, effective_mode)}")
        if caption_mode == "none" and not args.soft_subs:
            print("   💡 To enable hard-burned captions: brew reinstall ffmpeg")
    elif not srt_path and not args.soft_subs:
        pass  # already warned above

    print(f"\n🔊 Narration clips to overlay:")
    print(f"   {'Offset':>10}   {'Duration':>10}   File")
    print(f"   {'──────':>10}   {'────────':>10}   ────")
    for step in audio_steps:
        fname    = os.path.basename(step["path"])
        offset_s = step["delay_ms"] / 1000
        dur_result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", step["path"]],
            capture_output=True, text=True,
        )
        try:
            dur_str = f"{float(dur_result.stdout.strip()):.2f}s"
        except ValueError:
            dur_str = "?"
        print(f"   {offset_s:>9.3f}s   {dur_str:>10}   {fname}")

    print(f"\n💾 Output:      {output_path}")

    # ── Build and run ffmpeg ─────────────────────────────────────────
    ffmpeg_cmd = build_ffmpeg_command(
        video_path       = args.video,
        audio_steps      = audio_steps,
        srt_path         = srt_path,
        output_path      = output_path,
        narration_volume = args.narration_volume,
        soft_subs        = args.soft_subs,
        has_video_audio  = has_video_audio,
        caption_mode     = caption_mode,
    )

    print("\n🔧 ffmpeg command:")
    print("   " + " ".join(
        f'"{a}"' if (" " in a or ";" in a) else a
        for a in ffmpeg_cmd
    ))

    print("\n⏳ Composing video... (this may take a moment)\n")
    result = subprocess.run(ffmpeg_cmd)

    if result.returncode != 0:
        print(f"\n✖  ffmpeg exited with code {result.returncode}.")
        sys.exit(result.returncode)

    out_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Done!")
    print(f"   Output: {output_path}  ({out_size:.1f} MB)")
    print(f"   Open in VLC or QuickTime to verify audio sync and captions.")
    print()


if __name__ == "__main__":
    main()

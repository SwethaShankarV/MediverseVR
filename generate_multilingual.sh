#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# generate_multilingual.sh
# MediverseVR — Generate a narrated video in a target language
#
# Usage:
#   ./generate_multilingual.sh zh        # Mandarin
#   ./generate_multilingual.sh hi        # Hindi
#   ./generate_multilingual.sh es        # Spanish
#
# This script does NOT touch your existing English pipeline outputs.
# Everything for the target language is written to:
#   final_mapped_<lang>.json
#   audio_steps_<lang>/
#   audio_manifest_<lang>.json
#   captions_<lang>.srt
#   narrated_<lang>.mp4
#
# Requires:
#   - final_mapped.json (English) already exists
#   - demo_recording.mp4 exists in the same folder
#   - transformers + sentencepiece installed: pip install transformers sentencepiece
# ═══════════════════════════════════════════════════════════════════════════

set -e  # exit on any error

LANG_CODE="${1:-}"
if [ -z "$LANG_CODE" ]; then
    echo "Usage: $0 <language_code>"
    echo "Examples: $0 zh    # Mandarin"
    echo "          $0 hi    # Hindi"
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON=".venv/bin/python"

# ── File paths for this language ─────────────────────────────────────────
FINAL_MAPPED_EN="final_mapped.json"
FINAL_MAPPED_LANG="final_mapped_${LANG_CODE}.json"
AUDIO_DIR="audio_steps_${LANG_CODE}"
AUDIO_MANIFEST="audio_manifest_${LANG_CODE}.json"
CAPTIONS="captions_${LANG_CODE}.srt"
OUTPUT_VIDEO="narrated_${LANG_CODE}.mp4"
VIDEO="demo_recording.mp4"

# ── Sanity checks ────────────────────────────────────────────────────────
if [ ! -f "$FINAL_MAPPED_EN" ]; then
    echo "✖  $FINAL_MAPPED_EN not found. Run the English pipeline first."
    exit 1
fi

if [ ! -f "$VIDEO" ]; then
    echo "✖  $VIDEO not found."
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  MediverseVR — Multilingual Narrated Video  [$LANG_CODE]"
echo "═══════════════════════════════════════════════════════════"

# ── Step A: Translate ────────────────────────────────────────────────────
echo ""
echo "▶ Step A: Translating English → $LANG_CODE"
$PYTHON translate_narration.py \
    --input "$FINAL_MAPPED_EN" \
    --output "$FINAL_MAPPED_LANG" \
    --language "$LANG_CODE"

# ── Step B: TTS in target language ───────────────────────────────────────
echo ""
echo "▶ Step B: Generating $LANG_CODE audio with XTTS v2"
$PYTHON generate_step_audio.py \
    --input "$FINAL_MAPPED_LANG" \
    --output-manifest "$AUDIO_MANIFEST" \
    --audio-dir "$AUDIO_DIR" \
    --language "$LANG_CODE"

# ── Step C: SRT ──────────────────────────────────────────────────────────
echo ""
echo "▶ Step C: Generating SRT subtitles"
$PYTHON generate_srt.py \
    --manifest "$AUDIO_MANIFEST" \
    --fallback "$FINAL_MAPPED_LANG" \
    --output "$CAPTIONS"

# ── Step D: Compose video ────────────────────────────────────────────────
echo ""
echo "▶ Step D: Composing narrated video with ffmpeg"
$PYTHON compose_video.py \
    --video "$VIDEO" \
    --manifest "$AUDIO_MANIFEST" \
    --srt "$CAPTIONS" \
    --output "$OUTPUT_VIDEO" \
    --no-original-audio

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Done!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Output:  $HERE/$OUTPUT_VIDEO"
echo ""
echo "  Open with:"
echo "    open $OUTPUT_VIDEO"
echo ""

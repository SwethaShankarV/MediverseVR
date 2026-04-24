#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# demo_reset.sh
# MediverseVR — Reset to clean demo state
#
# TWO MODES:
#
#   ./demo_reset.sh           FULL reset — deletes ALL outputs including
#                             final_mapped.json. Use this when you have time
#                             to run all 4 steps live (Step 1 takes ~22 min).
#
#   ./demo_reset.sh --quick   QUICK reset — keeps final_mapped.json (AI
#                             narration text already generated). Deletes only
#                             audio, SRT, and video outputs. Steps 2-4 run
#                             live in ~17 seconds. Best for time-limited demos.
#
# RECOMMENDED for tomorrow's demo: run in 2 parts
#   MORNING:  ./demo_reset.sh          (full reset)
#             .venv/bin/python generate_narration.py \
#               --events surgery_events.json --output final_mapped.json
#             (wait ~22 minutes — this generates the AI narration text)
#   DEMO:     ./demo_reset.sh --quick  (keeps AI text, clears audio+video)
#             Press "Generate Narrated Video" in Unity → completes in ~17 sec
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_DIR="/Users/swethashankar/Documents/OPT/MediverseVR"
PERSISTENT_PATH="/Users/swethashankar/Library/Application Support/Easley-Dunn Productions/MedicalVR-src"
EVENTS_DEST="$PERSISTENT_PATH/ToolInteractionLog/DefaultInteractionLog.json"
QUICK_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
    esac
done

echo ""
echo "═══════════════════════════════════════════════"
if [ "$QUICK_MODE" = true ]; then
    echo "  MediverseVR — Demo Reset (QUICK mode)"
    echo "  Keeping final_mapped.json — Steps 2-4 only"
else
    echo "  MediverseVR — Demo Reset (FULL mode)"
    echo "  All outputs cleared — all 4 steps will run"
fi
echo "═══════════════════════════════════════════════"

# ── 1. Clean pipeline outputs ─────────────────────────────────────────────────
echo ""
echo "▶  Cleaning pipeline outputs..."

if [ "$QUICK_MODE" = false ]; then
    rm -f "$PIPELINE_DIR/final_mapped.json"  && echo "   ✔  final_mapped.json removed"
else
    if [ -f "$PIPELINE_DIR/final_mapped.json" ]; then
        echo "   ✔  final_mapped.json kept (AI narration text preserved)"
    else
        echo "   ✖  WARNING: final_mapped.json not found!"
        echo "      Run WITHOUT --quick first, then run generate_narration.py"
        exit 1
    fi
fi

rm -f "$PIPELINE_DIR/audio_manifest.json"   && echo "   ✔  audio_manifest.json removed"
rm -f "$PIPELINE_DIR/captions.srt"          && echo "   ✔  captions.srt removed"
rm -f "$PIPELINE_DIR/audio_steps/"*.wav     2>/dev/null && echo "   ✔  audio WAV files removed"
rm -f "$PIPELINE_DIR/narrated_"*.mp4        2>/dev/null && echo "   ✔  narrated MP4 removed"

# ── 2. Copy surgery_events.json to Unity persistentDataPath ──────────────────
echo ""
echo "▶  Placing surgery_events.json at Unity persistentDataPath..."

mkdir -p "$PERSISTENT_PATH/ToolInteractionLog"

if [ -f "$PIPELINE_DIR/surgery_events.json" ]; then
    cp "$PIPELINE_DIR/surgery_events.json" "$EVENTS_DEST"
    echo "   ✔  Copied → $EVENTS_DEST"
else
    echo "   ✖  ERROR: surgery_events.json not found at $PIPELINE_DIR"
    exit 1
fi

# ── 3. Verify demo_recording.mp4 exists ──────────────────────────────────────
echo ""
echo "▶  Checking demo video..."

if [ -f "$PIPELINE_DIR/demo_recording.mp4" ]; then
    echo "   ✔  demo_recording.mp4 found"
else
    echo "   ⚠  Creating placeholder video..."
    ffmpeg -y -loglevel quiet \
        -f lavfi -i color=c=0x1a1a2e:size=1280x720:rate=30 \
        -f lavfi -i sine=frequency=0:sample_rate=44100 \
        -t 45 -c:v libx264 -c:a aac -shortest \
        "$PIPELINE_DIR/demo_recording.mp4"
    echo "   ✔  demo_recording.mp4 created (45s placeholder)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅  Reset complete. Ready for demo."
echo ""
if [ "$QUICK_MODE" = true ]; then
    echo "  Mode: QUICK — Steps 2-4 will run (~17 seconds)"
    echo "  Step 1 AI narration: already done ✔"
else
    echo "  Mode: FULL — All 4 steps will run"
    echo "  Step 1 (BioMistral): ~22 minutes"
    echo "  Steps 2-4: ~17 seconds"
fi
echo ""
echo "  Events for Unity: $EVENTS_DEST"
echo ""
echo "  Unity Inspector settings:"
echo "    Python Executable:   $PIPELINE_DIR/.venv/bin/python"
echo "    Pipeline Script:     $PIPELINE_DIR/run_pipeline.py"
echo "    Video Path:          $PIPELINE_DIR/demo_recording.mp4"
echo "    Events File Name:    DefaultInteractionLog"
if [ "$QUICK_MODE" = true ]; then
    echo "    Skip Narration:      ✓ (CHECK THIS BOX in Inspector)"
    echo "    Skip TTS:            ☐ (unchecked)"
else
    echo "    Skip Narration:      ☐ (unchecked)"
    echo "    Skip TTS:            ☐ (unchecked)"
fi
echo "═══════════════════════════════════════════════"
echo ""

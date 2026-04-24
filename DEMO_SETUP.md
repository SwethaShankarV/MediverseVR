# MediverseVR Demo Setup — Step by Step

## What the demo shows
1. VR surgery session records live → saves `surgery_events.json` + a video  
2. Click **"Generate Narrated Video"** — AI pipeline runs in ~17 seconds:  
   - Step 2: Coqui TTS → voice audio files  
   - Step 3: SRT subtitle file  
   - Step 4: ffmpeg stitches video + audio + subtitles → MP4  
3. Click **"Open Video"** → narrated MP4 opens in VLC/QuickTime

> **BioMistral (Step 1) is pre-run the morning of the demo** because it takes ~22 minutes.  
> Everything else runs live in front of the audience.

---

## Morning of demo — pre-run BioMistral

Open Terminal and run:

```bash
cd /Users/swethashankar/Documents/OPT/MediverseVR
.venv/bin/python generate_narration.py \
  --events surgery_events.json \
  --output final_mapped.json
```

Wait ~22 minutes. When it finishes, `final_mapped.json` will exist.  
**Do not delete this file before the demo.**

---

## One-time Unity scene setup

### Step 1 — Copy the script into Unity

Copy `NarrationPipelineClient.cs` into your Unity project's `Assets/Scripts/` folder  
(or wherever your other C# scripts live). Unity will compile it automatically.

---

### Step 2 — Create the UI in your post-session scene

You need these UI objects. Create them under a **Canvas** in the scene:

| Object | Type | Name it |
|---|---|---|
| A panel to hold everything | Panel | `NarrationPanel` |
| "Generate Narrated Video" trigger | Button | `GenerateButton` |
| Status message | TextMeshPro - Text (UI) | `StatusLabel` |
| Container that hides until pipeline starts | Panel | `ProgressPanel` |
| Progress bar | Slider | `ProgressSlider` |
| Save path text | TextMeshPro - Text (UI) | `SavePathLabel` |
| **"Open Video" button** | Button | `OpenVideoButton` |

**Hierarchy should look like:**
```
Canvas
└── NarrationPanel
    ├── GenerateButton       ("Generate Narrated Video")
    ├── StatusLabel          (empty text)
    └── ProgressPanel
        ├── ProgressSlider
        ├── SavePathLabel    (empty text)
        └── OpenVideoButton  ("Open Video")
```

**Slider settings** (select `ProgressSlider` in Inspector):
- Min Value: `0`
- Max Value: `1`
- Interactable: **OFF** (untick) — it's display-only

**OpenVideoButton** — set its label text to `"Open Video"`. It starts hidden (the script hides it on Start and shows it when the video is ready).

---

### Step 3 — Add the controller script

1. Create an empty GameObject in the scene. Name it `NarrationPipelineController`.
2. Drag `NarrationPipelineClient.cs` (from Assets/Scripts) onto `NarrationPipelineController`.

---

### Step 4 — Wire up the Inspector fields

Select `NarrationPipelineController`. In the Inspector you'll see these fields:

**UI References**

| Field | Drag from Hierarchy |
|---|---|
| Generate Button | `GenerateButton` |
| Status Label | `StatusLabel` |
| Progress Panel | `ProgressPanel` |
| Progress Slider | `ProgressSlider` |
| Save Path Label | `SavePathLabel` |
| Open Video Button | `OpenVideoButton` |

**Python / Pipeline**

| Field | Value |
|---|---|
| Python Executable | `/Users/swethashankar/Documents/OPT/MediverseVR/.venv/bin/python` |
| Pipeline Script Path | `/Users/swethashankar/Documents/OPT/MediverseVR/run_pipeline.py` |

**File Paths**

| Field | Value |
|---|---|
| Video Path | *(leave blank — script auto-detects `recording.mp4` in persistentDataPath, OR set the full path if your recorder saves elsewhere)* |
| Events File Name | `DefaultInteractionLog` ← must match JsonInteractionLogger's filename, no `.json` |
| Output File Name | `narrated_session.mp4` |

**Pipeline Options**

| Field | Value for demo |
|---|---|
| Skip Narration | ✅ **TICK THIS** — uses the pre-run `final_mapped.json` |
| Skip TTS | ❌ leave unticked — TTS runs live (~15 seconds) |

---

### Step 5 — Check persistentDataPath on your Mac

Unity saves files to:
```
/Users/swethashankar/Library/Application Support/Easley-Dunn Productions/MedicalVR-src/
```

The pipeline expects these files there at demo time:

| File | Created by |
|---|---|
| `final_mapped.json` | Pre-run BioMistral (morning of demo) |
| `ToolInteractionLog/DefaultInteractionLog.json` | JsonInteractionLogger during VR session |
| `recording.mp4` (or your video filename) | Unity video recorder during VR session |

**Copy `final_mapped.json` to persistentDataPath before the demo:**
```bash
cp /Users/swethashankar/Documents/OPT/MediverseVR/final_mapped.json \
   "/Users/swethashankar/Library/Application Support/Easley-Dunn Productions/MedicalVR-src/"
```

---

## Demo day checklist

### Morning (one-time, ~22 min)
- [ ] Run `generate_narration.py` → `final_mapped.json` exists in MediverseVR folder
- [ ] Copy `final_mapped.json` to Unity's persistentDataPath (see Step 5 above)
- [ ] Verify ffmpeg is installed: `ffmpeg -version`
- [ ] Verify Coqui TTS works: `.venv/bin/python -c "from TTS.api import TTS; print('ok')"`

### Right before demo
- [ ] Open Unity project, enter Play mode
- [ ] Confirm `NarrationPipelineController` is in the scene with all fields wired up
- [ ] **Skip Narration = ✅ ticked**
- [ ] `ProgressPanel` is hidden at start (script does this automatically)
- [ ] `OpenVideoButton` is hidden at start (script does this automatically)

### During demo (~17 seconds live)
1. Run the VR surgery session → video + `DefaultInteractionLog.json` are saved automatically
2. When session ends, the post-session UI appears
3. Show audience `surgery_events.json` (raw sensor data from headset)
4. Show audience `final_mapped.json` (AI-generated narration — pre-run)
5. Click **"Generate Narrated Video"**  
   Watch the status label cycle through:
   - *"Step 2 of 4 — Generating voice audio…"* (~15 sec)
   - *"Step 3 of 4 — Creating subtitles…"* (< 1 sec)
   - *"Step 4 of 4 — Composing final video…"* (~2 sec)
   - *"Done! Your narrated video is ready."*
6. Click **"Open Video"** → VLC/QuickTime opens `narrated_session.mp4`
7. Show the audience: voice narration + subtitles over the recorded surgery

### What to say about Step 1 (BioMistral)
> *"The AI narration step takes about 20 minutes on this Mac without a GPU — I ran it this morning. On a cloud server with a GPU it's under a minute. What you're watching now — the voice, the subtitles, the video — all generated live in about 17 seconds."*

---

## Quick reset between demo runs

```bash
# Full reset — clears everything including final_mapped.json
./demo_reset.sh

# Quick reset — keeps final_mapped.json, clears audio + video only (use this between runs)
./demo_reset.sh --quick
```

After a quick reset, **copy `final_mapped.json` back to persistentDataPath again** (Step 5 above).

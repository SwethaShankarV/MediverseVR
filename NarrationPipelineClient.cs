// ═══════════════════════════════════════════════════════════════════════════
// NarrationPipelineClient.cs
// MediverseVR — "Generate Narrated Video" button for the post-session UI.
//
// HOW IT WORKS (non-technical summary)
// ──────────────────────────────────────
// When the student presses the button, this script silently launches the
// Python narration pipeline on the same PC that Unity is running on.
// It watches the pipeline's progress and updates a status label + progress
// bar in the VR UI.  When done, the finished MP4 is saved to the headset's
// persistent storage folder and the student can keep it.
//
// No internet connection, no web browser, no server URL required for users.
// Everything runs locally on the PC the headset is plugged into.
//
// ═══════════════════════════════════════════════════════════════════════════
//
// SETUP INSTRUCTIONS
// ──────────────────────────────────────────────────────────────────────────
// 1. COPY THIS FILE into your Unity project's Scripts folder.
//
// 2. Make sure the MediverseVR Python project is on this PC and all
//    dependencies are installed:
//      cd /path/to/MediverseVR
//      pip3 install -r requirements-tts.txt
//      pip3 install fastapi uvicorn aiofiles python-multipart
//      pip3 install spacy nltk && python3 -m spacy download en_core_web_sm
//      brew install ffmpeg          (Mac)  or  choco install ffmpeg  (Windows)
//
// 3. CREATE AN EMPTY GAMEOBJECT in the post-session UI scene.
//    Name it: NarrationPipelineController
//
// 4. DRAG THIS SCRIPT onto that GameObject.
//
// 5. WIRE UP THE INSPECTOR FIELDS (see [Header] sections below):
//
//    UI References:
//      generateButton  → the "Generate Narrated Video" Button
//      statusLabel     → a TextMeshProUGUI that shows status messages
//      progressPanel   → a Panel/container that hides until pipeline starts
//      progressSlider  → a Unity UI Slider (min=0, max=1, interactable=OFF)
//      savePathLabel   → (optional) shows where the video was saved
//
//    Python / Pipeline:
//      pythonExecutable   → "python3" on Mac/Linux, "python" on Windows
//                           If it doesn't work, use the full path e.g.
//                           /usr/bin/python3  or  C:\Python311\python.exe
//      pipelineScriptPath → FULL absolute path to run_pipeline.py
//                           e.g. /Users/yourname/Documents/MediverseVR/run_pipeline.py
//
//    File Paths:
//      videoPath       → FULL absolute path to where the recording system
//                        saves the session video.
//                        Leave blank to auto-look for "recording.mp4" in
//                        Application.persistentDataPath.
//      eventsFileName  → Must match the "Json File Name" field on the
//                        JsonInteractionLogger component (default: DefaultInteractionLog)
//                        Do NOT include .json extension here.
//      outputFileName  → What to name the finished narrated video file.
//                        Default: narrated_session.mp4
//
//    Pipeline Options:
//      skipNarration   → Tick this to skip the AI text generation step
//                        (uses existing final_mapped.json — much faster,
//                        no GPU needed, good for demos)
//      skipTts         → Tick this to skip voice generation
//                        (uses existing WAV files — fastest option)
//
// 6. IMPORTANT: The JsonInteractionLogger must have finished saving its
//    events file BEFORE the student presses this button.  Call
//    JsonInteractionLogger.SaveJson() from whatever script ends the
//    surgical session (e.g. on session-end event, before showing the
//    post-session UI).
//
// ═══════════════════════════════════════════════════════════════════════════

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class NarrationPipelineClient : MonoBehaviour
{
    // ── Inspector fields ──────────────────────────────────────────────────────

    [Header("UI References")]
    [SerializeField] private Button           generateButton;
    [SerializeField] private TextMeshProUGUI  statusLabel;
    [SerializeField] private GameObject       progressPanel;
    [SerializeField] private Slider           progressSlider;
    [SerializeField] private TextMeshProUGUI  savePathLabel;

    [Header("Python / Pipeline")]
    [Tooltip("The Python interpreter. Usually 'python3' on Mac, 'python' on Windows.")]
    [SerializeField] private string pythonExecutable   = "python3";

    [Tooltip("FULL absolute path to run_pipeline.py on this PC.")]
    [SerializeField] private string pipelineScriptPath = "";

    [Header("File Paths")]
    [Tooltip("FULL absolute path to the recorded session video. Leave blank to auto-detect.")]
    [SerializeField] private string videoPath = "";

    [Tooltip("Filename (no extension) from JsonInteractionLogger's 'Json File Name' field.")]
    [SerializeField] private string eventsFileName = "DefaultInteractionLog";

    [Tooltip("What to call the finished narrated video file.")]
    [SerializeField] private string outputFileName = "narrated_session.mp4";

    [Header("Pipeline Options")]
    [Tooltip("Skip AI text generation — uses existing final_mapped.json. Faster, no GPU needed.")]
    [SerializeField] private bool skipNarration = false;

    [Tooltip("Skip voice generation — uses existing WAV files. Fastest option.")]
    [SerializeField] private bool skipTts = false;

    // ── Runtime state ─────────────────────────────────────────────────────────

    private bool    _isRunning     = false;
    private bool    _processExited = false;
    private int     _exitCode      = -1;
    private string  _outputVideoPath;

    // Thread-safe queue: Python process runs on a background thread,
    // but Unity UI can only be updated from the main thread.
    // Log lines are enqueued by the background thread and drained here.
    private ConcurrentQueue<string> _logQueue = new ConcurrentQueue<string>();

    // Track current progress fraction so we can nudge it forward on each log line
    private float _currentProgress = 0f;

    // ── Unity lifecycle ───────────────────────────────────────────────────────

    void Start()
    {
        if (generateButton != null)
            generateButton.onClick.AddListener(OnGenerateClicked);

        if (progressPanel != null)
            progressPanel.SetActive(false);

        if (savePathLabel != null)
            savePathLabel.gameObject.SetActive(false);

        SetStatus("Ready to generate narrated video.");
        SetProgress(0f);
    }

    void Update()
    {
        // Drain the log queue every frame — safe because this is the main thread
        string line;
        while (_logQueue.TryDequeue(out line))
            ProcessLogLine(line);
    }

    void OnDestroy()
    {
        // If the user quits mid-pipeline, kill the Python process cleanly
        if (_process != null && !_process.HasExited)
        {
            try { _process.Kill(); }
            catch { /* already gone */ }
        }
    }

    // ── Button callback ───────────────────────────────────────────────────────

    public void OnGenerateClicked()
    {
        if (_isRunning)
        {
            Debug.LogWarning("[NarrationPipeline] Already running — ignoring button press.");
            return;
        }
        StartCoroutine(RunPipeline());
    }

    // ── Main pipeline coroutine ───────────────────────────────────────────────

    private System.Diagnostics.Process _process;

    private IEnumerator RunPipeline()
    {
        _isRunning       = false;  // set true after validation passes
        _processExited   = false;
        _exitCode        = -1;
        _currentProgress = 0f;
        _logQueue        = new ConcurrentQueue<string>();

        generateButton.interactable = false;

        if (progressPanel != null) progressPanel.SetActive(true);
        if (savePathLabel != null) savePathLabel.gameObject.SetActive(false);

        // ── Resolve file paths ────────────────────────────────────────────────
        string resolvedVideo  = ResolveVideoPath();
        string resolvedEvents = Path.Combine(
            Application.persistentDataPath,
            "ToolInteractionLog",
            eventsFileName + ".json");
        _outputVideoPath = Path.Combine(
            Application.persistentDataPath,
            outputFileName);

        // ── Validate before touching anything ────────────────────────────────
        if (string.IsNullOrEmpty(pipelineScriptPath) || !File.Exists(pipelineScriptPath))
        {
            SetStatus("Setup error: Pipeline script not found.\n" +
                      "Set 'Pipeline Script Path' in the Inspector.\n" + pipelineScriptPath);
            FinishPipeline(false);
            yield break;
        }

        if (!File.Exists(resolvedVideo))
        {
            SetStatus("Error: Session video not found.\n" +
                      "Expected at: " + resolvedVideo);
            FinishPipeline(false);
            yield break;
        }

        if (!File.Exists(resolvedEvents))
        {
            SetStatus("Error: Events log not found.\n" +
                      "Make sure the surgery session has finished saving.\n" +
                      resolvedEvents);
            FinishPipeline(false);
            yield break;
        }

        // ── Build the command ─────────────────────────────────────────────────
        //
        // Equivalent to running this in a terminal:
        //   python3 run_pipeline.py --video "..." --events "..." --output "..."
        //
        string args = string.Format(
            "\"{0}\" --video \"{1}\" --events \"{2}\" --output \"{3}\"",
            pipelineScriptPath,
            resolvedVideo,
            resolvedEvents,
            _outputVideoPath);

        if (skipNarration) args += " --skip-narration";
        if (skipTts)       args += " --skip-tts";

        Debug.Log("[NarrationPipeline] Command: " + pythonExecutable + " " + args);

        // ── Launch Python process ─────────────────────────────────────────────
        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName               = pythonExecutable,
            Arguments              = args,
            UseShellExecute        = false,     // required for output redirection
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            CreateNoWindow         = true,      // no terminal window pops up
        };

        try
        {
            _process = new System.Diagnostics.Process();
            _process.StartInfo          = psi;
            _process.EnableRaisingEvents = true;

            // These callbacks run on background threads — only enqueue, never touch Unity objects
            _process.OutputDataReceived += (sender, e) => {
                if (e.Data != null) _logQueue.Enqueue(e.Data);
            };
            _process.ErrorDataReceived += (sender, e) => {
                if (e.Data != null) _logQueue.Enqueue(e.Data);
            };
            _process.Exited += (sender, e) => {
                _exitCode      = _process.ExitCode;
                _processExited = true;
            };

            _process.Start();
            _process.BeginOutputReadLine();
            _process.BeginErrorReadLine();
        }
        catch (Exception ex)
        {
            SetStatus("Failed to start pipeline:\n" + ex.Message +
                      "\nIs '" + pythonExecutable + "' installed and on PATH?");
            FinishPipeline(false);
            yield break;
        }

        _isRunning = true;
        SetStatus("Pipeline running\u2026 this takes a few minutes.");
        SetProgress(0.05f);

        // ── Wait for the process to finish ────────────────────────────────────
        // WaitUntil checks every frame without blocking Unity's main loop.
        yield return new WaitUntil(() => _processExited);

        // Drain any remaining log lines that arrived just before exit
        string remaining;
        while (_logQueue.TryDequeue(out remaining))
            ProcessLogLine(remaining);

        // ── Result ────────────────────────────────────────────────────────────
        bool success = (_exitCode == 0) && File.Exists(_outputVideoPath);
        FinishPipeline(success);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Called from Update() (main thread) for every line the Python process prints.
    /// Parses progress from run_pipeline.py's known output patterns.
    /// </summary>
    private void ProcessLogLine(string line)
    {
        if (string.IsNullOrEmpty(line)) return;

        // Always log to Unity console so the teammate can debug in the Editor
        Debug.Log("[Pipeline] " + line);

        // ── Map known output patterns to progress + status messages ──────────
        //
        // run_pipeline.py prints lines like:
        //   "Step 1: Generate narration text  (BioMistral-7B)"
        //   "  ⏭  Skipped — ..."
        //   "  ✔  Step 1 complete  (12.3s)"
        //   "Step 2: Text-to-speech audio generation  (Coqui TTS)"
        //   ...
        //
        if (line.Contains("Step 1:"))
        {
            SetStatus("Step 1 of 4\nGenerating narration with AI\u2026");
            SetProgress(0.10f);
        }
        else if (line.Contains("Step 2:"))
        {
            SetStatus("Step 2 of 4\nGenerating voice audio\u2026");
            SetProgress(0.30f);
        }
        else if (line.Contains("Step 3:"))
        {
            SetStatus("Step 3 of 4\nCreating subtitles\u2026");
            SetProgress(0.65f);
        }
        else if (line.Contains("Step 4:"))
        {
            SetStatus("Step 4 of 4\nComposing final video\u2026");
            SetProgress(0.80f);
        }
        else if (line.Contains("complete") || line.Contains("Skipped"))
        {
            // Nudge the bar forward a little when a step finishes or is skipped
            SetProgress(Mathf.Min(_currentProgress + 0.05f, 0.95f));
        }
        else if (line.Contains("failed") || line.Contains("Error") || line.Contains("error"))
        {
            SetStatus("Something went wrong:\n" + line.Trim());
        }
    }

    /// <summary>
    /// Called when the pipeline finishes (success or failure).
    /// Always runs on the main thread (called from the coroutine).
    /// </summary>
    private void FinishPipeline(bool success)
    {
        _isRunning = false;
        generateButton.interactable = true;

        if (success)
        {
            SetStatus("Done! Your narrated video has been saved.");
            SetProgress(1.0f);

            if (savePathLabel != null)
            {
                savePathLabel.text = "Saved to:\n" + _outputVideoPath;
                savePathLabel.gameObject.SetActive(true);
            }

            Debug.Log("[NarrationPipeline] Video saved to: " + _outputVideoPath);
        }
        else
        {
            // Don't hide the progress panel — leave the error message visible
            SetProgress(0f);
            Debug.LogError("[NarrationPipeline] Pipeline failed (exit code " + _exitCode + ")");
        }
    }

    /// <summary>
    /// Find the session video.  Checks Inspector field first, then a sensible fallback.
    /// </summary>
    private string ResolveVideoPath()
    {
        if (!string.IsNullOrEmpty(videoPath) && File.Exists(videoPath))
            return videoPath;

        // Fallback: common recording names in persistentDataPath
        string[] fallbacks = { "recording.mp4", "session.mp4", "capture.mp4" };
        foreach (string name in fallbacks)
        {
            string p = Path.Combine(Application.persistentDataPath, name);
            if (File.Exists(p))
            {
                Debug.LogWarning("[NarrationPipeline] 'Video Path' not set — " +
                                 "using fallback: " + p);
                return p;
            }
        }

        // Return the configured path even if it doesn't exist
        // (the validation step will catch this and show a clear error)
        return videoPath;
    }

    private void SetStatus(string msg)
    {
        if (statusLabel != null) statusLabel.text = msg;
    }

    private void SetProgress(float value)
    {
        _currentProgress = Mathf.Clamp01(value);
        if (progressSlider != null)
            progressSlider.value = _currentProgress;
    }
}

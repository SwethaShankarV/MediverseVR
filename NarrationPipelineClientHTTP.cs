// ═══════════════════════════════════════════════════════════════════════════
// NarrationPipelineClientHTTP.cs
// MediverseVR — "Generate Narrated Video" button (HTTP version)
//
// USE THIS SCRIPT when:
//   ✅ Standalone Quest headset (Quest 2 / Quest 3 / Quest Pro)
//   ✅ Tethered PC headset (Rift, Valve Index, etc.)
//   ✅ Unity Editor testing without a headset
//
// HOW IT WORKS
// ────────────
// Sends the surgery_events.json file to the Render server over HTTP.
// The server runs the AI pipeline, composes the video, and uploads
// everything to Firebase automatically.
// Unity polls for progress every 2 seconds, then shows a "Done" message
// when the replay is available on Firebase for others to download.
//
// No Python installation needed on the device running the app.
// All heavy AI processing happens on the server.
//
// ═══════════════════════════════════════════════════════════════════════════
//
// SETUP INSTRUCTIONS FOR TEAMMATE
// ────────────────────────────────────────────────────────────────────────────
// 1. COPY THIS FILE into your Unity project's Scripts folder.
//
// 2. CREATE AN EMPTY GAMEOBJECT in the post-session UI scene.
//    Name it: NarrationPipelineController
//
// 3. DRAG THIS SCRIPT onto that GameObject.
//
// 4. WIRE UP THE INSPECTOR FIELDS:
//
//    UI References:
//      generateButton  → the "Generate Narrated Video" Button
//      statusLabel     → a TextMeshProUGUI that shows status messages
//      progressPanel   → a Panel/container (hidden until pipeline starts)
//      progressSlider  → a Unity UI Slider (min=0, max=1, interactable=OFF)
//      savePathLabel   → (optional) shows Firebase replay URL when done
//
//    Server:
//      serverUrl       → http://localhost:8000    (local testing)
//                        https://your-app.onrender.com  (production)
//
//    File Paths:
//      eventsFileName  → must match JsonInteractionLogger's filename field
//                        (default: DefaultInteractionLog — no .json extension)
//      uploadVideo     → tick this ONLY on PC where video files are accessible.
//                        Leave UNTICKED on standalone Quest (video is large,
//                        upload over WiFi will be very slow).
//      videoPath       → full path to recorded video (only if uploadVideo=true)
//      outputFileName  → name for the local MP4 copy (default: narrated_session.mp4)
//
//    Pipeline Options:
//      skipNarration   → tick to skip BioMistral (uses existing final_mapped.json)
//      skipTts         → tick to skip voice generation (uses existing WAVs)
//      pollIntervalSec → how often to ask "is it done?" (default: 2.0)
//
// 5. Make sure JsonInteractionLogger.SaveJson() has been called before the
//    student presses this button (call it on session-end event).
//
// ═══════════════════════════════════════════════════════════════════════════

using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using TMPro;

public class NarrationPipelineClientHTTP : MonoBehaviour
{
    // ── Inspector fields ──────────────────────────────────────────────────────

    [Header("UI References")]
    [SerializeField] private Button          generateButton;
    [SerializeField] private TextMeshProUGUI statusLabel;
    [SerializeField] private GameObject      progressPanel;
    [SerializeField] private Slider          progressSlider;
    [SerializeField] private TextMeshProUGUI savePathLabel;  // optional

    [Header("Server")]
    [Tooltip("URL of the Render (or local) server. No trailing slash.")]
    [SerializeField] private string serverUrl = "http://localhost:8000";

    [Header("File Paths")]
    [Tooltip("Filename without extension — must match JsonInteractionLogger's Json File Name field.")]
    [SerializeField] private string eventsFileName = "DefaultInteractionLog";

    [Tooltip("Tick to also upload the video file. Leave OFF for standalone Quest (video is too large for WiFi upload).")]
    [SerializeField] private bool uploadVideo = false;

    [Tooltip("Full path to the recorded session video. Only used when Upload Video is ticked.")]
    [SerializeField] private string videoPath = "";

    [Tooltip("Filename for the downloaded narrated video saved locally.")]
    [SerializeField] private string outputFileName = "narrated_session.mp4";

    [Header("Pipeline Options")]
    [Tooltip("Skip AI text generation — uses existing final_mapped.json on the server.")]
    [SerializeField] private bool skipNarration = false;

    [Tooltip("Skip voice generation — uses existing WAV files on the server.")]
    [SerializeField] private bool skipTts = false;

    [Tooltip("How often (seconds) to ask the server if the job is done.")]
    [SerializeField] private float pollIntervalSec = 2.0f;

    // ── Runtime state ─────────────────────────────────────────────────────────

    private bool   _isRunning = false;
    private string _jobId;
    private string _outputVideoPath;

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

    // ── Button callback ───────────────────────────────────────────────────────

    public void OnGenerateClicked()
    {
        if (_isRunning)
        {
            Debug.LogWarning("[NarrationHTTP] Already running — ignoring button press.");
            return;
        }
        StartCoroutine(RunPipeline());
    }

    // ── Main pipeline coroutine ───────────────────────────────────────────────

    private IEnumerator RunPipeline()
    {
        _isRunning = false;
        generateButton.interactable = false;

        if (progressPanel != null) progressPanel.SetActive(true);
        if (savePathLabel != null) savePathLabel.gameObject.SetActive(false);

        SetProgress(0f);

        // ── Resolve paths ─────────────────────────────────────────────────────
        string eventsPath = Path.Combine(
            Application.persistentDataPath, "ToolInteractionLog", eventsFileName + ".json");

        _outputVideoPath = Path.Combine(Application.persistentDataPath, outputFileName);

        // ── Validate events file ──────────────────────────────────────────────
        if (!File.Exists(eventsPath))
        {
            SetStatus("Error: Events log not found.\n" +
                      "Make sure the surgery session has finished saving.\n" + eventsPath);
            FinishPipeline(false, null);
            yield break;
        }

        // ── Phase 1: Health check ─────────────────────────────────────────────
        SetStatus("Connecting to server…");
        using (var healthReq = UnityWebRequest.Get(serverUrl + "/health"))
        {
            healthReq.timeout = 10;
            yield return healthReq.SendWebRequest();

            if (healthReq.result != UnityWebRequest.Result.Success)
            {
                SetStatus("Cannot reach the server.\n" +
                          "Is it running at: " + serverUrl + "\n" +
                          healthReq.error);
                FinishPipeline(false, null);
                yield break;
            }
        }

        SetStatus("Server reachable. Uploading events…");
        SetProgress(0.05f);

        // ── Phase 2: Upload events (and optionally video) ─────────────────────
        var form = new WWWForm();

        // Events JSON — always sent
        byte[] eventsBytes = File.ReadAllBytes(eventsPath);
        form.AddBinaryData("events", eventsBytes, Path.GetFileName(eventsPath), "application/json");

        // Video — only if user opted in AND file exists
        bool videoSent = false;
        if (uploadVideo && !string.IsNullOrEmpty(videoPath) && File.Exists(videoPath))
        {
            byte[] videoBytes = File.ReadAllBytes(videoPath);
            string ext        = Path.GetExtension(videoPath).ToLower();
            form.AddBinaryData("video", videoBytes, Path.GetFileName(videoPath), "video/mp4");
            videoSent = true;
            SetStatus("Uploading events + video…\n(this may take a moment over WiFi)");
        }
        else
        {
            // Server requires a video field — send a tiny placeholder so the
            // server knows to skip video composition
            form.AddBinaryData("video", new byte[0], "none.mp4", "video/mp4");
        }

        form.AddField("skip_narration", skipNarration ? "true" : "false");
        form.AddField("skip_tts",       skipTts       ? "true" : "false");

        string processUrl = serverUrl + "/process";
        Debug.Log("[NarrationHTTP] POST " + processUrl + (videoSent ? " (with video)" : " (events only)"));

        using (var uploadReq = UnityWebRequest.Post(processUrl, form))
        {
            uploadReq.timeout = videoSent ? 300 : 30;  // longer timeout for video uploads
            yield return uploadReq.SendWebRequest();

            if (uploadReq.result != UnityWebRequest.Result.Success)
            {
                SetStatus("Upload failed:\n" + uploadReq.downloadHandler.text);
                FinishPipeline(false, null);
                yield break;
            }

            // Parse job_id from response
            var response = JsonUtility.FromJson<ProcessResponse>(uploadReq.downloadHandler.text);
            _jobId = response?.job_id;

            if (string.IsNullOrEmpty(_jobId))
            {
                SetStatus("Server returned an unexpected response.\nCheck server logs.");
                FinishPipeline(false, null);
                yield break;
            }
        }

        Debug.Log("[NarrationHTTP] Job started: " + _jobId);
        SetStatus("Job started. Pipeline running on server…");
        SetProgress(0.10f);
        _isRunning = true;

        // ── Phase 3+4: Poll for progress ──────────────────────────────────────
        string statusUrl = serverUrl + "/status/" + _jobId;
        string jobStatus = "running";
        string lastStep  = "";

        while (jobStatus == "running" || jobStatus == "queued")
        {
            yield return new WaitForSeconds(pollIntervalSec);

            using (var pollReq = UnityWebRequest.Get(statusUrl))
            {
                pollReq.timeout = 10;
                yield return pollReq.SendWebRequest();

                if (pollReq.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning("[NarrationHTTP] Poll failed (will retry): " + pollReq.error);
                    continue;  // transient network blip — try again next interval
                }

                var statusResp = JsonUtility.FromJson<StatusResponse>(pollReq.downloadHandler.text);
                if (statusResp == null) continue;

                jobStatus = statusResp.status ?? jobStatus;
                string step = statusResp.step ?? "";

                // Update UI only when the step changes (avoids flicker)
                if (step != lastStep)
                {
                    lastStep = step;
                    UpdateProgressFromStep(step, statusResp.message);
                }
            }
        }

        // ── Phase 5: Result ───────────────────────────────────────────────────
        bool success = (jobStatus == "done");

        // Optionally download the MP4 locally
        if (success && !string.IsNullOrEmpty(_outputVideoPath))
        {
            SetStatus("Downloading video to device…");
            yield return DownloadVideo(_jobId);
        }

        FinishPipeline(success, jobStatus == "done" ? null : "Pipeline failed on server. Check server logs.");
    }

    // ── Download the finished MP4 to device storage ───────────────────────────

    private IEnumerator DownloadVideo(string jobId)
    {
        string downloadUrl = serverUrl + "/download/" + jobId;
        using (var dlReq = new UnityWebRequest(downloadUrl, UnityWebRequest.kHttpVerbGET))
        {
            // DownloadHandlerFile streams directly to disk — no RAM spike for large videos
            dlReq.downloadHandler = new DownloadHandlerFile(_outputVideoPath);
            dlReq.timeout = 600;  // 10 minutes for large files
            yield return dlReq.SendWebRequest();

            if (dlReq.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning("[NarrationHTTP] Video download failed: " + dlReq.error);
                // Non-fatal — replay is still on Firebase even if local save fails
            }
            else
            {
                Debug.Log("[NarrationHTTP] Video saved to: " + _outputVideoPath);
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void UpdateProgressFromStep(string step, string message)
    {
        switch (step)
        {
            case "starting":
                SetStatus("Starting pipeline…");       SetProgress(0.08f); break;
            case "running":
                SetStatus("Pipeline running…");        SetProgress(0.12f); break;
            // run_pipeline.py logs "Step 1:", "Step 2:" etc. which end up in message
            default:
                if (!string.IsNullOrEmpty(message))
                {
                    if (message.Contains("Step 1") || message.Contains("narration"))
                    { SetStatus("Step 1 of 4 — Generating narration text…"); SetProgress(0.20f); }
                    else if (message.Contains("Step 2") || message.Contains("audio"))
                    { SetStatus("Step 2 of 4 — Generating voice audio…");    SetProgress(0.45f); }
                    else if (message.Contains("Step 3") || message.Contains("subtitle"))
                    { SetStatus("Step 3 of 4 — Creating subtitles…");         SetProgress(0.70f); }
                    else if (message.Contains("Step 4") || message.Contains("video"))
                    { SetStatus("Step 4 of 4 — Composing final video…");      SetProgress(0.85f); }
                    else if (message.Contains("Firebase") || message.Contains("upload"))
                    { SetStatus("Uploading to Firebase…");                    SetProgress(0.95f); }
                }
                break;
        }
    }

    private void FinishPipeline(bool success, string errorMessage)
    {
        _isRunning = false;
        if (generateButton != null) generateButton.interactable = true;

        if (success)
        {
            SetStatus("Done!\nReplay is available. Open the Replay Browser to watch or download.");
            SetProgress(1.0f);

            if (savePathLabel != null)
            {
                bool videoSavedLocally = File.Exists(_outputVideoPath);
                savePathLabel.text = videoSavedLocally
                    ? "Video saved to:\n" + _outputVideoPath
                    : "Replay available on Firebase.\nOpen the Replay Browser to download.";
                savePathLabel.gameObject.SetActive(true);
            }
        }
        else
        {
            SetStatus(errorMessage ?? "Something went wrong.\nPlease try again.");
            SetProgress(0f);
        }
    }

    private void SetStatus(string msg)
    {
        if (statusLabel != null) statusLabel.text = msg;
        Debug.Log("[NarrationHTTP] " + msg.Replace("\n", " "));
    }

    private void SetProgress(float value)
    {
        if (progressSlider != null)
            progressSlider.value = Mathf.Clamp01(value);
    }

    // ── JSON response models (JsonUtility-compatible) ──────────────────────────

    [Serializable]
    private class ProcessResponse
    {
        public string job_id;
        public string status;
        public string message;
    }

    [Serializable]
    private class StatusResponse
    {
        public string job_id;
        public string status;
        public string step;
        public string message;
    }
}

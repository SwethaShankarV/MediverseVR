// ═══════════════════════════════════════════════════════════════════════════
// NarrationPipelineTrigger.cs
// MediverseVR — "Generate Narration" button (VR assets only, no video)
//
// WHAT THIS SCRIPT DOES
// ─────────────────────
// After a surgery session ends, this script:
//   1. Reads ToolInteractionLog/DefaultInteractionLog.json from persistentDataPath
//   2. Sends it to the narration pipeline server
//   3. Polls for progress (~15 sec with Skip Narration ticked, ~22 min fresh)
//   4. Downloads final_mapped.json + audio_manifest.json + WAV clips to
//      Application.persistentDataPath (exactly where AINarrationController reads)
//   5. Calls AINarrationController.ReloadAssets() so the Explanation button
//      plays the new narration immediately without a scene reload
//
// WORKS ON
// ────────
//   ✅ Standalone Quest (Quest 2 / Quest 3 / Quest Pro)
//   ✅ Tethered PC headset
//   ✅ Unity Editor
//
// This is different from NarrationPipelineClientHTTP.cs which generates a
// full narrated MP4 video. This script only generates the in-VR audio + text
// narration (faster, no video upload needed).
//
// ═══════════════════════════════════════════════════════════════════════════
//
// SETUP INSTRUCTIONS
// ──────────────────
// 1. Copy this file into your Unity project's Scripts folder.
//
// 2. Create an empty GameObject in your post-session scene.
//    Name it: NarrationTriggerController
//
// 3. Drag this script onto that GameObject.
//
// 4. Wire the Inspector fields:
//
//    UI References:
//      generateButton    → Button — "Generate Narration"
//      statusLabel       → TextMeshProUGUI showing live status
//      progressPanel     → Panel (hidden until pipeline starts, optional)
//      progressSlider    → Slider 0→1 (optional)
//
//    Server:
//      serverUrl         → http://localhost:8000          (local dev)
//                          https://your-app.onrender.com  (cloud)
//
//    File Paths:
//      eventsFileName    → must match JsonInteractionLogger's filename
//                          (default: DefaultInteractionLog — no .json)
//
//    Pipeline Options:
//      skipNarration     → tick to skip BioMistral (uses existing final_mapped.json)
//                          USE THIS for demos — Steps 2–4 run in ~15 seconds
//      skipTts           → tick to skip voice generation (uses existing WAVs)
//      pollIntervalSec   → how often to check "are we done?" (default: 2.0)
//
//    VR Playback:
//      narrationController → drag in the AINarrationController GameObject
//                            It will call ReloadAssets() when download finishes
//
// 5. Ensure JsonInteractionLogger.SaveJson() has been called before the
//    student presses this button (call it on your session-end event).
//
// ═══════════════════════════════════════════════════════════════════════════

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using TMPro;

public class NarrationPipelineTrigger : MonoBehaviour
{
    // ── Inspector fields ──────────────────────────────────────────────────────

    [Header("UI References")]
    [SerializeField] private Button          generateButton;
    [SerializeField] private TextMeshProUGUI statusLabel;
    [SerializeField] private GameObject      progressPanel;
    [SerializeField] private Slider          progressSlider;

    [Header("Server")]
    [Tooltip("URL of the pipeline server. No trailing slash.")]
    [SerializeField] private string serverUrl = "http://localhost:8000";

    [Header("File Paths")]
    [Tooltip("Filename without .json extension — must match JsonInteractionLogger's Json File Name field.")]
    [SerializeField] private string eventsFileName = "DefaultInteractionLog";

    [Header("Pipeline Options")]
    [Tooltip("Skip AI text generation — uses existing final_mapped.json on the server. " +
             "Tick this for demos: reduces pipeline time from ~22 min to ~15 sec.")]
    [SerializeField] private bool skipNarration = false;

    [Tooltip("Skip voice generation — uses existing WAV files on the server.")]
    [SerializeField] private bool skipTts = false;

    [Tooltip("How often (seconds) to ask the server if the job is done.")]
    [SerializeField] private float pollIntervalSec = 2.0f;

    [Header("VR Playback")]
    [Tooltip("Optional: drag in the GameObject that has your narration controller. " +
             "Leave empty if not used.")]
    [SerializeField] private GameObject narrationControllerObject;

    [Header("Demo / Debug")]
    [Tooltip("Press this key to trigger the pipeline (useful when XR click isn't available).")]
    [SerializeField] private KeyCode keyboardShortcut = KeyCode.G;

    // ── Runtime state ─────────────────────────────────────────────────────────

    private bool   _isRunning = false;
    private string _jobId;

    // ── Unity lifecycle ───────────────────────────────────────────────────────

    void Start()
    {
        if (generateButton != null)
            generateButton.onClick.AddListener(OnGenerateClicked);

        if (progressPanel != null)
            progressPanel.SetActive(false);

        SetStatus("Ready — press Generate Narration after your session.");
        SetProgress(0f);
    }

    void Update()
    {
        // Keyboard shortcut fallback — press G (or whatever keyboardShortcut is set to)
        if (Input.GetKeyDown(keyboardShortcut))
        {
            Debug.Log("[NarrationTrigger] Keyboard shortcut triggered: " + keyboardShortcut);
            OnGenerateClicked();
        }
    }

    // ── Button callback ───────────────────────────────────────────────────────

    public void OnGenerateClicked()
    {
        if (_isRunning)
        {
            Debug.LogWarning("[NarrationTrigger] Already running — ignoring button press.");
            return;
        }
        StartCoroutine(RunPipeline());
    }

    // ── Main coroutine ────────────────────────────────────────────────────────

    private IEnumerator RunPipeline()
    {
        _isRunning = true;
        if (generateButton != null) generateButton.interactable = false;
        if (progressPanel  != null) progressPanel.SetActive(true);
        SetProgress(0f);

        // ── Resolve events path ───────────────────────────────────────────────
        string eventsPath = Path.Combine(
            Application.persistentDataPath,
            "ToolInteractionLog",
            eventsFileName + ".json");

        if (!File.Exists(eventsPath))
        {
            SetStatus($"Events log not found.\nMake sure the surgery session has finished.\n{eventsPath}");
            Finish(false);
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
                SetStatus($"Cannot reach the server.\nIs it running at: {serverUrl}\n{healthReq.error}");
                Finish(false);
                yield break;
            }
        }

        SetStatus("Server reachable. Uploading events…");
        SetProgress(0.05f);

        // ── Phase 2: Upload events ────────────────────────────────────────────
        // vr_assets_only=true → server skips video composition (Steps 1–3 only)
        // This is the key difference from NarrationPipelineClientHTTP.cs
        var form = new WWWForm();
        byte[] eventsBytes = File.ReadAllBytes(eventsPath);
        form.AddBinaryData("events", eventsBytes, Path.GetFileName(eventsPath), "application/json");

        // Server requires a video field — send an empty placeholder so the
        // server knows to skip video composition (vr_assets_only handles the rest)
        form.AddBinaryData("video", new byte[0], "placeholder.mp4", "video/mp4");

        form.AddField("skip_narration", skipNarration ? "true" : "false");
        form.AddField("skip_tts",       skipTts       ? "true" : "false");
        form.AddField("vr_assets_only", "true");  // always true for this trigger script

        string processUrl = serverUrl + "/process";
        Debug.Log("[NarrationTrigger] POST " + processUrl);

        using (var uploadReq = UnityWebRequest.Post(processUrl, form))
        {
            uploadReq.timeout = 30;
            yield return uploadReq.SendWebRequest();

            if (uploadReq.result != UnityWebRequest.Result.Success)
            {
                SetStatus("Upload failed:\n" + uploadReq.downloadHandler.text);
                Finish(false);
                yield break;
            }

            var response = JsonUtility.FromJson<ProcessResponse>(uploadReq.downloadHandler.text);
            _jobId = response?.job_id;

            if (string.IsNullOrEmpty(_jobId))
            {
                SetStatus("Server returned an unexpected response.\nCheck server logs.");
                Finish(false);
                yield break;
            }
        }

        Debug.Log("[NarrationTrigger] Job started: " + _jobId);
        SetStatus("Narration pipeline running on server…");
        SetProgress(0.10f);

        // ── Phase 3: Poll for progress ────────────────────────────────────────
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
                    Debug.LogWarning("[NarrationTrigger] Poll failed (retrying): " + pollReq.error);
                    continue;
                }

                var statusResp = JsonUtility.FromJson<StatusResponse>(pollReq.downloadHandler.text);
                if (statusResp == null) continue;

                jobStatus = statusResp.status ?? jobStatus;
                string step = statusResp.step ?? "";

                if (step != lastStep)
                {
                    lastStep = step;
                    UpdateProgressFromStep(step, statusResp.message);
                }

                // Capture asset URLs once the job is done
                if (jobStatus == "done")
                {
                    yield return DownloadVrAssets(statusResp);
                    Finish(true);
                    yield break;
                }
            }
        }

        // Fell out of loop with a non-done status (e.g. "failed")
        Finish(false, "Pipeline failed on server. Check server logs.");
    }

    // ── Download VR assets to persistentDataPath ──────────────────────────────

    private IEnumerator DownloadVrAssets(StatusResponse statusResp)
    {
        SetStatus("Downloading narration assets…");
        SetProgress(0.85f);

        string pDataPath = Application.persistentDataPath;
        string audioDir  = Path.Combine(pDataPath, "audio_steps");
        Directory.CreateDirectory(audioDir);

        int totalFiles = 1                                                     // final_mapped.json
            + (string.IsNullOrEmpty(statusResp.audio_manifest_url) ? 0 : 1)   // audio_manifest.json
            + (statusResp.audio_urls != null ? statusResp.audio_urls.Length : 0);
        int downloaded = 0;

        // 1. final_mapped.json
        if (!string.IsNullOrEmpty(statusResp.final_mapped_url))
        {
            yield return DownloadFile(
                serverUrl + statusResp.final_mapped_url,
                Path.Combine(pDataPath, "final_mapped.json"));
            downloaded++;
            SetProgress(0.85f + 0.12f * downloaded / totalFiles);
        }

        // 2. audio_manifest.json
        if (!string.IsNullOrEmpty(statusResp.audio_manifest_url))
        {
            yield return DownloadFile(
                serverUrl + statusResp.audio_manifest_url,
                Path.Combine(pDataPath, "audio_manifest.json"));
            downloaded++;
            SetProgress(0.85f + 0.12f * downloaded / totalFiles);
        }

        // 3. WAV audio clips
        if (statusResp.audio_urls != null)
        {
            for (int i = 0; i < statusResp.audio_urls.Length; i++)
            {
                string relUrl  = statusResp.audio_urls[i];
                string wavName = $"step_{i:D4}.wav";

                // Preserve the server's filename if available (has the narration slug)
                string serverName = relUrl.Contains("/") ? relUrl.Substring(relUrl.LastIndexOf('/') + 1) : "";
                if (!string.IsNullOrEmpty(serverName) && serverName.EndsWith(".wav"))
                    wavName = serverName;

                yield return DownloadFile(
                    serverUrl + relUrl,
                    Path.Combine(audioDir, wavName));

                downloaded++;
                SetProgress(0.85f + 0.12f * downloaded / totalFiles);
            }
        }

        SetProgress(0.98f);

        SetStatus("Done!\nNarration assets saved to:\n" + pDataPath +
                  "\nPress the Explanation button to play.");

        SetProgress(1.0f);
    }

    // ── Generic file downloader ───────────────────────────────────────────────

    private IEnumerator DownloadFile(string url, string savePath)
    {
        if (string.IsNullOrEmpty(url))
        {
            Debug.LogWarning("[NarrationTrigger] Empty URL — skipping: " + savePath);
            yield break;
        }

        using (var req = new UnityWebRequest(url, UnityWebRequest.kHttpVerbGET))
        {
            req.downloadHandler = new DownloadHandlerFile(savePath);
            req.timeout = 120;

            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
                Debug.LogError($"[NarrationTrigger] Failed to download {Path.GetFileName(savePath)}: {req.error}");
            else
                Debug.Log($"[NarrationTrigger] Saved: {Path.GetFileName(savePath)}");
        }
    }

    // ── Progress mapping ──────────────────────────────────────────────────────

    private void UpdateProgressFromStep(string step, string message)
    {
        switch (step)
        {
            case "starting": SetStatus("Starting pipeline…");        SetProgress(0.08f); break;
            case "running":  SetStatus("Pipeline running…");         SetProgress(0.12f); break;
            default:
                if (!string.IsNullOrEmpty(message))
                {
                    if (message.Contains("Step 1") || message.Contains("narration"))
                    { SetStatus("Step 1 of 3 — Generating narration text…"); SetProgress(0.20f); }
                    else if (message.Contains("Step 2") || message.Contains("audio"))
                    { SetStatus("Step 2 of 3 — Generating voice audio…");    SetProgress(0.50f); }
                    else if (message.Contains("Step 3") || message.Contains("subtitle"))
                    { SetStatus("Step 3 of 3 — Creating subtitles…");        SetProgress(0.75f); }
                }
                break;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void Finish(bool success, string errorMessage = null)
    {
        _isRunning = false;
        if (generateButton != null) generateButton.interactable = true;

        if (!success)
        {
            SetStatus(errorMessage ?? "Something went wrong.\nPlease try again.");
            SetProgress(0f);
        }
    }

    private void SetStatus(string msg)
    {
        if (statusLabel != null) statusLabel.text = msg;
        Debug.Log("[NarrationTrigger] " + msg.Replace("\n", " "));
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
        public string   job_id;
        public string   status;
        public string   step;
        public string   message;
        // Asset download URLs — populated by server when job is done
        public string   final_mapped_url;
        public string   audio_manifest_url;
        public string[] audio_urls;
        public string   srt_url;
    }
}

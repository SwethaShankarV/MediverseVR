// ═══════════════════════════════════════════════════════════════════════════
// FirebaseReplayBrowser.cs
// MediverseVR — Browse and download past surgery replays from Firebase
//
// WORKS ON:
//   ✅ Standalone Quest (Quest 2 / Quest 3 / Quest Pro)
//   ✅ Tethered PC headset
//   ✅ Unity Editor (for testing without a headset)
//
// HOW IT WORKS
// ────────────
// 1. Calls GET /replays on the server → gets a list of all past sessions
//    (the server reads from Firestore or falls back to local jobs)
// 2. Shows each replay as a card: surgery title + date + duration
// 3. User presses "Play in VR" → downloads final_mapped.json + WAV files
//    to persistentDataPath → hands off to ExplanationUIController
// 4. User presses "Download Video" → downloads narrated.mp4 to device
//
// No Firebase SDK package required — uses standard UnityWebRequest only.
//
// ═══════════════════════════════════════════════════════════════════════════
//
// SETUP INSTRUCTIONS FOR TEAMMATE
// ────────────────────────────────────────────────────────────────────────────
// 1. COPY THIS FILE into your Unity project's Scripts folder.
//
// 2. CREATE AN EMPTY GAMEOBJECT in the replay-browser scene.
//    Name it: ReplayBrowserController
//
// 3. DRAG THIS SCRIPT onto that GameObject.
//
// 4. WIRE UP THE INSPECTOR FIELDS:
//
//    Server:
//      serverUrl         → same URL as NarrationPipelineClientHTTP
//                          e.g. https://your-app.onrender.com
//
//    UI References:
//      refreshButton     → "Refresh" button
//      replayListContent → the Content object inside a ScrollView
//      replayItemPrefab  → a prefab with:
//                            - titleLabel    (TextMeshProUGUI)
//                            - dateLabel     (TextMeshProUGUI)
//                            - durationLabel (TextMeshProUGUI)
//                            - vrButton      (Button) — "Play in VR"
//                            - videoButton   (Button) — "Download Video"
//      statusLabel       → a TextMeshProUGUI for status messages
//      downloadProgress  → a Slider showing download progress (0-1)
//
//    VR Playback:
//      explanationController → drag in your ExplanationUIController here
//                              (this hands off downloaded files to it)
//
// 5. Call RefreshReplays() from your menu-open event, or tick
//    autoRefreshOnEnable to refresh every time the panel opens.
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

public class FirebaseReplayBrowser : MonoBehaviour
{
    // ── Inspector fields ──────────────────────────────────────────────────────

    [Header("Server")]
    [Tooltip("Same server URL used by NarrationPipelineClientHTTP.")]
    [SerializeField] private string serverUrl = "http://localhost:8000";

    [Header("UI References")]
    [SerializeField] private Button          refreshButton;
    [SerializeField] private Transform       replayListContent;   // ScrollView → Viewport → Content
    [SerializeField] private GameObject      replayItemPrefab;
    [SerializeField] private TextMeshProUGUI statusLabel;
    [SerializeField] private Slider          downloadProgress;

    [Header("VR Playback")]
    [Tooltip("Drag in your ExplanationUIController — called after VR assets are downloaded.")]
    [SerializeField] private MonoBehaviour   explanationController;  // typed as MonoBehaviour to avoid hard dependency

    [Header("Options")]
    [Tooltip("Automatically fetch the replay list every time this GameObject becomes active.")]
    [SerializeField] private bool autoRefreshOnEnable = true;

    // ── Runtime state ─────────────────────────────────────────────────────────

    private bool _isDownloading = false;
    private List<ReplayItem> _replays = new List<ReplayItem>();

    // ── Unity lifecycle ───────────────────────────────────────────────────────

    void Start()
    {
        if (refreshButton != null)
            refreshButton.onClick.AddListener(RefreshReplays);

        if (downloadProgress != null)
            downloadProgress.gameObject.SetActive(false);
    }

    void OnEnable()
    {
        if (autoRefreshOnEnable)
            RefreshReplays();
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public void RefreshReplays()
    {
        StartCoroutine(FetchReplays());
    }

    // ── Fetch replay list from server ─────────────────────────────────────────

    private IEnumerator FetchReplays()
    {
        SetStatus("Loading replays…");
        ClearList();

        string url = serverUrl + "/replays";
        using (var req = UnityWebRequest.Get(url))
        {
            req.timeout = 15;
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                SetStatus("Could not load replays.\nIs the server running at " + serverUrl + "?");
                Debug.LogError("[ReplayBrowser] " + req.error);
                yield break;
            }

            var response = JsonUtility.FromJson<ReplaysResponse>(req.downloadHandler.text);
            if (response == null || response.replays == null || response.replays.Length == 0)
            {
                SetStatus("No replays found yet.\nGenerate one using the surgery session button!");
                yield break;
            }

            _replays.Clear();
            foreach (var r in response.replays)
                _replays.Add(r);

            PopulateList();
            SetStatus($"{_replays.Count} replay{(_replays.Count == 1 ? "" : "s")} available.");
        }
    }

    // ── Build the replay list UI ──────────────────────────────────────────────

    private void PopulateList()
    {
        if (replayItemPrefab == null || replayListContent == null) return;

        foreach (var replay in _replays)
        {
            var item = Instantiate(replayItemPrefab, replayListContent);

            // Title
            var titleLbl = item.transform.Find("titleLabel")?.GetComponent<TextMeshProUGUI>();
            if (titleLbl != null) titleLbl.text = replay.title ?? "Surgery Session";

            // Date — format ISO string to something readable
            var dateLbl = item.transform.Find("dateLabel")?.GetComponent<TextMeshProUGUI>();
            if (dateLbl != null)
            {
                if (DateTime.TryParse(replay.uploaded_at, out DateTime dt))
                    dateLbl.text = dt.ToLocalTime().ToString("MMM d, yyyy  h:mm tt");
                else
                    dateLbl.text = replay.uploaded_at ?? "";
            }

            // Duration
            var durLbl = item.transform.Find("durationLabel")?.GetComponent<TextMeshProUGUI>();
            if (durLbl != null)
            {
                int secs    = (int)replay.duration_seconds;
                durLbl.text = secs > 0 ? $"{secs / 60}m {secs % 60}s" : "";
            }

            // "Play in VR" button
            var vrBtn = item.transform.Find("vrButton")?.GetComponent<Button>();
            if (vrBtn != null)
            {
                // Disable if no VR assets available
                bool hasVrAssets = !string.IsNullOrEmpty(replay.final_mapped_url);
                vrBtn.interactable = hasVrAssets;

                ReplayItem captured = replay;
                vrBtn.onClick.AddListener(() => OnPlayInVR(captured));
            }

            // "Download Video" button
            var videoBtn = item.transform.Find("videoButton")?.GetComponent<Button>();
            if (videoBtn != null)
            {
                bool hasVideo = !string.IsNullOrEmpty(replay.video_url);
                videoBtn.interactable = hasVideo;

                ReplayItem captured = replay;
                videoBtn.onClick.AddListener(() => OnDownloadVideo(captured));
            }
        }
    }

    private void ClearList()
    {
        if (replayListContent == null) return;
        foreach (Transform child in replayListContent)
            Destroy(child.gameObject);
    }

    // ── "Play in VR" handler ──────────────────────────────────────────────────

    private void OnPlayInVR(ReplayItem replay)
    {
        if (_isDownloading) return;
        StartCoroutine(DownloadVrAssets(replay));
    }

    private IEnumerator DownloadVrAssets(ReplayItem replay)
    {
        _isDownloading = true;
        SetStatus($"Downloading VR assets for:\n{replay.title}…");
        ShowProgress(0f);

        string replayDir = Path.Combine(Application.persistentDataPath, "Replays", replay.job_id);
        Directory.CreateDirectory(replayDir);
        Directory.CreateDirectory(Path.Combine(replayDir, "audio_steps"));

        // Download final_mapped.json (narration text + timestamps)
        yield return DownloadFile(
            replay.final_mapped_url,
            Path.Combine(replayDir, "final_mapped.json"),
            0.10f, 0.30f);

        // Download audio_manifest.json
        if (!string.IsNullOrEmpty(replay.audio_manifest_url))
            yield return DownloadFile(
                replay.audio_manifest_url,
                Path.Combine(replayDir, "audio_manifest.json"),
                0.30f, 0.40f);

        // Download all WAV files
        if (replay.audio_urls != null)
        {
            for (int i = 0; i < replay.audio_urls.Length; i++)
            {
                string wavUrl  = replay.audio_urls[i];
                string wavName = $"step_{i:D4}.wav";
                float  pStart  = 0.40f + (i / (float)replay.audio_urls.Length) * 0.55f;
                float  pEnd    = 0.40f + ((i + 1) / (float)replay.audio_urls.Length) * 0.55f;

                yield return DownloadFile(
                    wavUrl,
                    Path.Combine(replayDir, "audio_steps", wavName),
                    pStart, pEnd);
            }
        }

        ShowProgress(1.0f);
        SetStatus("Assets downloaded. Starting VR playback…");
        HideProgress();

        // ── Hand off to ExplanationUIController ──────────────────────────────
        if (explanationController != null)
        {
            // ExplanationUIController expects files in Application.persistentDataPath.
            // We copy the replay folder there and call its load method.
            // Adjust the method name if your ExplanationUIController uses a different one.
            var method = explanationController.GetType().GetMethod("LoadReplay");
            if (method != null)
            {
                method.Invoke(explanationController, new object[] { replayDir });
                Debug.Log("[ReplayBrowser] Handed off to ExplanationUIController: " + replayDir);
            }
            else
            {
                // Fallback: copy files to the default persistentDataPath location
                // that ExplanationUIController already looks for
                string defaultDir = Application.persistentDataPath;
                File.Copy(Path.Combine(replayDir, "final_mapped.json"),
                          Path.Combine(defaultDir, "final_mapped.json"), overwrite: true);
                Debug.Log("[ReplayBrowser] Copied final_mapped.json to: " + defaultDir);
                SetStatus("Assets ready. Press Play in your VR scene to start.");
            }
        }
        else
        {
            SetStatus("Assets downloaded to:\n" + replayDir);
        }

        _isDownloading = false;
    }

    // ── "Download Video" handler ──────────────────────────────────────────────

    private void OnDownloadVideo(ReplayItem replay)
    {
        if (_isDownloading) return;
        StartCoroutine(DownloadVideoFile(replay));
    }

    private IEnumerator DownloadVideoFile(ReplayItem replay)
    {
        _isDownloading = true;

        string savePath = Path.Combine(
            Application.persistentDataPath,
            $"narrated_{replay.job_id}.mp4");

        SetStatus($"Downloading video:\n{replay.title}…");
        ShowProgress(0f);

        yield return DownloadFile(replay.video_url, savePath, 0f, 1f);

        SetStatus($"Video saved!\n{savePath}");
        HideProgress();
        Debug.Log("[ReplayBrowser] Video saved to: " + savePath);

        _isDownloading = false;
    }

    // ── Generic file downloader ───────────────────────────────────────────────

    private IEnumerator DownloadFile(string url, string savePath, float progressStart, float progressEnd)
    {
        if (string.IsNullOrEmpty(url))
        {
            Debug.LogWarning("[ReplayBrowser] Empty URL — skipping: " + savePath);
            yield break;
        }

        using (var req = new UnityWebRequest(url, UnityWebRequest.kHttpVerbGET))
        {
            req.downloadHandler = new DownloadHandlerFile(savePath);
            req.timeout = 300;

            var op = req.SendWebRequest();
            while (!op.isDone)
            {
                ShowProgress(Mathf.Lerp(progressStart, progressEnd, req.downloadProgress));
                yield return null;
            }

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ReplayBrowser] Failed to download {Path.GetFileName(savePath)}: {req.error}");
                SetStatus($"Download failed:\n{req.error}");
            }
        }
    }

    // ── UI helpers ────────────────────────────────────────────────────────────

    private void SetStatus(string msg)
    {
        if (statusLabel != null) statusLabel.text = msg;
        Debug.Log("[ReplayBrowser] " + msg.Replace("\n", " "));
    }

    private void ShowProgress(float value)
    {
        if (downloadProgress == null) return;
        downloadProgress.gameObject.SetActive(true);
        downloadProgress.value = Mathf.Clamp01(value);
    }

    private void HideProgress()
    {
        if (downloadProgress != null)
            downloadProgress.gameObject.SetActive(false);
    }

    // ── JSON models (JsonUtility-compatible) ──────────────────────────────────

    [Serializable]
    private class ReplaysResponse
    {
        public string        source;
        public int           total;
        public ReplayItem[]  replays;
    }

    [Serializable]
    private class ReplayItem
    {
        public string   job_id;
        public string   title;
        public string   uploaded_at;
        public float    duration_seconds;
        public string   events_url;
        public string   final_mapped_url;
        public string   audio_manifest_url;
        public string[] audio_urls;
        public string   srt_url;
        public string   video_url;
    }
}

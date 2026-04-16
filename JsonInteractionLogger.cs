using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[Serializable]
public class InteractionEvent
{
    public float start_time;
    public float end_time;
    public string tool;
    public string action;
    public string target;
}

[Serializable]
public class Metadata
{
    public string schema_version = "1.0";
    public Units units = new Units();
}
[Serializable]
public class Units
{
    public string time = "seconds";
}

[Serializable]
public class InteractionFile
{
    public Metadata metadata = new Metadata();
    public List<InteractionEvent> data = new List<InteractionEvent>();
}

public class JsonInteractionLogger : MonoBehaviour, IToolLogger
{
    [SerializeField] private string toolName;
    [SerializeField] private string enterActionName;
    [SerializeField] private string exitActionName;
    [SerializeField] private string jsonFileName = "DefaultInteractionLog.json";

    // Completed events
    private List<InteractionEvent> completedEvents = new List<InteractionEvent>();

    // Active (in-progress) events keyed by GameObject instance id
    private Dictionary<int, InteractionEvent> activeEvents = new Dictionary<int, InteractionEvent>();

    private float collisionCooldown = 0.5f;
    private float lastCollisionTime;

    void Start()
    {
        lastCollisionTime = Time.time;
    }

    void Update()
    {
        // press the S key to save the json file
        if (Input.GetKeyDown(KeyCode.S)) { SaveJson(); Debug.Log("Interaction JSON manually saved."); }
    }

    void OnTriggerEnter(Collider other)
    {
        float curTime = Time.time;
        if (Time.time - lastCollisionTime < collisionCooldown)
        {
            Debug.Log("Collision During Cooldown");
            return;
        }

        GameObject go = other.gameObject;
        int id = go.GetInstanceID();
        string targetName = go.name;

        // If already tracking this object, ignore duplicate enters
        if (activeEvents.ContainsKey(id))
        {
            Debug.Log($"Enter ignored: already tracking instance {id} ({targetName})");
            return;
        }

        InteractionEvent ev = new InteractionEvent
        {
            start_time = Time.time,
            end_time = Time.time, // will be updated on exit; set to start_time as fallback
            tool = toolName, // use "engage" as fallback if action is not set in the Inspector
            action = string.IsNullOrEmpty(enterActionName) ? "engage" : enterActionName,
            target = targetName
        };

        activeEvents[id] = ev;
        Debug.Log($"[Log] EnterEvent => start: {ev.start_time}, tool: {ev.tool}, target: {ev.target}, action: {ev.action}");

        lastCollisionTime = curTime;
    }

    void OnTriggerExit(Collider other)
    {
        GameObject go = other.gameObject;
        int id = go.GetInstanceID();
        string targetName = go.name;

        if (!activeEvents.ContainsKey(id))
        {
            Debug.Log("No matching enter event for this exit (ignored).");
            return;
        }

        InteractionEvent ev = activeEvents[id];

        // Update end_time
        ev.end_time = Time.time;

        // If exitActionName is provided, optionally update action to the exit verb.
        // Keep enterActionName as the primary action unless user prefers otherwise.
        if (!string.IsNullOrEmpty(exitActionName))
        {
            ev.action = exitActionName;
        }

        completedEvents.Add(ev);
        activeEvents.Remove(id);

        Debug.Log($"[Log] ExitEvent => start: {ev.start_time}, end: {ev.end_time}, tool: {ev.tool}, target: {ev.target}, action: {ev.action}");

        // auto-save policy: save after every N events
        if (completedEvents.Count % 10 == 0) SaveJson();
    }

    // To avoid incomplete logs: if the tool is touching something when we exit, the simulation should automatically set the end_time and save the file
    void OnApplicationQuit()
    {
        // Close out any active events by setting their end_time to current time before saving
        float now = Time.time;
        foreach (var kv in activeEvents)
        {
            InteractionEvent ev = kv.Value;
            if (ev.end_time <= ev.start_time)
            {
                ev.end_time = now;
            }
            completedEvents.Add(ev);
        }
        activeEvents.Clear();

        SaveJson();
        Debug.Log("Interaction JSON saved on application quit.");
    }

    public void SaveJson()
    {
        InteractionFile file = new InteractionFile();
        file.metadata = new Metadata(); // default values; adjust if necessary
        file.data = completedEvents;

        string json = JsonUtility.ToJson(file, true);
        string folderPath = Path.Combine(Application.persistentDataPath, "ToolInteractionLog");
        string path = Path.Combine(folderPath, jsonFileName);

        if (!Directory.Exists(folderPath)) Directory.CreateDirectory(folderPath);
        File.WriteAllText(path, json);
        Debug.Log($"Saved interaction log to: {path}");
    }
}
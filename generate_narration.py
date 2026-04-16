"""
generate_narration.py
─────────────────────
Standalone conversion of the Final_Biomistral.ipynb pipeline.

Pipeline
────────
  surgery_events.json
        │
        ▼
  SimpleMediverseFormatter
     • Normalises start_time/end_time OR legacy timestamp field
     • Filters out non-anatomical targets (table, wall, etc.)
     • Anatomy encoder  → human-readable body part names
     • Timestamp encoder → groups repeated actions within 5 s
        │
        ▼
  BioMistral-7B (HuggingFace: BioMistral/BioMistral-7B)
     • Generates a short clinical narration paragraph
     • Strips the prompt prefix from the output
        │
        ▼
  Hallucination detector (spaCy)
     • Flags sentences whose actions/tools are unrecognised
     • Prints warnings but does NOT block the pipeline
        │
        ▼
  Sentence → Timestamp mapper (NLTK)
     • Tokenises narration into individual sentences
     • Maps each sentence to the event-group timestamp
     • Produces start_time / end_time / timestamp / sentence
        │
        ▼
  final_mapped.json   ← consumed by generate_step_audio.py

Usage examples
──────────────
  # Basic (uses surgery_events.json in the same folder)
  python generate_narration.py

  # Explicit paths
  python generate_narration.py \\
      --events   surgery_events.json \\
      --output   final_mapped.json \\
      --model    BioMistral/BioMistral-7B

  # Use a fully local model directory (no internet required)
  python generate_narration.py --model ./local_biomistral

  # Dry-run: format events and print the prompt WITHOUT running the LLM
  python generate_narration.py --dry-run

Dependencies
────────────
  pip install transformers accelerate sentencepiece torch spacy nltk
  python -m spacy download en_core_web_sm
  python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
"""

import argparse
import json
import os
import re
import sys
import warnings
from typing import List, Optional, Tuple

# Silence noisy HuggingFace / torch deprecation warnings at import time
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Constants ─────────────────────────────────────────────────────────────────

# Targets that are not anatomy (should be filtered out before prompting)
NON_ANATOMY_PATTERNS = [
    "table", "wall", "floor", "ceiling", "tray", "stand",
    "clinic_table", "instrument", "drape",
]

# ── SimpleMediverseFormatter ──────────────────────────────────────────────────

class SimpleMediverseFormatter:
    """
    Converts raw VR event logs into a BioMistral-ready clinical prompt.

    Faithfully extracted from Final_Biomistral.ipynb with the following
    additions:
      • Accepts both surgery_events.json format (start_time/end_time) and
        legacy CombinedToolLog.json format (timestamp).
      • Filters out non-anatomical collision targets (e.g. tables).
      • Expanded action map for surgery_events.json action names.
    """

    # ── Encoders ──────────────────────────────────────────────────────────────

    ANATOMY_MAP = {
        "footskin":            "foot skin",
        "metararsal1":         "first metatarsal",
        "metatarsal1":         "first metatarsal",
        "metatarsal1_remainingpiece": "first metatarsal (distal fragment)",
        "proximalphalanx1":    "proximal phalanx of first toe",
        "footr":               "right foot",
        "footl":               "left foot",
        "foot":                "foot",
        "unknown":             "anatomical region",
    }

    TOOL_MAP = {
        "Scalpel":          "scalpel",
        "BoneSaw":          "bone saw",
        "OscillatingSaw":   "oscillating saw",
        "Drill":            "drill",
        "Marker":           "marking pen",
        "Forceps":          "forceps",
        "Retractor":        "retractor",
    }

    # Maps raw action strings from surgery_events.json and CombinedToolLog.json
    # to a simple verb phrase used when building step descriptions.
    ACTION_VERB_MAP = {
        # surgery_events.json style
        "scalpel_incise":       "incision",
        "bonessaw_engage":      "osteotomy",
        "bonessaw_cut":         "osteotomy",
        "bonesaw_engage":       "osteotomy",
        "bonesaw_cut":          "osteotomy",
        "chevroncut_engage":    "osteotomy",
        "oscillatingsaw_engage": "osteotomy",
        "drill_insert":         "drilling",
        "drill_engage":         "drilling",
        # CombinedToolLog style
        "cutting_start":        "incision",
        "drilling_start":       "drilling",
        "drawing_start":        "marking",
        "sawing_start":         "osteotomy",
    }

    def encode_anatomy(self, target: str) -> str:
        """Map a raw Unity target name to a human-readable medical term."""
        t = target.lower().replace(" ", "").replace("(", "").replace(")", "")
        if t in self.ANATOMY_MAP:
            return self.ANATOMY_MAP[t]
        for key, val in self.ANATOMY_MAP.items():
            if key in t:
                return val
        return "anatomical structure"

    def _action_verb(self, action: str) -> str:
        """Return a clean verb phrase for a raw action string."""
        key = action.lower().replace(" ", "_")
        if key in self.ACTION_VERB_MAP:
            return self.ACTION_VERB_MAP[key]
        # Fallback: look for keywords
        if any(k in key for k in ("incis", "cut", "scalpel")):
            return "incision"
        if any(k in key for k in ("saw", "osteotom", "chevron")):
            return "osteotomy"
        if any(k in key for k in ("drill",)):
            return "drilling"
        if any(k in key for k in ("mark", "draw")):
            return "marking"
        return action.lower()

    def _normalise_entry(self, entry: dict) -> dict:
        """
        Normalise a single event entry to always have:
          start_time, end_time, timestamp, tool, action, target
        Works for both surgery_events.json and CombinedToolLog.json formats.
        """
        # Prefer start_time; fall back to timestamp field
        start = float(entry.get("start_time", entry.get("timestamp", 0.0)))
        end   = float(entry.get("end_time",   entry.get("timestamp", start)))
        return {
            "start_time": start,
            "end_time":   end,
            "timestamp":  start,
            "tool":       entry.get("tool",   ""),
            "action":     entry.get("action", ""),
            "target":     entry.get("target", entry.get("bodypart", "unknown")),
        }

    def _is_non_anatomy(self, target: str) -> bool:
        """Return True if the target is a non-anatomical object (table, wall…)."""
        t = target.lower()
        return any(pat in t for pat in NON_ANATOMY_PATTERNS)

    def encode_timestamps(self, entries: List[dict]) -> List[dict]:
        """
        Sort entries by start_time, filter out non-anatomy targets, then
        group consecutive entries that share the same action + target and
        are within 5 seconds of each other.
        """
        normalised = [self._normalise_entry(e) for e in entries]
        # Filter non-anatomy
        normalised = [e for e in normalised if not self._is_non_anatomy(e["target"])]
        # Sort by start time
        normalised.sort(key=lambda x: x["start_time"])

        grouped: List[dict] = []
        i = 0
        while i < len(normalised):
            current = normalised[i]
            similar = [current]

            j = i + 1
            while j < len(normalised):
                nxt      = normalised[j]
                time_gap = nxt["start_time"] - current["start_time"]
                same_act = nxt["action"] == current["action"]
                same_tgt = nxt["target"] == current["target"]
                if time_gap <= 5 and same_act and same_tgt:
                    similar.append(nxt)
                    j += 1
                else:
                    break

            if len(similar) > 1:
                grouped.append({
                    "start_time": current["start_time"],
                    "end_time":   similar[-1]["end_time"],
                    "timestamp":  current["start_time"],
                    "tool":       current["tool"],
                    "action":     current["action"],
                    "target":     current["target"],
                    "count":      len(similar),
                    "grouped":    True,
                })
            else:
                grouped.append(current)

            i = j if j > i + 1 else i + 1

        return grouped

    def convert_to_medical_step(self, entry: dict) -> str:
        """Convert one (possibly grouped) event entry to a plain English step."""
        tool          = entry.get("tool", "")
        action        = entry.get("action", "")
        target        = entry.get("target", "unknown")
        medical_target = self.encode_anatomy(target)
        clean_tool     = self.TOOL_MAP.get(tool, tool.lower())
        verb           = self._action_verb(action)
        count          = entry.get("count", 1)
        is_grouped     = entry.get("grouped", False)

        if is_grouped and count > 1:
            if verb == "incision":
                return f"Multiple incisions made on {medical_target} using {clean_tool} ({count} cuts)"
            if verb == "osteotomy":
                return f"Multiple osteotomies on {medical_target} using {clean_tool} ({count} passes)"
            if verb == "drilling":
                return f"Multiple drilling procedures on {medical_target} using {clean_tool} ({count} holes)"
            return f"Multiple {verb} actions on {medical_target} using {clean_tool} ({count}×)"

        if verb == "incision":
            return f"Incision made on {medical_target} using {clean_tool}"
        if verb == "osteotomy":
            return f"Osteotomy performed on {medical_target} using {clean_tool}"
        if verb == "drilling":
            return f"Drilling procedure performed on {medical_target} using {clean_tool}"
        if verb == "marking":
            return f"Surgical site marked on {medical_target}"
        return f"Surgical procedure ({verb}) on {medical_target} using {clean_tool}"

    def format_for_biomistral(
        self, json_data: dict
    ) -> Tuple[List[dict], List[str], str]:
        """
        Main entry point.

        Returns:
          grouped_entries  — list of normalised, grouped event dicts
                             (used later for timestamp mapping)
          medical_steps    — list of plain-English step strings
          prompt           — full BioMistral-ready prompt string
        """
        raw_entries = (
            json_data.get("data")
            or json_data.get("Items")
            or (json_data if isinstance(json_data, list) else [])
        )
        grouped_entries = self.encode_timestamps(raw_entries)
        medical_steps   = [self.convert_to_medical_step(e) for e in grouped_entries]

        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(medical_steps))

        prompt = (
            "You are a clinical documentation AI trained on surgical protocols "
            "and biomedical literature.\n\n"
            "Task: Convert the following surgical steps into a brief, clinical "
            "narration suitable for a surgical operative note.\n\n"
            "Instructions:\n"
            "- Do NOT invent or infer any patient details, anatomical locations, "
            "or conditions.\n"
            "- Do NOT describe any complications, findings, or outcomes unless "
            "specified.\n"
            "- Use only the surgical actions provided below.\n"
            "- Use precise medical terminology in a concise, professional tone.\n\n"
            f"Steps:\n{steps_text}\n\n"
            "Clinical Narration:"
        )

        return grouped_entries, medical_steps, prompt


# ── BioMistral model ──────────────────────────────────────────────────────────

def load_model(model_name_or_path: str, offload_folder: str):
    """
    Load BioMistral-7B tokenizer + model.

    offload_folder is used for CPU weight offloading when VRAM is insufficient
    (matches the existing /offload directory in the project).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"   Loading tokenizer from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"   Loading model (this takes a minute on first run)…")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.float16,          # half precision — saves ~7 GB VRAM
        device_map="auto",            # automatically distribute across GPU/CPU
        offload_folder=offload_folder if os.path.isdir(offload_folder) else None,
    )
    return tokenizer, model


def generate_narration(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    """
    Run BioMistral on the formatted prompt and return only the narration text
    (everything after 'Clinical Narration:').
    """
    import torch

    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip prompt prefix — keep only the generated narration
    if "Clinical Narration:" in full_text:
        narration = full_text.split("Clinical Narration:", 1)[1].strip()
    else:
        narration = full_text.strip()

    return narration


# ── Hallucination detection ───────────────────────────────────────────────────

NOMENCLATURE = {
    "drawing":       ("Marking",        ["Skin marking", "Demarcation"]),
    "incision":      ("Incision",       ["Cutaneous incision"]),
    "cutting":       ("Cutting",        ["Division"]),
    "sawing":        ("Osteotomy",      ["Sawing", "Bone cutting"]),
    "drilling":      ("Drilling",       ["Burring"]),
    "cauterization": ("Cauterization",  ["Coagulation", "Electrocautery"]),
    "irrigation":    ("Irrigation",     ["Lavage"]),
    "suction":       ("Suctioning",     ["Aspiration"]),
    "retraction":    ("Retraction",     ["Exposure"]),
    "closure":       ("Closure",        ["Wound closure"]),
    "suturing":      ("Suturing",       ["Approximation"]),
    "dissection":    ("Dissection",     []),
    "removal":       ("Removal",        ["Extraction"]),
    "insertion":     ("Insertion",      ["Placement"]),
    "excision":      ("Excision",       ["Resection"]),
}

VERBS = {
    "incision":  ("Incised",   ["Opened", "incised", "incision"]),
    "cutting":   ("Cut",       ["Divided", "degloved"]),
    "sawing":    ("Sawed",     ["Osteotomized", "Cut through", "osteotom"]),
    "drilling":  ("Drilled",   ["Burred", "drill", "drilling"]),
    "marking":   ("Marked",    ["Outlined", "Demarcated"]),
    "closure":   ("Closed",    ["Wound closed"]),
    "suturing":  ("Sutured",   ["Approximated"]),
    "removal":   ("Removed",   ["Extracted"]),
    "insertion": ("Inserted",  ["Placed"]),
}

KNOWN_TOOLS = {
    "scalpel", "bone saw", "oscillating saw", "drill",
    "forceps", "retractor", "suction", "irrigator", "marking pen",
}


def detect_hallucinations(narration_text: str) -> List[str]:
    """
    Use spaCy sentence tokenisation to flag sentences whose actions or
    tools are not recognised against the known nomenclature.

    Returns a list of warning strings (empty list = no issues found).
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        return ["spaCy model 'en_core_web_sm' not found — skipping hallucination check. "
                "Run: python -m spacy download en_core_web_sm"]

    doc       = nlp(narration_text)
    sentences = list(doc.sents)
    issues    = []

    for i, sent in enumerate(sentences, 1):
        text = sent.text.strip()

        action_found = any(
            re.search(rf"\b{act}\w*\b", text, re.IGNORECASE)
            for act in NOMENCLATURE
        ) or any(
            any(re.search(rf"\b{alias.lower()}\b", text.lower()) for alias in [v] + aliases)
            for act, (v, aliases) in VERBS.items()
        )

        tool_found = any(tool in text.lower() for tool in KNOWN_TOOLS)

        if not action_found:
            issues.append(f"  ⚠  Sentence {i}: no recognised surgical action → \"{text[:80]}\"")
        if not tool_found:
            issues.append(f"  ℹ  Sentence {i}: no surgical tool mentioned → \"{text[:80]}\"")

    return issues


# ── Sentence → timestamp mapper ───────────────────────────────────────────────

def map_sentences_to_timestamps(
    sentences:       List[str],
    grouped_entries: List[dict],
) -> List[dict]:
    """
    Sequentially map narration sentences to event-group timestamps.

    If sentence count == group count: direct 1-to-1 mapping.
    If counts differ: distribute sentences across available timestamps
    using round-robin (same strategy the notebook used).

    Output format matches what generate_step_audio.py expects:
      {start_time, end_time, timestamp, sentence}
    """
    n_sentences = len(sentences)
    n_groups    = len(grouped_entries)

    if n_sentences != n_groups:
        print(f"  ⚠  {n_sentences} sentence(s) vs {n_groups} event group(s) — "
              f"distributing sentences across available timestamps.")

    result = []
    for i, sentence in enumerate(sentences):
        # Cycle through groups if more sentences than groups
        group = grouped_entries[i % n_groups]
        result.append({
            "start_time": group["start_time"],
            "end_time":   group["end_time"],
            "timestamp":  group["start_time"],
            "sentence":   sentence,
        })

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Generate clinical narration from VR surgery event logs using BioMistral-7B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--events", "-e",
        default=os.path.join(here, "surgery_events.json"),
        help="Path to the VR event log JSON (default: %(default)s)")
    parser.add_argument("--output", "-o",
        default=os.path.join(here, "final_mapped.json"),
        help="Output path for final_mapped.json (default: %(default)s)")
    parser.add_argument("--model", "-m",
        default="BioMistral/BioMistral-7B",
        help="HuggingFace model name or local path (default: %(default)s)")
    parser.add_argument("--offload-folder",
        default=os.path.join(here, "offload"),
        help="CPU offload folder for large model weights (default: %(default)s)")
    parser.add_argument("--max-new-tokens", type=int, default=300,
        help="Maximum new tokens for generation (default: 300)")
    parser.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature for generation (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true",
        help="Format events and print the prompt, but do NOT load/run the LLM.")
    parser.add_argument("--skip-hallucination-check", action="store_true",
        help="Skip the spaCy hallucination detection step.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  MediverseVR — Narration Generator (BioMistral)")
    print("=" * 60)

    # ── 1. Load events ───────────────────────────────────────────────
    if not os.path.exists(args.events):
        print(f"\n✖  Events file not found: {args.events}")
        sys.exit(1)

    print(f"\n📂 Loading events: {args.events}")
    with open(args.events, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # ── 2. Format with SimpleMediverseFormatter ──────────────────────
    print("\n🔧 Formatting surgical events…")
    formatter = SimpleMediverseFormatter()
    grouped_entries, medical_steps, prompt = formatter.format_for_biomistral(json_data)

    print(f"\n   Formatted {len(grouped_entries)} event group(s):")
    for i, step in enumerate(medical_steps, 1):
        grp = grouped_entries[i - 1]
        print(f"   {i}. [{grp['start_time']:.2f}s] {step}")

    print(f"\n{'─'*60}")
    print("BIOMISTRAL PROMPT:")
    print('─'*60)
    print(prompt)
    print('─'*60)

    # ── Dry-run: stop here ───────────────────────────────────────────
    if args.dry_run:
        print("\n✅ Dry-run complete — LLM not loaded.")
        return

    # ── 3. Load model and generate narration ─────────────────────────
    print(f"\n🤖 Loading BioMistral model: {args.model}")
    try:
        tokenizer, model = load_model(args.model, args.offload_folder)
    except Exception as exc:
        print(f"\n✖  Failed to load model: {exc}")
        print("   Make sure you have transformers, accelerate, sentencepiece installed.")
        print("   Try: pip install transformers accelerate sentencepiece")
        sys.exit(1)

    print("\n⏳ Generating narration…")
    narration = generate_narration(
        prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"\n{'─'*60}")
    print("GENERATED NARRATION:")
    print('─'*60)
    print(narration)
    print('─'*60)

    # ── 4. Hallucination detection ───────────────────────────────────
    if not args.skip_hallucination_check:
        print("\n🔍 Running hallucination check…")
        issues = detect_hallucinations(narration)
        if issues:
            print(f"   Found {len(issues)} potential issue(s):")
            for issue in issues:
                print(issue)
            print("   ℹ  These are warnings only — pipeline will continue.")
        else:
            print("   ✔  No hallucination flags.")

    # ── 5. Sentence tokenisation ─────────────────────────────────────
    print("\n✂️  Tokenising narration into sentences…")
    try:
        import nltk
        # Download punkt silently if not present
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        sentences = nltk.sent_tokenize(narration)
    except Exception as exc:
        print(f"   NLTK unavailable ({exc}) — falling back to simple sentence split.")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narration) if s.strip()]

    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"   {len(sentences)} sentence(s) detected.")
    for i, s in enumerate(sentences, 1):
        print(f"   {i}. {s}")

    # ── 6. Map sentences to timestamps ──────────────────────────────
    print("\n🗺  Mapping sentences to event timestamps…")
    final_output = map_sentences_to_timestamps(sentences, grouped_entries)

    print(f"\n   {'#':<4} {'Start':>8}   {'End':>8}   Sentence")
    print(f"   {'─'*4} {'─'*8}   {'─'*8}   {'─'*40}")
    for i, entry in enumerate(final_output, 1):
        preview = entry["sentence"][:50] + "…" if len(entry["sentence"]) > 51 else entry["sentence"]
        print(f"   {i:<4} {entry['start_time']:>7.2f}s   {entry['end_time']:>7.2f}s   {preview}")

    # ── 7. Write output ──────────────────────────────────────────────
    print(f"\n💾 Writing → {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"   ✔  final_mapped.json written ({len(final_output)} entries).")
    print("\n✅ Done! Next step: run generate_step_audio.py to create WAV files.")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Batch Coqui TTS: one WAV per narration line from a JSON array, plus a sidecar manifest.

Automatically recognizes `overlay_timeline.json` shape (`time` + `content` arrays) and
collapses it to one segment per non-empty `content`, so the number of WAVs matches the
number of spoken lines (not the number of boundary rows). By default (`strict` on), the
run fails if any line is missing audio or any WAV file is missing on disk.

Language support
────────────────
Pass --language <code> to generate non-English audio. When language is anything
other than "en", the model automatically switches to XTTS v2 (multilingual) and
passes the language code to the TTS engine. Supported language codes:
  en  English (default, uses glow-tts for speed)
  zh  Mandarin Chinese
  hi  Hindi
  es  Spanish
  fr  French
  de  German
  ar  Arabic
  ja  Japanese
  ko  Korean
  pt  Portuguese
  ru  Russian
  it  Italian
  pl  Polish
  tr  Turkish
  nl  Dutch
  cs  Czech
  hu  Hungarian

Defaults favor macOS-friendly English (Gruut phonemizer). Models using eSpeak (e.g. many
VITS checkpoints) need espeak-ng on PATH: brew install espeak-ng

Environment (optional):
  COQUI_TTS_MODEL   Model name (default: tts_models/en/ljspeech/glow-tts)
  COQUI_SPEAKER     Speaker id for multi-speaker models (e.g. VCTK), unset for single-speaker
  COQUI_USE_GPU     "1"/"true" to use CUDA when available
  TTS_AUDIO_DIR     Output directory for WAV files (default: audio_steps)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path


# ── Multilingual TTS configuration ───────────────────────────────────────────

# XTTS v2 handles all non-English languages with good quality.
# Uses a built-in speaker to avoid requiring a voice sample for cloning.
MULTILINGUAL_MODEL  = "tts_models/multilingual/multi-dataset/xtts_v2"
MULTILINGUAL_SPEAKER = "Ana Florence"  # neutral female voice bundled with XTTS v2

# Supported language codes — must be in XTTS v2's language list
SUPPORTED_LANGS = {
    "en", "zh", "hi", "es", "fr", "de", "ar", "ja", "ko", "pt", "ru",
    "it", "pl", "tr", "nl", "cs", "hu",
}


def _wav_stats(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        rate = w.getframerate()
        frames = w.getnframes()
        duration = frames / float(rate) if rate else 0.0
    return {
        "channels": ch,
        "sample_rate": rate,
        "duration_seconds": round(duration, 6),
        "mono": ch == 1,
    }


def _content_hash(text: str, settings: dict) -> str:
    payload = json.dumps(
        {"text": text.strip(), **settings},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_overlay_timeline(entries: list) -> bool:
    """True if JSON looks like overlay_timeline.json: [{time, content: [...]}, ...]."""
    if not entries:
        return False
    first = entries[0]
    if not isinstance(first, dict):
        return False
    if "time" not in first or "content" not in first:
        return False
    return isinstance(first.get("content"), list)


def _normalize_overlay_timeline(entries: list) -> list[dict]:
    """
    Collapse overlay timeline rows into one step per non-empty content (one narration line).
    start_time = row's time; end_time = the next row's time (boundary), matching final_mapped.
    """
    normalized: list[dict] = []
    for i, row in enumerate(entries):
        if not isinstance(row, dict):
            continue
        content = row.get("content") or []
        if not isinstance(content, list):
            continue
        text = " ".join(str(s).strip() for s in content if s).strip()
        if not text:
            continue
        start = row["time"]
        end_time = None
        for j in range(i + 1, len(entries)):
            nxt = entries[j]
            if isinstance(nxt, dict) and "time" in nxt:
                end_time = nxt["time"]
                break
        if end_time is None:
            end_time = float(start) + 1.0
        normalized.append(
            {
                "start_time": float(start),
                "end_time": float(end_time),
                "timestamp": float(start),
                "sentence": text,
            }
        )
    return normalized


def _normalize_input_entries(entries: list) -> tuple[list[dict], str]:
    """Return (flat step list, description of transform for manifest)."""
    if _is_overlay_timeline(entries):
        out = _normalize_overlay_timeline(entries)
        return out, f"overlay_timeline → {len(out)} narration segment(s) from {len(entries)} row(s)"
    return entries, "passthrough"


def _resolve_text_field(entry: dict, explicit: str | None) -> tuple[str, str]:
    if explicit:
        if explicit not in entry:
            raise KeyError(f"Text field {explicit!r} missing from entry: {entry!r}")
        return explicit, str(entry[explicit] or "").strip()

    if entry.get("sentence"):
        return "sentence", str(entry["sentence"]).strip()
    if entry.get("narration"):
        return "narration", str(entry["narration"]).strip()

    raise ValueError(
        "No usable text: expected non-empty 'sentence' or 'narration' "
        "(use --text-field to pick another key)."
    )


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _load_tts(model_name: str, gpu: bool):
    from TTS.api import TTS

    return TTS(model_name=model_name, progress_bar=False, gpu=gpu)


# XTTS v2 enforces per-language character limits. Sentences longer than the
# limit get truncated by Coqui (silently — only a warning is printed). To
# preserve the full narration we chunk over-limit text on sentence/comma
# boundaries, generate audio for each chunk, then concatenate the WAVs.
#
# Limits sourced from Coqui TTS source: TTS/tts/layers/xtts/tokenizer.py
XTTS_CHAR_LIMITS = {
    "en": 250, "es": 239, "fr": 273, "de": 253, "it": 213, "pt": 203,
    "pl": 224, "tr": 226, "ru": 182, "nl": 251, "cs": 186, "ar": 166,
    "zh": 82,  "hu": 224, "ko": 95,  "ja": 71,  "hi": 150,
}


def _split_text_into_chunks(text: str, max_chars: int) -> list:
    """
    Split a long sentence into chunks no larger than max_chars, breaking
    on punctuation in priority order: sentence enders, commas, then spaces.
    Each returned chunk is a complete-sounding piece — never breaks
    mid-word.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Prefer to split on these characters, in this order. We also include
    # CJK punctuation so this works for Mandarin/Japanese.
    BREAK_CHARS = ['. ', '。', '! ', '? ', '？', '！', ', ', '，', '; ', '：', ' ']

    chunks = []
    remaining = text
    while len(remaining) > max_chars:
        # Find the latest break char within the limit window
        cut = -1
        for breaker in BREAK_CHARS:
            idx = remaining.rfind(breaker, 0, max_chars)
            if idx > cut:
                cut = idx + len(breaker)
        # If no good break point, hard-cut at the limit
        if cut <= 0:
            cut = max_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _trim_trailing_hallucination(wav_path, min_silence_ms: int = 400,
                                  silence_threshold: int = 500) -> None:
    """
    Trim trailing XTTS hallucination from a WAV file in place.

    XTTS sometimes appends 2-5 seconds of nonsense phonemes after the actual
    translation ends. There's typically a brief pause (300-500ms of silence)
    between the real speech and the hallucinated tail.

    This function finds the LAST stretch of silence ≥ min_silence_ms and
    truncates everything after it. The result keeps real speech intact while
    removing the hallucinated tail. If no such silence exists (clip is all
    real speech), the WAV is left unchanged.
    """
    import struct

    with wave.open(str(wav_path), "rb") as w:
        params = w.getparams()
        audio  = w.readframes(w.getnframes())

    sample_rate  = params.framerate
    sample_width = params.sampwidth
    n_channels   = params.nchannels
    n_samples    = len(audio) // (sample_width * n_channels)

    if n_samples == 0:
        return

    samples = struct.unpack(f"<{n_samples * n_channels}h", audio)

    # Build a per-sample "is silent" boolean mask
    silence_mask = [abs(s) < silence_threshold for s in samples]

    # Find the last "long enough" silent stretch
    min_silence_samples = int(sample_rate * min_silence_ms / 1000)

    # Walk backwards, looking for a run of silent samples ≥ min_silence_samples
    cut_at = None
    run_length = 0
    for i in range(len(samples) - 1, -1, -1):
        if silence_mask[i]:
            run_length += 1
            if run_length >= min_silence_samples:
                # Found a long silence; cut at the START of this silence
                # (i.e. the end of the speech that precedes it).
                # +run_length puts us back at the silence's start moving forward.
                cut_at = i + run_length
                # Now look for any non-silent sample BEFORE this silence —
                # if there isn't one, the silence is at the start, not a
                # speech-then-silence-then-hallucination pattern.
                has_speech_before = any(not silence_mask[j] for j in range(i))
                if has_speech_before:
                    break
                else:
                    # No speech before — this isn't a hallucination boundary
                    cut_at = None
                    run_length = 0
        else:
            run_length = 0

    if cut_at is None:
        # No long silence found — leave the file alone
        return

    # Keep a small tail (200ms) past the speech-end for natural fade-out
    keep_tail = int(sample_rate * 0.2)
    end_idx = min(len(samples), cut_at + keep_tail - run_length)

    # Only rewrite if we'd actually save meaningful time (≥ 500ms cut)
    saved_samples = len(samples) - end_idx
    if saved_samples < int(sample_rate * 0.5):
        return

    saved_seconds = saved_samples / sample_rate
    print(f"    ↳ Trimmed {saved_seconds:.2f}s of trailing audio (likely XTTS hallucination)")

    trimmed = audio[:end_idx * sample_width * n_channels]
    with wave.open(str(wav_path), "wb") as out:
        out.setparams(params)
        out.writeframes(trimmed)


def _concatenate_wavs(input_paths: list, output_path) -> None:
    """
    Concatenate multiple WAV files into one. Uses Python's wave module
    (no ffmpeg dependency). All inputs must share the same format —
    XTTS v2 always outputs 24kHz mono int16 so this is safe.

    To avoid audible noise at chunk boundaries (XTTS produces brief noisy
    "tails" at the end of each generation), we:
      1. Trim trailing low-energy samples from each chunk except the last
      2. Insert a short (180 ms) silent gap between chunks for natural flow
    """
    if len(input_paths) == 1:
        Path(input_paths[0]).rename(output_path)
        return

    import struct

    # Read all chunks
    chunk_audio = []
    params      = None
    for path in input_paths:
        with wave.open(str(path), "rb") as w:
            if params is None:
                params = w.getparams()
            chunk_audio.append(w.readframes(w.getnframes()))

    sample_rate    = params.framerate
    sample_width   = params.sampwidth      # bytes per sample (2 for int16)
    n_channels     = params.nchannels

    # Threshold for "silence" — anything below this absolute amplitude
    # is treated as background noise to be trimmed. 500 / 32768 ≈ 1.5%
    # of full scale, low enough to preserve quiet speech tails.
    SILENCE_THRESHOLD = 500
    GAP_MS            = 180          # gap between chunks
    gap_samples       = int(sample_rate * GAP_MS / 1000)
    gap_bytes         = b"\x00\x00" * gap_samples * n_channels

    def trim_trailing_silence(audio_bytes: bytes) -> bytes:
        """Strip trailing samples below the silence threshold."""
        # int16 samples
        n_samples = len(audio_bytes) // (sample_width * n_channels)
        # Walk backwards and find the last non-silent sample
        last_loud_idx = 0
        # Decode chunked to avoid loading huge ints — but our clips are short.
        samples = struct.unpack(f"<{n_samples * n_channels}h", audio_bytes)
        for i in range(len(samples) - 1, -1, -1):
            if abs(samples[i]) > SILENCE_THRESHOLD:
                last_loud_idx = i + 1
                break
        # Keep a small tail (50 ms) so words don't get clipped mid-phoneme
        keep_tail_samples = int(sample_rate * 0.05)
        last_loud_idx = min(len(samples), last_loud_idx + keep_tail_samples)
        return audio_bytes[:last_loud_idx * sample_width]

    # Trim every chunk except the last (let the natural ending breathe)
    cleaned = []
    for i, audio in enumerate(chunk_audio):
        if i < len(chunk_audio) - 1:
            cleaned.append(trim_trailing_silence(audio))
        else:
            cleaned.append(audio)

    # Stitch: chunk1 + gap + chunk2 + gap + ... + chunkN
    final = cleaned[0]
    for piece in cleaned[1:]:
        final += gap_bytes + piece

    # Write output
    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        out.writeframes(final)

    # Clean up per-chunk WAVs
    for path in input_paths:
        try:
            Path(path).unlink()
        except OSError:
            pass


def _tts_to_file_chunked(tts, text: str, out_path, language: str | None,
                          speaker: str | None, base_kwargs: dict) -> None:
    """
    Wrapper around tts.tts_to_file that chunks text exceeding XTTS v2's
    per-language character limit, generates one WAV per chunk, then
    concatenates them into the final output_path.
    """
    char_limit = XTTS_CHAR_LIMITS.get(language, 250) if language else 250

    # Apply post-generation hallucination trimming for non-English (XTTS v2)
    # but skip it for English glow-tts which doesn't have this issue.
    should_trim = language is not None and language != "en"

    if len(text) <= char_limit:
        # Fast path — single call, no chunking
        kwargs = dict(base_kwargs)
        kwargs["text"] = text
        kwargs["file_path"] = str(out_path)
        if speaker:
            kwargs["speaker"] = speaker
        if language:
            kwargs["language"] = language
        tts.tts_to_file(**kwargs)
        if should_trim:
            _trim_trailing_hallucination(out_path)
        return

    # Long text — chunk, render each piece, then concatenate
    chunks = _split_text_into_chunks(text, char_limit)
    print(f"    ↳ Text exceeds {char_limit}-char limit for '{language}'. "
          f"Splitting into {len(chunks)} chunk(s).")

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = out_path.parent / f"{out_path.stem}_part{i:02d}.wav"
        kwargs = dict(base_kwargs)
        kwargs["text"] = chunk
        kwargs["file_path"] = str(chunk_path)
        if speaker:
            kwargs["speaker"] = speaker
        if language:
            kwargs["language"] = language
        tts.tts_to_file(**kwargs)
        if should_trim:
            _trim_trailing_hallucination(chunk_path)
        chunk_paths.append(chunk_path)

    _concatenate_wavs(chunk_paths, out_path)


def _resolve_model_and_settings(
    requested_model: str,
    language: str,
    requested_speaker: str | None,
) -> tuple[str, str | None, dict]:
    """
    Given the user's requested model and language, return the effective
    (model_name, speaker, tts_kwargs_extra) to use.

    Logic:
      - language == "en"   → keep whatever model the user asked for (glow-tts by default)
      - language != "en"   → force XTTS v2 multilingual model with a built-in speaker,
                             and pass language=XX to tts_to_file()
    """
    tts_kwargs_extra: dict = {}
    if language and language != "en":
        if language not in SUPPORTED_LANGS:
            raise ValueError(
                f"Language '{language}' not supported. Choose from: "
                + ", ".join(sorted(SUPPORTED_LANGS))
            )
        # Override the model regardless of what the user passed, since
        # non-English requires a multilingual model.
        effective_model   = MULTILINGUAL_MODEL
        effective_speaker = requested_speaker or MULTILINGUAL_SPEAKER
        tts_kwargs_extra["language"] = language
        return effective_model, effective_speaker, tts_kwargs_extra

    # English — keep whatever the user requested
    return requested_model, requested_speaker, tts_kwargs_extra


def _validate_audio_manifest(
    manifest_steps: list[dict],
    expected_clips: int,
    manifest_path: Path,
) -> None:
    """Ensure one successful WAV per expected narration segment."""
    failed = [s for s in manifest_steps if s.get("error")]
    if failed:
        msg = failed[0].get("error", "unknown")
        raise RuntimeError(
            f"TTS failed for {len(failed)} step(s); first: step_index={failed[0].get('step_index')!r} {msg!r}"
        )
    with_audio = [s for s in manifest_steps if s.get("audio_path")]
    if len(with_audio) != expected_clips:
        raise RuntimeError(
            f"Text/audio mismatch: expected {expected_clips} audio clip(s), "
            f"manifest has {len(with_audio)} (total steps recorded: {len(manifest_steps)})."
        )
    base = manifest_path.parent.resolve()
    for s in with_audio:
        rel = s["audio_path"]
        p = Path(rel)
        if not p.is_absolute():
            p = base / p
        if not p.is_file():
            raise RuntimeError(f"Missing WAV for step_index={s.get('step_index')!r}: {p}")


def generate(
    *,
    input_json: Path,
    manifest_path: Path,
    audio_dir: Path,
    text_field: str | None,
    model_name: str,
    speaker: str | None,
    use_gpu: bool,
    force: bool,
    strict: bool,
    language: str = "en",
) -> dict:
    with input_json.open(encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("Input JSON must be an array of step objects.")

    entries, input_note = _normalize_input_entries(entries)
    expected_clips = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        try:
            _, text = _resolve_text_field(e, text_field)
            if text:
                expected_clips += 1
        except ValueError:
            pass

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the effective model/speaker for the requested language
    effective_model, effective_speaker, tts_kwargs_extra = _resolve_model_and_settings(
        model_name, language, speaker
    )

    print(f"  Language: {language}")
    print(f"  Model:    {effective_model}")
    if effective_speaker:
        print(f"  Speaker:  {effective_speaker}")

    hash_settings = {
        "provider": "coqui",
        "model_name": effective_model,
        "speaker": effective_speaker or "",
        "language": language,
        "gpu": use_gpu,
        "split_sentences": False,
    }

    manifest_steps: list[dict] = []
    tts = None

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {idx} must be an object, got {type(entry)}")

        step_id = f"step_{idx:04d}"
        try:
            field_name, text = _resolve_text_field(entry, text_field)
        except ValueError as e:
            manifest_steps.append(
                {
                    "step_index": idx,
                    "step_id": step_id,
                    "error": str(e),
                    "source_entry": entry,
                }
            )
            continue

        if not text:
            manifest_steps.append(
                {
                    "step_index": idx,
                    "step_id": step_id,
                    "text_field": field_name,
                    "text": text,
                    "error": "empty text after strip",
                    "source_entry": entry,
                }
            )
            continue

        digest = _content_hash(text, hash_settings)
        short = digest[:16]
        filename = f"{step_id}_{short}.wav"
        out_path = audio_dir / filename
        rel_audio = str(out_path.as_posix())
        try:
            rel_audio = str(out_path.relative_to(manifest_path.parent.resolve()))
        except ValueError:
            pass

        cached = out_path.is_file() and not force
        if cached:
            stats = _wav_stats(out_path)
            manifest_steps.append(
                {
                    "step_index": idx,
                    "step_id": step_id,
                    "text_field": field_name,
                    "text": text,
                    "content_hash": digest,
                    "audio_path": rel_audio,
                    "reused_cached_audio": True,
                    "tts_settings": dict(hash_settings),
                    **stats,
                    "source_times": {
                        k: entry.get(k)
                        for k in ("start_time", "end_time", "timestamp")
                        if k in entry
                    },
                }
            )
            continue

        if tts is None:
            tts = _load_tts(effective_model, gpu=use_gpu)

        # Base kwargs that apply to every call (chunked or not)
        base_kwargs = {"split_sentences": False}

        # Use the chunked wrapper — it handles XTTS v2's per-language char limits
        # by splitting long text into chunks and concatenating the resulting WAVs.
        # Falls back to a single call when text fits within the limit.
        _tts_to_file_chunked(
            tts          = tts,
            text         = text,
            out_path     = out_path,
            language     = tts_kwargs_extra.get("language"),
            speaker      = effective_speaker,
            base_kwargs  = base_kwargs,
        )
        stats = _wav_stats(out_path)
        manifest_steps.append(
            {
                "step_index": idx,
                "step_id": step_id,
                "text_field": field_name,
                "text": text,
                "content_hash": digest,
                "audio_path": rel_audio,
                "reused_cached_audio": False,
                "tts_settings": dict(hash_settings),
                **stats,
                "source_times": {
                    k: entry.get(k)
                    for k in ("start_time", "end_time", "timestamp")
                    if k in entry
                },
            }
        )

    if strict:
        _validate_audio_manifest(manifest_steps, expected_clips, manifest_path)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "coqui",
        "input_json": str(input_json.as_posix()),
        "input_normalization": input_note,
        "expected_audio_clips": expected_clips,
        "audio_dir": str(audio_dir.as_posix()),
        "language": language,
        "batching": {
            "mode": "sequential",
            "note": "Local Coqui runs sequentially to limit RAM/CPU thrash; one clip per step.",
        },
        "future_enhancements": [
            "Optional medical pronunciation pass (custom lexicon / SSML / post-edit)",
            "Optional upload step: sync audio_dir to object storage and add audio_url",
        ],
        "model_name": effective_model,
        "speaker": effective_speaker,
        "steps": manifest_steps,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("final_mapped.json"),
        help="JSON array of steps (default: final_mapped.json)",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("audio_manifest.json"),
        help="Sidecar manifest path (default: audio_manifest.json)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory for WAV files (default: env TTS_AUDIO_DIR or ./audio_steps)",
    )
    parser.add_argument(
        "--text-field",
        choices=("auto", "sentence", "narration"),
        default="auto",
        help="Which field to speak (default: auto prefers sentence, then narration)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("COQUI_TTS_MODEL", "tts_models/en/ljspeech/glow-tts"),
        help="Coqui model_name (default: glow-tts / env COQUI_TTS_MODEL). "
             "Overridden to XTTS v2 when --language is non-English.",
    )
    parser.add_argument(
        "--speaker",
        default=os.environ.get("COQUI_SPEAKER") or None,
        help="Multi-speaker id (default: env COQUI_SPEAKER)",
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        choices=sorted(SUPPORTED_LANGS),
        help="Language code for the TTS output (default: en). Non-English "
             "auto-switches to XTTS v2 multilingual model.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if Coqui detects CUDA (default: env COQUI_USE_GPU or off)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and regenerate every WAV",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not require every narration line to produce a successful WAV (not recommended)",
    )

    args = parser.parse_args(argv)

    audio_dir = args.audio_dir or Path(os.environ.get("TTS_AUDIO_DIR", "audio_steps"))
    use_gpu = args.gpu or _bool_env("COQUI_USE_GPU", False)

    text_field = None if args.text_field == "auto" else args.text_field

    try:
        result = generate(
            input_json=args.input,
            manifest_path=args.output_manifest,
            audio_dir=audio_dir,
            text_field=text_field,
            model_name=args.model,
            speaker=args.speaker,
            use_gpu=use_gpu,
            force=args.force,
            strict=not args.no_strict,
            language=args.language,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    n = result.get("expected_audio_clips", 0)
    print(f"Wrote manifest: {args.output_manifest} ({n} narration clip(s) == {n} WAV file(s))")
    print(f"Audio directory: {audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
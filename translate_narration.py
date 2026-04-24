"""
translate_narration.py
──────────────────────
Translates final_mapped.json from English into a target language, producing
final_mapped_<lang>.json with translated sentences. Preserves all timing
metadata so the rest of the pipeline (TTS, SRT, video compose) works
unchanged on the translated file.

Uses a local Hugging Face translation model (no API key required, no rate
limits, no internet needed after first download). Default model is Helsinki-
NLP/opus-mt-en-{target} which covers 100+ languages with decent quality.

For medical terminology specifically, the default model is good enough for
a demo but not clinical-grade. For production use, plug in a specialized
medical translation service (Lilt, SYSTRAN Health, or a fine-tuned model).

Usage examples
──────────────
  # Translate to Mandarin Chinese
  python translate_narration.py --language zh

  # Explicit paths
  python translate_narration.py \\
      --input  final_mapped.json \\
      --output final_mapped_zh.json \\
      --language zh

  # Hindi
  python translate_narration.py --language hi

  # Force re-translation even if output exists
  python translate_narration.py --language zh --force

Supported languages (short list — model supports 100+):
  zh   Mandarin Chinese
  hi   Hindi
  es   Spanish
  fr   French
  de   German
  ar   Arabic
  ja   Japanese
  ko   Korean
  pt   Portuguese
  ru   Russian

Output
──────
  final_mapped_<lang>.json  — same structure as final_mapped.json with
                              sentence/narration fields replaced by the
                              translated text. start_time/end_time/etc.
                              are preserved verbatim.
"""

import argparse
import json
import os
import sys
import time


# Hugging Face model IDs for Helsinki-NLP's opus-mt family.
# Most target languages use a direct en-XX model; a few need special handling.
MODEL_MAP = {
    "zh":  "Helsinki-NLP/opus-mt-en-zh",
    "hi":  "Helsinki-NLP/opus-mt-en-hi",
    "es":  "Helsinki-NLP/opus-mt-en-es",
    "fr":  "Helsinki-NLP/opus-mt-en-fr",
    "de":  "Helsinki-NLP/opus-mt-en-de",
    "ar":  "Helsinki-NLP/opus-mt-en-ar",
    "ja":  "Helsinki-NLP/opus-mt-en-jap",
    "ko":  "Helsinki-NLP/opus-mt-en-ko",
    "pt":  "Helsinki-NLP/opus-mt-en-pt",
    "ru":  "Helsinki-NLP/opus-mt-en-ru",
}


def load_translator(language: str):
    """
    Lazy-load the Hugging Face translation pipeline for the given language.
    Imports inside the function so users who don't use this script don't pay
    the transformers import cost.
    """
    if language not in MODEL_MAP:
        print(f"✖  Language '{language}' not supported. Choose from: {', '.join(MODEL_MAP)}")
        sys.exit(1)

    model_id = MODEL_MAP[language]
    print(f"\n🤖 Loading model: {model_id}")
    print(f"   (first run will download ~300 MB — subsequent runs are cached)")

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        print("\n✖  transformers not installed. Run:")
        print("   .venv/bin/pip install transformers sentencepiece")
        sys.exit(1)

    t0 = time.time()
    # Load tokenizer + model directly — avoids the pipeline() API whose task
    # names have changed across transformers versions.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print(f"   ✔  Model loaded ({time.time() - t0:.1f}s)")

    # Return a small callable that mimics the pipeline() interface used below.
    def translator(text: str, max_length: int = 512):
        inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"translation_text": decoded}]

    return translator


def translate_sentences(translator, sentences: list) -> list:
    """
    Translate a list of English sentences into the target language.
    Returns the list of translated strings in the same order.
    Skips empty strings so the model doesn't produce garbage on them.
    """
    results = []
    for i, sent in enumerate(sentences):
        if not sent.strip():
            results.append("")
            continue

        print(f"   [{i+1}/{len(sentences)}] Translating: {sent[:60]}{'…' if len(sent) > 60 else ''}")
        try:
            output = translator(sent, max_length=512)
            translated = output[0]["translation_text"].strip()
            results.append(translated)
            print(f"           →  {translated[:60]}{'…' if len(translated) > 60 else ''}")
        except Exception as exc:
            print(f"   ⚠  Translation failed for sentence {i+1}: {exc}")
            results.append(sent)  # fallback to English so pipeline doesn't break

    return results


def translate_final_mapped(input_path: str, output_path: str, language: str) -> None:
    """
    Read final_mapped.json, translate every sentence/narration field,
    and write the result to output_path preserving all other keys.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"✖  Expected a JSON array in {input_path}, got {type(data).__name__}")
        sys.exit(1)

    # Extract the text field — final_mapped.json uses 'sentence' or 'narration'
    # depending on which version generated it. We write back to whichever key
    # was present originally so downstream scripts don't need to change.
    text_key = None
    for item in data:
        if "sentence" in item:
            text_key = "sentence"
            break
        if "narration" in item:
            text_key = "narration"
            break

    if text_key is None:
        print(f"✖  No 'sentence' or 'narration' field found in {input_path}")
        sys.exit(1)

    print(f"\n📄 Input:      {input_path}")
    print(f"   Items:      {len(data)}")
    print(f"   Text field: '{text_key}'")

    # ── Load model and translate ──────────────────────────────────────
    translator = load_translator(language)

    print(f"\n🌐 Translating {len(data)} sentence(s) → {language}:")
    english_sentences = [item.get(text_key, "") for item in data]
    translated        = translate_sentences(translator, english_sentences)

    # ── Build output: copy each item, replace the text field ──────────
    output_data = []
    for item, new_text in zip(data, translated):
        new_item            = dict(item)           # shallow copy
        new_item[text_key]  = new_text
        new_item["language"] = language            # tag the language for clarity
        new_item["original_english"] = item.get(text_key, "")  # keep the source
        output_data.append(new_item)

    # ── Write output ──────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Output:     {output_path}")
    print(f"   ✔  Wrote {len(output_data)} translated item(s).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Translate final_mapped.json into a target language.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i",
        default=os.path.join(here, "final_mapped.json"),
        help="Input final_mapped.json (default: %(default)s)")
    parser.add_argument("--output", "-o",
        default=None,
        help="Output path. Default: final_mapped_<lang>.json next to the input.")
    parser.add_argument("--language", "-l", required=True,
        choices=list(MODEL_MAP.keys()),
        help="Target language code. Choices: " + ", ".join(MODEL_MAP))
    parser.add_argument("--force", "-f", action="store_true",
        help="Re-translate even if the output file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  MediverseVR — Narration Translator")
    print("=" * 60)

    # ── Resolve output path ──────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        input_dir  = os.path.dirname(os.path.abspath(args.input))
        output_path = os.path.join(input_dir, f"final_mapped_{args.language}.json")

    # ── Input exists? ────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"\n✖  Input file not found: {args.input}")
        print(f"   Run generate_narration.py first to create final_mapped.json.")
        sys.exit(1)

    # ── Output already exists? ───────────────────────────────────────
    if os.path.exists(output_path) and not args.force:
        print(f"\n⏭  Output already exists: {output_path}")
        print(f"   Use --force to regenerate.")
        sys.exit(0)

    translate_final_mapped(args.input, output_path, args.language)

    print("\n✅ Done!")
    print(f"   Next steps:")
    print(f"     1. python generate_step_audio.py --input {output_path} \\")
    print(f"            --audio-dir audio_steps_{args.language}/ \\")
    print(f"            --output-manifest audio_manifest_{args.language}.json \\")
    print(f"            --language {args.language}")
    print(f"     2. python generate_srt.py \\")
    print(f"            --manifest audio_manifest_{args.language}.json \\")
    print(f"            --output captions_{args.language}.srt")
    print(f"     3. python compose_video.py --video demo_recording.mp4 \\")
    print(f"            --manifest audio_manifest_{args.language}.json \\")
    print(f"            --srt captions_{args.language}.srt \\")
    print(f"            --output narrated_{args.language}.mp4 \\")
    print(f"            --no-original-audio")
    print()


if __name__ == "__main__":
    main()

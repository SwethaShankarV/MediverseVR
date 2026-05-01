"""
translate_narration.py
──────────────────────
Translates final_mapped.json from English into a target language. Preserves
all timing metadata so the rest of the pipeline (TTS, SRT, video compose)
works unchanged on the translated file.

Translation backends (in priority order)
────────────────────────────────────────
  1. OpenAI GPT-4    (default, best quality, requires OPENAI_API_KEY)
                     Significantly better medical/technical translation
                     because GPT-4 was trained on enough clinical content
                     to use correct anatomical terminology and clinical
                     register naturally.

  2. NLLB-200 local  (fallback, no internet/key required)
                     Meta's multilingual model — handles 200 languages
                     but produces weaker medical translations.

The script auto-selects:
  - GPT-4    if OPENAI_API_KEY env var is set OR --backend openai is passed
  - NLLB     if OPENAI_API_KEY is missing AND --backend openai not forced
  - You can force either with --backend {openai,nllb}

Usage examples
──────────────
  # Use GPT-4 (default if OPENAI_API_KEY is set)
  export OPENAI_API_KEY=sk-...
  python translate_narration.py --language zh

  # Force NLLB even if API key is set
  python translate_narration.py --language zh --backend nllb

  # Custom GPT-4 model
  python translate_narration.py --language zh --openai-model gpt-4o

Cost (approximate, OpenAI):
  ~$0.01 per 10 sentences with gpt-4o-mini (default — good enough for medical)
  ~$0.10 per 10 sentences with gpt-4o      (highest quality)

Supported languages
───────────────────
GPT-4 supports any natural language. NLLB supports 200 languages — see
NLLB_LANG_CODES below for the codes accepted by --language.
"""

import argparse
import json
import os
import sys
import time


# ── NLLB language codes (used when GPT-4 is not available) ────────────────────
NLLB_LANG_CODES = {
    "zh":    "zho_Hans",   "zh-tw": "zho_Hant",
    "hi":    "hin_Deva",   "es":    "spa_Latn",
    "fr":    "fra_Latn",   "de":    "deu_Latn",
    "ar":    "arb_Arab",   "ja":    "jpn_Jpan",
    "ko":    "kor_Hang",   "pt":    "por_Latn",
    "ru":    "rus_Cyrl",   "it":    "ita_Latn",
    "bn":    "ben_Beng",   "ur":    "urd_Arab",
    "ta":    "tam_Taml",   "te":    "tel_Telu",
    "vi":    "vie_Latn",   "th":    "tha_Thai",
    "id":    "ind_Latn",   "tr":    "tur_Latn",
    "pl":    "pol_Latn",   "nl":    "nld_Latn",
    "sv":    "swe_Latn",   "fi":    "fin_Latn",
    "el":    "ell_Grek",   "he":    "heb_Hebr",
    "fa":    "pes_Arab",   "uk":    "ukr_Cyrl",
}

NLLB_MODELS = {
    "600M":  "facebook/nllb-200-distilled-600M",
    "1.3B":  "facebook/nllb-200-distilled-1.3B",
    "3.3B":  "facebook/nllb-200-3.3B",
}

# Human-readable names for the GPT-4 system prompt
LANG_DISPLAY_NAMES = {
    "zh":    "Simplified Chinese (Mandarin)",
    "zh-tw": "Traditional Chinese (Mandarin)",
    "hi":    "Hindi",
    "es":    "Spanish",
    "fr":    "French",
    "de":    "German",
    "ar":    "Modern Standard Arabic",
    "ja":    "Japanese",
    "ko":    "Korean",
    "pt":    "Brazilian Portuguese",
    "ru":    "Russian",
    "it":    "Italian",
    "bn":    "Bengali",
    "ur":    "Urdu",
    "ta":    "Tamil",
    "te":    "Telugu",
    "vi":    "Vietnamese",
    "th":    "Thai",
    "id":    "Indonesian",
    "tr":    "Turkish",
    "pl":    "Polish",
    "nl":    "Dutch",
    "sv":    "Swedish",
    "fi":    "Finnish",
    "el":    "Greek",
    "he":    "Hebrew",
    "fa":    "Persian (Farsi)",
    "uk":    "Ukrainian",
}


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI GPT-4 backend
# ─────────────────────────────────────────────────────────────────────────────

def load_openai_translator(language: str, model: str = "gpt-4o-mini"):
    """
    Build a translator that calls OpenAI's chat API. Returns translate(text)->str.

    Uses a medical-context system prompt to coax clinically-correct output.
    The prompt is critical here — without it GPT-4 sometimes outputs casual
    register instead of clinical terminology.
    """
    if language not in LANG_DISPLAY_NAMES:
        print(f"✖  Language '{language}' has no display name configured.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("\n✖  openai package not installed. Run:")
        print("   .venv/bin/pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n✖  OPENAI_API_KEY environment variable not set.")
        print("   Set it with:  export OPENAI_API_KEY=sk-...")
        print("   Or use:       --backend nllb  (slower, lower quality, no API key)")
        sys.exit(1)

    client       = OpenAI(api_key=api_key)
    target_name  = LANG_DISPLAY_NAMES[language]

    # XTTS v2 has per-language character limits — exceeding them silently
    # truncates audio. We ask GPT-4 to keep translations within a safe budget
    # so we never need to chunk-and-concatenate (which causes XTTS to
    # hallucinate phonemes on tail fragments). Set to 90% of the XTTS limit
    # to leave headroom for tokenizer differences.
    XTTS_LIMITS_FOR_PROMPT = {
        "en": 230, "es": 215, "fr": 245, "de": 230, "it": 195, "pt": 185,
        "pl": 200, "tr": 205, "ru": 165, "nl": 225, "cs": 170, "ar": 150,
        "zh":  75, "hu": 200, "ko":  85, "ja":  65, "hi": 135,
    }
    char_budget = XTTS_LIMITS_FOR_PROMPT.get(language, 230)

    print(f"\n🤖 Backend:    OpenAI {model}")
    print(f"   Target:     {language} ({target_name})")
    print(f"   Char budget: ≤{char_budget} chars per sentence (XTTS-safe)")
    print(f"   Cost:       ~$0.01–0.10 per 10 sentences")

    SYSTEM_PROMPT = (
        f"You are a professional medical translator. Translate the user's "
        f"English text into {target_name}, preserving:\n"
        f"  - Correct anatomical and surgical terminology (e.g. 'metatarsal' "
        f"    must use the proper anatomical term in {target_name}, not a "
        f"    phonetic transliteration or a generic word like 'foot bone').\n"
        f"  - Clinical register (formal, professional medical voice).\n"
        f"  - Sentence structure and meaning — do not summarize or omit.\n"
        f"\n"
        f"Surgical instruments and procedures — IMPORTANT:\n"
        f"  Use the term that a practicing surgeon in a {target_name}-speaking\n"
        f"  hospital would actually say or write in clinical notes. For surgical\n"
        f"  instruments and procedures, this typically means using the English\n"
        f"  loanword (transliterated into the target script) rather than a\n"
        f"  generic everyday word. Examples of what NOT to do:\n"
        f"    - 'scalpel' should NOT become a generic word for 'knife'.\n"
        f"      Use 'scalpel' transliterated, or the proper surgical term.\n"
        f"    - 'oscillating saw' should NOT become 'vibrating tool'.\n"
        f"      Use the proper surgical instrument name.\n"
        f"    - 'osteotomy' should NOT become 'bone cutting'.\n"
        f"      Use the surgical procedure name.\n"
        f"  When in doubt, prefer the term used in published medical literature\n"
        f"  in {target_name}, not the term used in everyday conversation.\n"
        f"\n"
        f"IMPORTANT — character limit:\n"
        f"  The translation MUST be {char_budget} characters or fewer. This is a hard\n"
        f"  technical constraint of the downstream text-to-speech engine, not a\n"
        f"  stylistic preference. If a faithful translation would exceed this limit,\n"
        f"  rephrase concisely while preserving all medical facts and terminology.\n"
        f"  Prefer clinical brevity over verbose explanation. Count characters\n"
        f"  carefully — exceeding the limit will cause audio truncation.\n"
        f"\n"
        f"Output ONLY the translation. No quotes, no preamble, no explanation, "
        f"no source text, no parenthetical notes. Just the translated sentence."
    )

    def translate(text: str) -> str:
        # First attempt — normal translation
        response = client.chat.completions.create(
            model       = model,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature = 0.2,
            max_tokens  = 512,
        )
        result = response.choices[0].message.content.strip()

        # Safety net: if GPT-4 ignored the limit, retry with explicit
        # shortening instruction. We do up to 2 retries before giving up.
        for attempt in range(2):
            if len(result) <= char_budget:
                return result
            print(f"    ↳ Translation is {len(result)} chars (budget {char_budget}). "
                  f"Asking GPT-4 to shorten (attempt {attempt + 1}/2)...")
            response = client.chat.completions.create(
                model       = model,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                    {"role": "assistant", "content": result},
                    {"role": "user",   "content":
                        f"Your translation is {len(result)} characters but the "
                        f"limit is {char_budget}. Rephrase concisely, preserving "
                        f"all medical terminology. Output ONLY the shortened "
                        f"translation, no explanation."},
                ],
                temperature = 0.2,
                max_tokens  = 512,
            )
            result = response.choices[0].message.content.strip()

        # If still over after retries, return what we have — chunking will
        # be the fallback in generate_step_audio.py
        if len(result) > char_budget:
            print(f"    ⚠  Final translation is {len(result)} chars — "
                  f"audio chunking may be triggered downstream.")
        return result

    return translate


# ─────────────────────────────────────────────────────────────────────────────
# NLLB-200 local backend (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def load_nllb_translator(language: str, model_size: str = "1.3B"):
    """Fallback translator — Meta's NLLB-200 model running locally."""
    if language not in NLLB_LANG_CODES:
        print(f"✖  Language '{language}' not supported by NLLB.")
        print(f"   Supported: {', '.join(sorted(NLLB_LANG_CODES))}")
        sys.exit(1)

    if model_size not in NLLB_MODELS:
        print(f"✖  Unknown NLLB model size '{model_size}'. Choose: {', '.join(NLLB_MODELS)}")
        sys.exit(1)

    model_id    = NLLB_MODELS[model_size]
    target_code = NLLB_LANG_CODES[language]
    source_code = "eng_Latn"

    print(f"\n🤖 Backend:    NLLB-200 ({model_size}) — {model_id}")
    print(f"   Target:     {language} → {target_code}")
    size_str = "2.4" if model_size == "600M" else ("5.2" if model_size == "1.3B" else "13")
    print(f"   First run downloads ~{size_str} GB. Subsequent runs are cached.")

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError:
        print("\n✖  transformers/torch not installed. Run:")
        print("   .venv/bin/pip install transformers sentencepiece torch")
        sys.exit(1)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=source_code)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print(f"   ✔  Model loaded ({time.time() - t0:.1f}s)")

    target_token_id = tokenizer.convert_tokens_to_ids(target_code)
    if target_token_id == tokenizer.unk_token_id:
        print(f"✖  Tokenizer doesn't recognize target code '{target_code}'.")
        sys.exit(1)

    def translate(text: str) -> str:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id = target_token_id,
                max_length          = 512,
                num_beams           = 5,
                early_stopping      = True,
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    return translate


# ─────────────────────────────────────────────────────────────────────────────
# Main translation loop (backend-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def translate_sentences(translator, sentences: list) -> list:
    """Translate a list of English sentences. Empty inputs pass through."""
    results = []
    for i, sent in enumerate(sentences):
        if not sent.strip():
            results.append("")
            continue

        preview_in = sent[:60] + ("…" if len(sent) > 60 else "")
        print(f"   [{i+1}/{len(sentences)}] EN: {preview_in}")
        try:
            translated  = translator(sent)
            preview_out = translated[:60] + ("…" if len(translated) > 60 else "")
            print(f"           →  {preview_out}")
            results.append(translated)
        except Exception as exc:
            print(f"   ⚠  Translation failed for sentence {i+1}: {exc}")
            results.append(sent)  # fallback to English so pipeline doesn't break

    return results


def translate_final_mapped(input_path: str, output_path: str, language: str,
                           backend: str, openai_model: str,
                           nllb_size: str) -> None:
    """Read final_mapped.json, translate, and write output preserving timing."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"✖  Expected a JSON array in {input_path}, got {type(data).__name__}")
        sys.exit(1)

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

    if backend == "openai":
        translator = load_openai_translator(language, model=openai_model)
    else:
        translator = load_nllb_translator(language, model_size=nllb_size)

    print(f"\n🌐 Translating {len(data)} sentence(s) → {language}:")
    english_sentences = [item.get(text_key, "") for item in data]
    translated        = translate_sentences(translator, english_sentences)

    output_data = []
    for item, new_text in zip(data, translated):
        new_item                      = dict(item)
        new_item[text_key]             = new_text
        new_item["language"]           = language
        new_item["original_english"]   = item.get(text_key, "")
        new_item["translation_backend"] = backend
        output_data.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Output:     {output_path}")
    print(f"   ✔  Wrote {len(output_data)} translated item(s).")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

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
        help="Target language code. GPT-4 supports any code; NLLB supports: "
             + ", ".join(sorted(NLLB_LANG_CODES)))
    parser.add_argument("--backend", "-b",
        choices=("auto", "openai", "nllb"),
        default="auto",
        help="auto = use openai if OPENAI_API_KEY set, else nllb (default).")
    parser.add_argument("--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini, cheap+good). "
             "Use gpt-4o for highest quality.")
    parser.add_argument("--nllb-size", "-s",
        default="1.3B",
        choices=list(NLLB_MODELS.keys()),
        help="NLLB model size (only used when backend=nllb).")
    parser.add_argument("--force", "-f", action="store_true",
        help="Re-translate even if the output file already exists.")
    return parser.parse_args()


def resolve_backend(requested: str) -> str:
    """auto → pick openai if API key present, else nllb."""
    if requested != "auto":
        return requested
    return "openai" if os.environ.get("OPENAI_API_KEY") else "nllb"


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  MediverseVR — Narration Translator")
    print("=" * 60)

    backend = resolve_backend(args.backend)

    if args.output:
        output_path = args.output
    else:
        input_dir   = os.path.dirname(os.path.abspath(args.input))
        output_path = os.path.join(input_dir, f"final_mapped_{args.language}.json")

    if not os.path.exists(args.input):
        print(f"\n✖  Input file not found: {args.input}")
        print(f"   Run generate_narration.py first to create final_mapped.json.")
        sys.exit(1)

    if os.path.exists(output_path) and not args.force:
        print(f"\n⏭  Output already exists: {output_path}")
        print(f"   Use --force to regenerate.")
        sys.exit(0)

    translate_final_mapped(
        args.input, output_path, args.language,
        backend       = backend,
        openai_model  = args.openai_model,
        nllb_size     = args.nllb_size,
    )

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
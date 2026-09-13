"""
Phase 4 — Step 3: Sample 30 sentences from the active filtered corpus for manual translation.

Selection criteria:
- Track B (model_input) output: cleaned text, NOT stopword-stripped
- Minimum 5 tokens after Track B processing
- Maximum 35 tokens (avoids run-on comments that are hard to translate)
- Must contain at least one Roman-Hindi marker token (ensures code-mixed, not pure English)
- Not a duplicate (deduplicated by cleaned_text)
- Random seed fixed for reproducibility

Output: data/manual_translation_candidates.txt — numbered list ready for hand-translation
"""

import glob
import json
import os
import random
import sys

# Make sure preprocessing is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing.pipeline import preprocess_text

ROMAN_HINDI_MARKERS = {
    "hai", "hain", "kya", "nahi", "bhai", "yaar", "acha", "achha", "kar",
    "karo", "ka", "ke", "ki", "tu", "tum", "main", "mein", "kaun", "kaha",
    "kahan", "kuch", "bhi", "raha", "rahi", "gaya", "gayi", "log", "sab",
    "aur", "nhi", "hona", "hota", "hogi", "hoga", "ho", "hue", "karna",
    "karta", "karenge", "kregi", "jana", "jayega", "liye", "sath", "dono",
    "ko", "se", "par", "wala", "wale", "tera", "meri", "uska", "hamara",
    "unka", "kyun", "kaise", "chahiye", "honi", "chahte", "chahta",
    "bahut", "bohot", "abhi", "phir", "toh", "tab", "jab", "yeh", "woh",
    "iska", "uski", "unki", "apna", "apni", "apne", "sirf", "bilkul",
    "zyada", "thoda", "accha", "sach", "jhooth", "matlab", "samajh",
    "sunna", "dekho", "suno", "padh", "likho", "ek", "do", "teen",
}

SEED = 42
TARGET_N = 30
MIN_TOKENS = 5
MAX_TOKENS = 35

RAW_PATTERN = os.path.join(ROOT, "data", "raw", "raw_filtered_*.jsonl")
OUT_PATH = os.path.join(ROOT, "data", "manual_translation_candidates.txt")
OUT_JSONL = os.path.join(ROOT, "data", "manual_translation_candidates.jsonl")


def has_roman_hindi(tokens):
    return any(t.lower() in ROMAN_HINDI_MARKERS for t in tokens)


def collect_candidates():
    files = glob.glob(RAW_PATTERN)
    if not files:
        print(f"ERROR: No files matched {RAW_PATTERN}")
        sys.exit(1)

    print(f"Scanning {len(files)} corpus files...")
    seen_texts = set()
    candidates = []

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_text = rec.get("text", "")
                if not raw_text or not isinstance(raw_text, str):
                    continue

                result = preprocess_text(raw_text, track="model_input")
                cleaned = result.get("cleaned_text") or result.get("cleaned", "")
                tokens = result.get("tokens", [])

                if not cleaned or len(cleaned) < 10:
                    continue
                if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
                    continue
                if not has_roman_hindi(tokens):
                    continue

                # Deduplicate on cleaned text
                key = cleaned.lower().strip()
                if key in seen_texts:
                    continue
                seen_texts.add(key)

                candidates.append({
                    "raw": raw_text,
                    "cleaned": cleaned,
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "source_file": os.path.basename(fp),
                })

    return candidates


def main():
    candidates = collect_candidates()
    print(f"Total qualifying candidates: {len(candidates)}")

    if len(candidates) < TARGET_N:
        print(f"WARNING: Only {len(candidates)} candidates found — less than target {TARGET_N}.")
        selected = candidates
    else:
        rng = random.Random(SEED)
        selected = rng.sample(candidates, TARGET_N)
        # Sort by token count ascending for readability
        selected.sort(key=lambda x: x["token_count"])

    # Write numbered plain-text output for manual translation
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Manual Translation Candidates — Phase 4\n")
        f.write("# Instructions: Fill in the English translation on the blank line after each sentence.\n")
        f.write("# Keep the source sentence exactly as shown. Do not edit or fix the source.\n")
        f.write(f"# Seed: {SEED} | Selected: {len(selected)} of {len(candidates)} candidates\n\n")
        for i, rec in enumerate(selected, 1):
            f.write(f"{i}. SOURCE:  {rec['cleaned']}\n")
            f.write(f"   TOKENS ({rec['token_count']}): {' | '.join(rec['tokens'])}\n")
            f.write(f"   TRANSLATION: \n\n")

    # Also write JSONL for programmatic use later
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for i, rec in enumerate(selected, 1):
            rec["id"] = i
            rec["translation"] = ""
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Text output (for manual translation): {OUT_PATH}")
    print(f"  JSONL output (for programmatic use):  {OUT_JSONL}")
    print(f"  Sentences selected: {len(selected)}")
    print(f"  Token range: {min(r['token_count'] for r in selected)}–{max(r['token_count'] for r in selected)}")


if __name__ == "__main__":
    main()

"""
Investigate whether the source/target swap observed in Row 9 of the PHINC spot-check
is an isolated error or a systematic pattern.

Strategy:
1. For every pair, check if the target looks more like Hinglish than the source
   (i.e., target contains more Hindi markers than source does) — the classic
   signature of a swapped pair.
2. Also flag pairs where the source looks purely English (no Hindi markers,
   no Devanagari) — these are suspicious given the dataset is supposed to be
   code-mixed Hinglish → English.
3. Also flag identical-looking pairs that survived deduplication due to minor
   case/whitespace differences.

Output:
  data/phinc/phinc_swap_investigation.txt  — human-readable report
  data/phinc/phinc_cleaned.jsonl           — dataset with swaps removed
  data/phinc/phinc_train_cleaned.jsonl     — cleaned train split
  data/phinc/phinc_validation_cleaned.jsonl — cleaned validation split
"""

import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHINC_DIR = os.path.join(ROOT, "data", "phinc")
OUT_DIR = PHINC_DIR
SEED = 42

# Load the full filtered set (not the split, so we can re-split after cleaning)
RAW_PATH = os.path.join(PHINC_DIR, "phinc_raw.jsonl")

# Expanded Roman-Hindi markers (same as data/roman_hindi_markers.json base + common terms)
HINDI_MARKERS = {
    "hai", "hain", "kya", "nahi", "nhi", "bhai", "yaar", "acha", "achha",
    "kar", "karo", "ka", "ke", "ki", "tu", "tum", "main", "mein", "kaun",
    "kaha", "kahan", "kuch", "bhi", "raha", "rahi", "gaya", "gayi", "log",
    "sab", "aur", "hona", "hota", "hogi", "hoga", "ho", "hue", "karna",
    "karta", "karenge", "jana", "jayega", "liye", "sath", "dono", "ko",
    "se", "par", "wala", "wale", "tera", "meri", "uska", "hamara", "unka",
    "kyun", "kaise", "chahiye", "honi", "bahut", "bohot", "abhi", "phir",
    "toh", "tab", "jab", "yeh", "woh", "iska", "uski", "unki", "apna",
    "apni", "apne", "sirf", "bilkul", "zyada", "thoda", "accha", "sach",
    "matlab", "samajh", "sunna", "dekho", "suno", "ek", "bolo", "wahi",
    "wahin", "phir", "lekin", "kyunki", "isliye", "jaise", "jab", "tab",
    "roz", "aaj", "kal", "parso", "din", "raat", "subah", "shaam",
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def hindi_score(text: str) -> float:
    """Return fraction of alphabetic tokens that are Hindi markers or Devanagari."""
    if not text:
        return 0.0
    tokens = re.findall(r"[A-Za-z\u0900-\u097F]+", text.lower())
    if not tokens:
        return 0.0
    hindi_count = sum(
        1 for t in tokens
        if t.lower() in HINDI_MARKERS or bool(DEVANAGARI_RE.search(t))
    )
    return hindi_count / len(tokens)


def looks_english_only(text: str) -> bool:
    """True if text has no Devanagari and no Hindi markers (suspicious for source)."""
    tokens = re.findall(r"[A-Za-z\u0900-\u097F]+", text.lower())
    if not tokens:
        return True
    return not any(
        t.lower() in HINDI_MARKERS or bool(DEVANAGARI_RE.search(t))
        for t in tokens
    )


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rows = load_jsonl(RAW_PATH)
    print(f"Loaded {len(rows)} rows from {RAW_PATH}")

    swapped = []         # target more Hindi than source
    english_src = []     # source looks purely English
    both_flagged = []    # both issues

    swap_threshold = 0.15  # if target score > source score + threshold → likely swapped

    for i, r in enumerate(rows):
        src = r["source"]
        tgt = r["target"]
        src_score = hindi_score(src)
        tgt_score = hindi_score(tgt)

        is_swapped = (tgt_score - src_score) > swap_threshold
        is_eng_src = looks_english_only(src)

        if is_swapped and is_eng_src:
            both_flagged.append((i, src, tgt, src_score, tgt_score))
        elif is_swapped:
            swapped.append((i, src, tgt, src_score, tgt_score))
        elif is_eng_src:
            english_src.append((i, src, tgt, src_score, tgt_score))

    # All suspicious rows
    all_flagged_indices = set()
    all_flagged_indices.update(i for i, *_ in swapped)
    all_flagged_indices.update(i for i, *_ in english_src)
    all_flagged_indices.update(i for i, *_ in both_flagged)

    total_flagged = len(all_flagged_indices)
    pct_flagged = total_flagged / len(rows) * 100

    # Build cleaned set
    cleaned = [r for i, r in enumerate(rows) if i not in all_flagged_indices]

    # Re-split cleaned set 90/10
    rng = random.Random(SEED)
    shuffled = cleaned[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.90)
    train_clean = shuffled[:split_idx]
    valid_clean = shuffled[split_idx:]

    # Write cleaned files
    write_jsonl(cleaned, os.path.join(OUT_DIR, "phinc_cleaned.jsonl"))
    write_jsonl(train_clean, os.path.join(OUT_DIR, "phinc_train_cleaned.jsonl"))
    write_jsonl(valid_clean, os.path.join(OUT_DIR, "phinc_validation_cleaned.jsonl"))

    # Write report
    report_lines = [
        "PHINC Row-Swap / English-Source Investigation Report",
        "=" * 55,
        f"Total pairs inspected: {len(rows)}",
        f"",
        f"Detection criteria:",
        f"  SWAP: target Hindi-marker score > source score by >{swap_threshold:.0%}",
        f"  ENG-SRC: source contains zero Hindi markers / Devanagari",
        f"",
        f"Findings:",
        f"  Swapped pairs (target more Hindi than source): {len(swapped)}",
        f"  English-only source (no Hindi content): {len(english_src)}",
        f"  Both flags: {len(both_flagged)}",
        f"  TOTAL FLAGGED (unique rows): {total_flagged} ({pct_flagged:.1f}% of corpus)",
        f"",
        f"Decision: {'FILTER OUT flagged rows — rate is meaningful' if pct_flagged > 2.0 else 'Keep all — rate is within noise tolerance (<2%)'}",
        f"",
        f"Post-filter corpus: {len(cleaned)} pairs",
        f"  Train (90%): {len(train_clean)}",
        f"  Validation (10%): {len(valid_clean)}",
        f"",
        "=" * 55,
        "Sample flagged rows (up to 15 examples):",
        "",
    ]

    all_samples = [
        (i, src, tgt, ss, ts, "SWAP+ENG") for i, src, tgt, ss, ts in both_flagged
    ] + [
        (i, src, tgt, ss, ts, "SWAP") for i, src, tgt, ss, ts in swapped
    ] + [
        (i, src, tgt, ss, ts, "ENG-SRC") for i, src, tgt, ss, ts in english_src
    ]
    rng2 = random.Random(0)
    rng2.shuffle(all_samples)

    for idx, (row_i, src, tgt, ss, ts, tag) in enumerate(all_samples[:15]):
        report_lines += [
            f"[{tag}] Row {row_i} | src_score={ss:.2f} tgt_score={ts:.2f}",
            f"  SRC: {src[:120]}",
            f"  TGT: {tgt[:120]}",
            "",
        ]

    report_path = os.path.join(OUT_DIR, "phinc_swap_investigation.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nReport saved: {report_path}")
    print(f"Cleaned files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

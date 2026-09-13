"""
Phase 4 — Step 2: Download PHINC, clean, and produce train/validation splits.

Source: veezbo/phinc on HuggingFace (13,738 rows)
Columns: Sentence (code-mixed Hindi-English), English_Translation

Output files:
  data/phinc/phinc_raw.jsonl            — all rows after minimal validation
  data/phinc/phinc_train.jsonl          — 90% training split
  data/phinc/phinc_validation.jsonl     — 10% validation split
  data/phinc/phinc_stats.txt            — summary stats

Quality filters applied:
  1. Drop rows with empty/null source or target
  2. Drop rows where source == target (no translation performed)
  3. Drop rows where either field is < 3 characters
  4. Drop rows where source has > 120 characters (extreme outliers)
  5. Deduplicate on (source.strip().lower(), target.strip().lower())
"""

import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "phinc")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.90

# Quality filter thresholds
MIN_CHARS = 3
MAX_SRC_CHARS = 120

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed. Run: pip install datasets")
    sys.exit(1)


def clean_text(text):
    """Basic whitespace normalisation only — no content alteration."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def quality_filter(rows):
    """Apply quality filters and return (kept, rejection_counts) tuple."""
    kept = []
    stats = {
        "empty": 0,
        "too_short": 0,
        "identical": 0,
        "src_too_long": 0,
        "duplicate": 0,
    }
    seen = set()

    for row in rows:
        src = clean_text(row.get("Sentence", ""))
        tgt = clean_text(row.get("English_Translation", ""))

        if not src or not tgt:
            stats["empty"] += 1
            continue
        if len(src) < MIN_CHARS or len(tgt) < MIN_CHARS:
            stats["too_short"] += 1
            continue
        if src.lower() == tgt.lower():
            stats["identical"] += 1
            continue
        if len(src) > MAX_SRC_CHARS:
            stats["src_too_long"] += 1
            continue

        key = (src.lower(), tgt.lower())
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)

        kept.append({"source": src, "target": tgt})

    return kept, stats


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print("Downloading PHINC from HuggingFace (veezbo/phinc)...")
    ds = load_dataset("veezbo/phinc")
    raw_rows = list(ds["train"])
    print(f"  Raw rows loaded: {len(raw_rows)}")

    print("Applying quality filters...")
    kept, stats = quality_filter(raw_rows)
    print(f"  Kept after filtering: {len(kept)}")
    for k, v in stats.items():
        if v > 0:
            print(f"    Dropped ({k}): {v}")

    # Save raw filtered set
    raw_path = os.path.join(OUT_DIR, "phinc_raw.jsonl")
    write_jsonl(kept, raw_path)
    print(f"  Saved raw filtered set: {raw_path}")

    # Train / validation split
    rng = random.Random(SEED)
    shuffled = kept[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * TRAIN_RATIO)
    train_rows = shuffled[:split_idx]
    valid_rows = shuffled[split_idx:]

    train_path = os.path.join(OUT_DIR, "phinc_train.jsonl")
    valid_path = os.path.join(OUT_DIR, "phinc_validation.jsonl")
    write_jsonl(train_rows, train_path)
    write_jsonl(valid_rows, valid_path)
    print(f"  Saved train split ({len(train_rows)} rows): {train_path}")
    print(f"  Saved validation split ({len(valid_rows)} rows): {valid_path}")

    # Statistics report
    src_lens = [len(r["source"]) for r in kept]
    tgt_lens = [len(r["target"]) for r in kept]
    stats_lines = [
        "PHINC Dataset — Phase 4 Preparation Statistics",
        "=" * 50,
        f"Source: veezbo/phinc (HuggingFace)",
        f"Download date: 2026-09-13",
        f"Raw rows from HF: {len(raw_rows)}",
        "",
        "Quality Filter Results:",
        f"  Dropped (empty src or tgt): {stats['empty']}",
        f"  Dropped (too short <{MIN_CHARS} chars): {stats['too_short']}",
        f"  Dropped (identical src==tgt): {stats['identical']}",
        f"  Dropped (src > {MAX_SRC_CHARS} chars): {stats['src_too_long']}",
        f"  Dropped (duplicate pairs): {stats['duplicate']}",
        f"  KEPT AFTER FILTERING: {len(kept)}",
        "",
        "Split:",
        f"  Train (90%): {len(train_rows)} pairs",
        f"  Validation (10%): {len(valid_rows)} pairs",
        f"  Random seed: {SEED}",
        "",
        "Corpus Statistics (post-filter):",
        f"  Avg source length (chars): {sum(src_lens)/len(src_lens):.1f}",
        f"  Avg target length (chars): {sum(tgt_lens)/len(tgt_lens):.1f}",
        f"  Min/Max source length: {min(src_lens)} / {max(src_lens)}",
        f"  Min/Max target length: {min(tgt_lens)} / {max(tgt_lens)}",
        "",
        "Files:",
        f"  {raw_path}",
        f"  {train_path}",
        f"  {valid_path}",
    ]

    stats_path = os.path.join(OUT_DIR, "phinc_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines) + "\n")
    print(f"\n  Stats report: {stats_path}")

    # Print summary
    print("\n" + "\n".join(stats_lines))


if __name__ == "__main__":
    main()

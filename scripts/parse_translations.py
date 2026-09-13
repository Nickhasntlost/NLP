"""
Parse completed manual_translation_candidates.txt into a clean evaluation JSONL.

Reads the hand-filled translation file, extracts all SOURCE / TRANSLATION pairs,
skips entries with blank translations (i.e. skipped sentences), and writes:
  data/youtube_eval.jsonl — domain-matched YouTube evaluation set

Format of each JSONL line:
  {"id": <int>, "source": "<code-mixed>", "target": "<English translation>"}

Run after filling in translations in data/manual_translation_candidates.txt.
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT, "data", "manual_translation_candidates.txt")
OUTPUT_PATH = os.path.join(ROOT, "data", "youtube_eval.jsonl")


def parse_translation_file(path):
    """
    Parse the numbered SOURCE / TRANSLATION block format.
    Returns list of dicts: {id, source, translation}
    Skips entries where TRANSLATION is blank.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Pattern: number. SOURCE: <text> (on one line)
    # Then: TRANSLATION: <text> (on next or same area)
    blocks = re.split(r"\n(?=\d+\.\s+SOURCE:)", content)

    results = []
    skipped_blank = []
    skipped_pii = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract ID
        id_match = re.match(r"^(\d+)\.\s+SOURCE:", block)
        if not id_match:
            continue
        entry_id = int(id_match.group(1))

        # Extract SOURCE
        src_match = re.search(r"SOURCE:\s*(.+?)(?:\n|$)", block)
        if not src_match:
            continue
        source = src_match.group(1).strip()

        # Check for PII placeholder — skip these
        if "PII" in source and re.search(r"[0-9a-f]{8,}", source):
            skipped_pii.append(entry_id)
            continue

        # Extract TRANSLATION — everything after "TRANSLATION:" on that line
        tgt_match = re.search(r"TRANSLATION:\s*(.*?)(?:\n\n|\Z)", block, re.DOTALL)
        if not tgt_match:
            skipped_blank.append(entry_id)
            continue

        translation = tgt_match.group(1).strip()
        # Remove any trailing TOKENS line that got swept in
        translation = re.sub(r"\nTOKENS\s*\(.*", "", translation).strip()

        if not translation:
            skipped_blank.append(entry_id)
            continue

        results.append({
            "id": entry_id,
            "source": source,
            "target": translation,
        })

    return results, skipped_blank, skipped_pii


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        sys.exit(1)

    print(f"Parsing: {INPUT_PATH}")
    pairs, skipped_blank, skipped_pii = parse_translation_file(INPUT_PATH)

    print(f"\nResults:")
    print(f"  Usable pairs: {len(pairs)}")
    if skipped_blank:
        print(f"  Skipped (blank translation): IDs {skipped_blank}")
    if skipped_pii:
        print(f"  Skipped (PII placeholder): IDs {skipped_pii}")

    # Write output JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(pairs)} pairs to: {OUTPUT_PATH}")

    # Print all pairs for verification
    print("\nFull contents of evaluation set:")
    print("-" * 60)
    for p in pairs:
        print(f"[{p['id']:2d}] SRC: {p['source']}")
        print(f"      TGT: {p['target']}")
        print()


if __name__ == "__main__":
    main()

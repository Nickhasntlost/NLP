"""Create an independent validation set from the Phase 1 holdout.

Produces:
- models/lid_independent_candidate.jsonl  (200 sampled sentences, tokens only)
- models/lid_independent_valid_auto.jsonl (auto-labeled with a different heuristic)

Modes:
- Auto-label: uses a small built-in English word list (independent from roman_hindi_markers.json)
- Interactive: run with `--interactive` to label tokens manually in the terminal

Usage:
  python models/create_independent_validation.py --count 200
  python models/create_independent_validation.py --interactive
"""
import argparse
import glob
import json
import os
import random
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(__file__))
HOLDOUT_PATH = os.path.join(ROOT, "data", "raw", "holdout_unbiased_sample.jsonl")
OUT_DIR = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# Small English word list (common words) — intentionally different from roman_hindi_markers
EN_WORDS = {
    "the", "and", "is", "in", "on", "for", "to", "of", "with", "as",
    "this", "that", "are", "was", "were", "it", "from", "by", "an",
    "be", "have", "has", "do", "does", "did", "not", "what", "where",
    "who", "why", "how", "when", "which", "will", "would", "can", "could",
}


def read_holdout(path: str) -> List[str]:
    texts = []
    if not os.path.exists(path):
        return texts
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for key in ("text", "comment", "sentence", "title"):
                if key in obj and obj[key]:
                    texts.append(str(obj[key]))
                    break
    return texts


def tokenize_texts(texts: List[str], preprocess_text):
    samples = []
    sid = 0
    for t in texts:
        sid += 1
        out = preprocess_text(t, track="model_input")
        tokens = out.get("tokens", [])
        if tokens:
            samples.append({"id": f"iv{sid}", "raw": t, "tokens": tokens})
    return samples


def auto_label(tokens: List[str]):
    labels = []
    for tok in tokens:
        if any('\u0900' <= ch <= '\u097F' for ch in tok):
            labels.append("HI")
        else:
            nl = tok.lower()
            if nl in EN_WORDS:
                labels.append("EN")
            elif nl.isdigit():
                labels.append("OTHER")
            else:
                # fallback: if token ends with typical Hindi suffix translit
                if nl.endswith(("hai","ka","ke","ki","mein","nahi","kar")):
                    labels.append("HI")
                else:
                    labels.append("EN")
    return labels


def interactive_label(samples):
    print("Interactive labeling - enter labels as space-separated tokens (HI/EN/OTHER). Type 'skip' to skip sentence.")
    out = []
    for s in samples:
        print('\n---')
        print(s['raw'])
        print('Tokens:', ' | '.join(s['tokens']))
        resp = input('Labels (or skip): ').strip()
        if not resp or resp.lower() == 'skip':
            print('Skipped')
            continue
        labs = resp.split()
        if len(labs) != len(s['tokens']):
            print('Length mismatch; skipping')
            continue
        out.append({"id": s['id'], "raw": s['raw'], "tokens": s['tokens'], "labels": labs})
    return out


def main(count: int = 200, interactive: bool = False, seed: int = 42):
    random.seed(seed)
    # ensure importable pipeline
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from preprocessing.pipeline import preprocess_text

    texts = read_holdout(HOLDOUT_PATH)
    if not texts:
        print('No holdout found at', HOLDOUT_PATH)
        return

    chosen = random.sample(texts, min(count, len(texts)))
    samples = tokenize_texts(chosen, preprocess_text)

    candidate_path = os.path.join(OUT_DIR, 'lid_independent_candidate.jsonl')
    with open(candidate_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print('Wrote candidate file:', candidate_path)

    if interactive:
        labeled = interactive_label(samples)
        out_path = os.path.join(OUT_DIR, 'lid_independent_valid.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for s in labeled:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print('Wrote manual validation file:', out_path)
    else:
        auto = []
        for s in samples:
            labs = auto_label(s['tokens'])
            auto.append({"id": s['id'], "raw": s['raw'], "tokens": s['tokens'], "labels": labs})
        out_auto = os.path.join(OUT_DIR, 'lid_independent_valid_auto.jsonl')
        with open(out_auto, 'w', encoding='utf-8') as f:
            for s in auto:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print('Wrote auto-labeled independent validation file:', out_auto)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=200)
    p.add_argument('--interactive', action='store_true')
    args = p.parse_args()
    main(count=args.count, interactive=args.interactive)

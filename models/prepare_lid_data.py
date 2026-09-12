"""Prepare labeled LID dataset from Track B (model_input) output.

Produces:
- models/lid_full.jsonl  (full weak-labeled dataset)
- models/lid_sample_100.jsonl (random 100-sample for small-sample testing)
- models/lid_train.jsonl / models/lid_holdout.jsonl (80/20 split)
- models/lid_dataset.zip (train+holdout packaged)

Labeling heuristic:
- If token's script == 'devanagari' => 'HI'
- Else if script == 'latin' and token.lower() in roman_hindi_markers => 'HI'
- Else if script == 'latin' => 'EN'
- Else => 'OTHER'

This is a deterministic weak-labeling step; review before full training.
"""
import glob
import json
import os
import random
from typing import List

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_PATTERN = os.path.join(ROOT, "data", "raw", "raw_filtered_*.jsonl")
OUT_DIR = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

MARKERS_PATH = os.path.join(ROOT, "data", "roman_hindi_markers.json")


def load_markers(path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
            return [x.lower() for x in payload.get("roman_hindi_markers", [])]
    except Exception:
        return []


def label_tokens(tokens: List[str], tags: List[str], markers: List[str]):
    labels = []
    for t, tag in zip(tokens, tags):
        nl = t.lower()
        if tag == "devanagari":
            labels.append("HI")
        elif tag == "latin":
            if nl in markers:
                labels.append("HI")
            else:
                labels.append("EN")
        elif tag == "mixed":
            labels.append("OTHER")
        else:
            labels.append("OTHER")
    return labels


def extract_texts_from_file(fp: str):
    with open(fp, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # common keys
            for key in ("text", "comment", "sentence", "title"):
                if key in obj and obj[key]:
                    yield str(obj[key])
                    break
            else:
                if isinstance(obj, str) and obj.strip():
                    yield obj


def main():
    # ensure project root is on sys.path so local packages can be imported
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from preprocessing.pipeline import preprocess_text

    markers = load_markers(MARKERS_PATH)
    files = sorted(glob.glob(RAW_PATTERN))
    entries = []
    sid = 0
    for fp in files:
        for text in extract_texts_from_file(fp):
            sid += 1
            out = preprocess_text(text, track="model_input")
            tokens = out.get("tokens", [])
            tagged = out.get("tagged", [])
            tags = [t.get("script", "garbage") for t in tagged]
            if not tokens:
                continue
            labels = label_tokens(tokens, tags, markers)
            entries.append({"id": f"s{sid}", "raw": text, "tokens": tokens, "labels": labels})

    # write full
    full_path = os.path.join(OUT_DIR, "lid_full.jsonl")
    with open(full_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # sample 100
    sample = random.sample(entries, min(100, len(entries)))
    sample_path = os.path.join(OUT_DIR, "lid_sample_100.jsonl")
    with open(sample_path, "w", encoding="utf-8") as f:
        for e in sample:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # train/holdout split (80/20)
    random.shuffle(entries)
    split = int(len(entries) * 0.8)
    train = entries[:split]
    hold = entries[split:]
    train_path = os.path.join(OUT_DIR, "lid_train.jsonl")
    hold_path = os.path.join(OUT_DIR, "lid_holdout.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for e in train:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(hold_path, "w", encoding="utf-8") as f:
        for e in hold:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # package
    import zipfile

    zip_path = os.path.join(OUT_DIR, "lid_dataset.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(full_path, arcname="lid_full.jsonl")
        z.write(sample_path, arcname="lid_sample_100.jsonl")
        z.write(train_path, arcname="lid_train.jsonl")
        z.write(hold_path, arcname="lid_holdout.jsonl")

    print("Wrote:", full_path)
    print("Sample:", sample_path)
    print("Train:", train_path)
    print("Holdout:", hold_path)
    print("Zip:", zip_path)


if __name__ == "__main__":
    main()

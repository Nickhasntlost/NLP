"""Run a lightweight inference-only sanity check on unseen sentences.

This script loads a trained token-classification model from a given directory,
selects up to 15 sentences from the untouched Phase 1 holdout sample, and prints
original text, tokenized tokens, and predicted HI/EN/OTHER labels for inspection.

Usage:
  python models/run_lid_sanity_check.py --model_dir /content/lid_model
  python models/run_lid_sanity_check.py --model_dir /path/to/model --holdout_path data/raw/holdout_unbiased_sample.jsonl
"""

import argparse
import json
import os
import re
from typing import List


HINDI_MARKERS = {
    "hai", "haii", "hain", "ho", "hoga", "hoge", "kar", "ki", "ke", "ka",
    "mein", "me", "nahi", "aur", "se", "ko", "bhi", "par", "ye", "vo", "aap",
    "sir", "mam", "bhai", "ji", "aaj", "kal", "aur", "toh", "kya"
}

LABEL_LIST = ["HI", "EN", "OTHER"]


def load_texts(path: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("sentence") or obj.get("comment")
            if text:
                texts.append(str(text))
    return texts


def contains_devanagari(s: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in s)


def score_text(s: str) -> float:
    score = 0.0
    if contains_devanagari(s):
        score += 2.0
    words = re.findall(r"[A-Za-z]+|[\u0900-\u097F]+|\d+", s.lower())
    if any(w in HINDI_MARKERS for w in words):
        score += 2.0
    if len(words) <= 18:
        score += 0.5
    return score


def choose_examples(texts: List[str], limit: int = 15) -> List[str]:
    ranked = sorted(texts, key=score_text, reverse=True)
    chosen = []
    seen = set()
    for t in ranked:
        if len(chosen) >= limit:
            break
        clean = " ".join(t.split())
        if clean in seen:
            continue
        seen.add(clean)
        chosen.append(clean)
    return chosen


def tokenize_for_inference(text: str, tokenizer):
    # Match the model-input tokenization used in training: split into words.
    tokens = text.split()
    tokenized = tokenizer(tokens, is_split_into_words=True, truncation=True, return_tensors="pt")
    return tokens, tokenized


def build_label_lookup(model):
    raw_lookup = getattr(model.config, "id2label", {}) or {}
    lookup = {}
    for key, value in raw_lookup.items():
        try:
            lookup[int(key)] = value
        except (TypeError, ValueError):
            continue
    if not lookup or all(str(v).startswith("LABEL_") for v in lookup.values()):
        lookup = {idx: label for idx, label in enumerate(LABEL_LIST)}
    return lookup


def predict_labels(model, tokenizer, text: str):
    tokens, tokenized = tokenize_for_inference(text, tokenizer)
    outputs = model(**tokenized)
    logits = outputs.logits[0]
    pred_ids = logits.argmax(dim=-1)
    word_ids = tokenized.word_ids(batch_index=0)
    label_lookup = build_label_lookup(model)
    labels = []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous_word_id:
            continue
        labels.append(label_lookup.get(int(pred_ids[idx]), f"LABEL_{int(pred_ids[idx])}"))
        previous_word_id = word_id
    return tokens, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference-only sanity check for the LID model")
    parser.add_argument("--model_dir", default="/content/lid_model", help="Directory with the trained model and tokenizer")
    parser.add_argument("--holdout_path", default="data/raw/holdout_unbiased_sample.jsonl", help="Original untouched holdout sample")
    parser.add_argument("--limit", type=int, default=15, help="Number of examples to show")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not os.path.exists(args.holdout_path):
        raise FileNotFoundError(f"Holdout file not found: {args.holdout_path}")

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    texts = load_texts(args.holdout_path)
    examples = choose_examples(texts, limit=args.limit)

    print(f"Running sanity check on {len(examples)} examples from original untouched holdout")
    print("" )

    for i, text in enumerate(examples, start=1):
        tokens, labels = predict_labels(model, tokenizer, text)
        print(f"Example {i}")
        print("Original:", text)
        print("Tokens:", tokens)
        print("Predicted:", labels)
        print("-")

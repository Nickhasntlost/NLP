"""
grammar.py — Model 4: Grammar & Fluency Correction Post-Processor
==================================================================
Takes raw English translations produced by Model 3 (IndicTrans2) and
polishes capitalization, punctuation, agreement, and phrasing.

Per RULES.md R3.1 and ARCHITECTURE.md §2.6:
    - Status: Pretrained / off-the-shelf model (no fine-tuning applied).
    - Base Model: `vennify/t5-base-grammar-correction` with rule-assisted
      fallback and semantic drift guard.
    - Input: raw English translation string from Model 3.
    - Output: fluent, grammatically corrected English sentence.
"""

import re
import sys
import warnings
from typing import Optional, Set

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

MODEL_NAME = "vennify/t5-base-grammar-correction"

_model = None
_tokenizer = None
_device = None

# Common English stopwords used for meaning drift calculation
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to",
    "of", "in", "on", "at", "by", "with", "as", "is", "was", "are", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "you", "i", "me", "my", "your", "our", "us",
    "do", "does", "did", "have", "has", "had", "so", "too", "very", "just",
}


def _load_grammar_model():
    """Lazy load the T5 grammar correction model."""
    global _model, _tokenizer, _device
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [Model 4] Loading {MODEL_NAME} on {_device.upper()}...", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    _model.to(_device)
    _model.eval()
    print("  [Model 4] Grammar correction model ready.", flush=True)


def _rule_based_polish(text: str) -> str:
    """
    Fast, deterministic rule-based grammar and orthography polish:
    1. Strips leading/trailing punctuation artifacts (e.g. '.. hello' -> 'hello').
    2. Capitalizes sentence starts and standalone pronoun 'I' / 'I'm' / 'I've'.
    3. Normalizes spacing around punctuation marks.
    4. Ensures terminal sentence punctuation (. or ?).
    """
    if not text or not text.strip():
        return ""

    s = text.strip()

    # Strip leading punctuation/dot artifacts common in translation decoder outputs
    s = re.sub(r"^[\.\,\:\;\-\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    # Normalize spaces before and after punctuation
    s = re.sub(r"\s+([,.?!;:])", r"\1", s)
    s = re.sub(r"([(\[\{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]\}])", r"\1", s)

    # Capitalize standalone pronoun "i" and common contractions ("i'm", "i've", "i'll", "i'd")
    s = re.sub(r"\bi\b", "I", s)
    s = re.sub(r"\bi'([a-zA-Z]+)\b", r"I'\1", s)

    # Capitalize the first letter of each sentence
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)

    # Ensure valid terminal punctuation if sentence has words
    if re.search(r"[a-zA-Z0-9]$", s):
        # Infer if sentence starts with interrogative word
        first_word = s.split()[0].lower() if s.split() else ""
        if first_word in {"who", "what", "where", "when", "why", "how", "is", "are", "can", "could", "would", "do", "does", "did"}:
            s += "?"
        else:
            s += "."

    return s


def _check_meaning_drift(raw: str, candidate: str) -> bool:
    """
    Check whether a neural grammar correction candidate distorts meaning.
    Returns True if safe (preserved meaning), False if drift is detected.
    """
    raw_words = re.findall(r"\b[A-Za-z0-9]+\b", raw.lower())
    cand_words = re.findall(r"\b[A-Za-z0-9]+\b", candidate.lower())

    if not raw_words or not cand_words:
        return True

    # Check length ratio: Candidate should not shrink < 50% or explode > 200%
    ratio = len(cand_words) / len(raw_words)
    if ratio < 0.5 or ratio > 2.0:
        return False

    # Extract non-stopword content tokens
    raw_content = {w for w in raw_words if w not in _STOPWORDS and len(w) >= 3}
    cand_content = {w for w in cand_words if w not in _STOPWORDS and len(w) >= 3}

    if not raw_content:
        return True

    # At least 60% of original content words must be retained
    overlap = len(raw_content & cand_content) / len(raw_content)
    return overlap >= 0.60


def correct_grammar(text: str, use_neural: bool = True) -> str:
    """
    Model 4 entry point:
    Polishes raw translation output from Model 3.

    Args:
        text: Raw English translation from Model 3.
        use_neural: If True, attempts neural T5 grammar correction with
                    rule-based fallback and meaning-drift guard.
                    If False, applies deterministic rule-based polish only.

    Returns:
        Fluent, polished English sentence with preserved meaning.
    """
    if not text or not text.strip():
        return ""

    # Always start with basic cleanup
    cleaned = _rule_based_polish(text)

    if not use_neural:
        return cleaned

    try:
        global _model, _tokenizer, _device
        if _model is None:
            _load_grammar_model()

        import torch
        prompt = f"grammar: {cleaned}"
        inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(_device)

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=3,
                early_stopping=True,
            )

        candidate = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Check for semantic drift before accepting neural output
        if _check_meaning_drift(cleaned, candidate):
            return _rule_based_polish(candidate)
        else:
            # Fallback to rule-cleaned version if neural rewrite drifted
            return cleaned

    except Exception:
        # Fallback to deterministic rule-based polish if neural model fails or is offline
        return cleaned


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    test_sentences = [
        "what are you doing today",
        "Today was a very tiring day.",
        "sir you are very good i understand",
        ".. who is watching today",
        "i was looking at Rich and i was dumbstruck",
        "he go to market yesterday",
    ]

    print("=== MODEL 4 (GRAMMAR & FLUENCY CORRECTION) DEMO ===")
    for s in test_sentences:
        rule_out = correct_grammar(s, use_neural=False)
        print(f"RAW  : {s}")
        print(f"POLISH: {rule_out}")
        print("-" * 50)

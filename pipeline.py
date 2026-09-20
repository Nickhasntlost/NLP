"""
pipeline.py — Unified End-to-End Hinglish Translation Pipeline (Phase 7)
========================================================================
Integrates all project stages conforming strictly to ARCHITECTURE.md §3:

  Stage 1: Preprocessing (Phase 2)
           Noise removal, mixed-script tokenization, script tagging.
  Stage 2: Model 1 — Language Identification (Phase 3)
           Token-level language classification: 'HI', 'EN', 'OTHER'.
  Stage 3: Model 2 — Normalization & Script Conversion (Phase 5)
           Elongation collapse, slang normalization, tag-restricted edit-distance,
           and Roman-to-Devanagari transliteration (ITRANS).
  Stage 4: Model 3 — Context-Aware Translation (Phase 4)
           IndicTrans2 1B baseline inference (hin_Deva -> eng_Latn).
  Stage 5: Model 4 — Grammar & Fluency Correction (Phase 6)
           Decoder artifact cleanup, capitalization, punctuation, and semantic drift guard.

Exit Criteria (EVALUATION.md Phase 7):
  - 10 end-to-end runs complete without manual patching.
  - Output schema at each interface matches ARCHITECTURE.md §3.
"""

from dataclasses import asdict, dataclass, field
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Module Imports from Local Packages ───────────────────────────────────────
from preprocessing.pipeline import preprocess_text
from models.normalize import (
    CANONICAL_SLANG_MAP,
    _load_canonical_vocab,
    is_roman_script,
    normalize_tokens,
    roman_to_deva,
)
from models.translate import translate_devanagari
from models.grammar import correct_grammar


@dataclass
class PipelineResult:
    """
    Structured container holding intermediate representations and telemetry
    across all stages of the translation pipeline per ARCHITECTURE.md §3.
    """
    input_text: str
    preprocessed_tokens: List[str]
    script_tags: List[Dict[str, str]]
    lid_tags: List[Tuple[str, str]]
    normalized_hinglish: str
    devanagari_input: str
    raw_translation: str
    final_translation: str
    timing_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return asdict(self)


# ── Stage 2: Token-Level Language Identification (Model 1) ───────────────────

_HI_MARKERS_VOCAB: Optional[set] = None

_EN_VOCAB: Set[str] = {
    "video", "videos", "picture", "pictures", "pic", "pics", "vid", "vids",
    "please", "thanks", "thank", "welcome", "sorry", "plz", "pls", "thx", "ty",
    "subscribe", "channel", "like", "share", "comment", "post", "content",
    "good", "bad", "best", "great", "nice", "awesome", "amazing", "love",
    "today", "yesterday", "tomorrow", "day", "night", "morning", "time",
    "music", "song", "movie", "game", "film", "story", "rich", "rice",
    "tiktok", "youtube", "twitter", "instagram", "facebook", "app", "phone",
    "help", "me", "you", "my", "your", "he", "she", "it", "we", "they",
    "is", "was", "are", "were", "have", "has", "had", "do", "does", "did",
}


def _get_hi_markers_vocab() -> set:
    global _HI_MARKERS_VOCAB
    if _HI_MARKERS_VOCAB is None:
        _HI_MARKERS_VOCAB = set(_load_canonical_vocab())
        # Add colloquial Hindi verbal roots and spelling variants
        _HI_MARKERS_VOCAB.update({
            "dekh", "dekho", "dekha", "dekhna", "dekhe", "dkh", "dkho", "dkha", "dkhna",
            "bol", "bolo", "bola", "bolna", "bole",
            "sun", "suno", "suna", "sunna", "sune",
            "samajh", "samjho", "smjh", "smjho",
            "bata", "batao", "batana", "btao",
            "bana", "banao", "banaya", "bna", "bnao",
            "khao", "peeyo", "padh", "padho",
            "chalo", "chal", "aao", "jaao", "ruk", "ruko",
            "nhi", "nhii", "nhn", "nhin", "bht", "bhut", "bhot",
            "rha", "rhi", "rhe", "rh", "rhna", "rhne",
            "gya", "gyi", "gye", "kr", "kro", "kra", "kri",
            "krrha", "krrhi", "krrhe", "krega", "kregi", "krenge", "krna",
            "aj", "phle", "shyd", "vakt", "waqt"
        })
    return _HI_MARKERS_VOCAB


def identify_languages(
    tokens: List[str],
    script_tags: Optional[List[Dict[str, str]]] = None,
) -> List[Tuple[str, str]]:
    """
    Model 1: Token-Level Language Identification.
    Labels each token as 'HI' (Hindi), 'EN' (English), or 'OTHER' (punctuation/digits/symbols).

    Conforms to Phase 3 specifications and ARCHITECTURE.md §3:
      - Devanagari tokens -> HI
      - Latin tokens in canonical Hindi vocabulary / markers -> HI
      - Latin tokens in common English vocabulary -> EN
      - Latin tokens otherwise default to EN
      - Numbers, punctuation, non-alphabetic -> OTHER
    """
    if not tokens:
        return []

    hi_vocab = _get_hi_markers_vocab()
    script_map = {}
    if script_tags:
        for item in script_tags:
            script_map[item.get("token", "")] = item.get("script", "")

    results: List[Tuple[str, str]] = []
    for tok in tokens:
        script = script_map.get(tok, "")
        tok_lower = tok.lower()

        # Pure numbers or non-alphabetic symbols
        if tok.isdigit() or not re.search(r"[a-zA-Z\u0900-\u097F]", tok):
            results.append((tok, "OTHER"))
        # Devanagari script tokens
        elif any("\u0900" <= ch <= "\u097F" for ch in tok) or script == "devanagari":
            results.append((tok, "HI"))
        # Known English tokens
        elif tok_lower in _EN_VOCAB:
            results.append((tok, "EN"))
        # Roman Hindi tokens / slang
        elif tok_lower in hi_vocab or tok_lower in {"h", "k", "n", "b", "r"}:
            results.append((tok, "HI"))
        # Default Latin tokens -> English
        else:
            results.append((tok, "EN"))

    return results


# ── Unified Pipeline Runner ──────────────────────────────────────────────────

def translate_pipeline(
    text: str,
    use_neural_grammar: bool = False,
) -> PipelineResult:
    """
    Run full end-to-end Hinglish -> English translation pipeline across Phases 2-6.

    Args:
        text: Raw input string (Roman Hinglish, pure Devanagari, or mixed).
        use_neural_grammar: If True, uses T5 neural grammar correction.
                            If False (default), uses sub-millisecond rule-assisted polish.

    Returns:
        PipelineResult with full telemetry and intermediate representations.
    """
    timing: Dict[str, float] = {}
    t_start_pipeline = time.perf_counter()

    # Guard: Empty or whitespace input
    if not text or not text.strip():
        total_ms = (time.perf_counter() - t_start_pipeline) * 1000
        timing["total_ms"] = round(total_ms, 2)
        return PipelineResult(
            input_text=text,
            preprocessed_tokens=[],
            script_tags=[],
            lid_tags=[],
            normalized_hinglish="",
            devanagari_input="",
            raw_translation="",
            final_translation="",
            timing_ms=timing,
        )

    # ── Stage 1: Preprocessing (Phase 2) ────────────────────────────────────
    t0 = time.perf_counter()
    prep_data = preprocess_text(text, track="model_input")
    tokens: List[str] = prep_data.get("tokens", [])
    script_tags: List[Dict[str, str]] = prep_data.get("tagged", [])
    timing["preprocessing_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ── Stage 2: Model 1 — Language Identification (Phase 3) ────────────────
    t0 = time.perf_counter()
    lid_tags = identify_languages(tokens, script_tags)
    timing["lid_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ── Stage 3: Model 2 — Normalization & Script Conversion (Phase 5) ──────
    t0 = time.perf_counter()
    if lid_tags:
        normalized_hinglish = normalize_tokens(lid_tags)
    else:
        normalized_hinglish = text.strip()

    # Transliterate to Devanagari if input is Roman-script
    if is_roman_script(normalized_hinglish):
        devanagari_input = roman_to_deva(normalized_hinglish)
    else:
        devanagari_input = normalized_hinglish
    timing["normalization_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ── Stage 4: Model 3 — Machine Translation (Phase 4) ─────────────────────
    t0 = time.perf_counter()
    raw_translation = translate_devanagari(devanagari_input)
    timing["translation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ── Stage 5: Model 4 — Grammar & Fluency Correction (Phase 6) ───────────
    t0 = time.perf_counter()
    final_translation = correct_grammar(raw_translation, use_neural=use_neural_grammar)
    timing["grammar_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    total_ms = (time.perf_counter() - t_start_pipeline) * 1000
    timing["total_ms"] = round(total_ms, 2)

    return PipelineResult(
        input_text=text,
        preprocessed_tokens=tokens,
        script_tags=script_tags,
        lid_tags=lid_tags,
        normalized_hinglish=normalized_hinglish,
        devanagari_input=devanagari_input,
        raw_translation=raw_translation,
        final_translation=final_translation,
        timing_ms=timing,
    )


def translate_text(text: str, use_neural_grammar: bool = False) -> str:
    """Convenience helper returning just the final translated string."""
    res = translate_pipeline(text, use_neural_grammar=use_neural_grammar)
    return res.final_translation


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    test_input = "Bhaaaai kyaaa kar rhe ho aajkal? plzz batao"
    print(f"Testing Pipeline on: '{test_input}'")
    result = translate_pipeline(test_input)
    print("\n--- Pipeline Result ---")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

"""
test_normalize.py — Unit Tests for Model 2 Normalization & Script Conversion
=============================================================================
Tests all primitives, elongation collapse, colloquial slang canonicalization,
case preservation, LID label-awareness, Devanagari preservation, and transliteration.
"""

import pytest
from models.normalize import (
    collapse_elongations,
    edit_distance_normalize,
    is_roman_script,
    normalize,
    normalize_and_transliterate,
    normalize_tokens,
    normalize_word,
    roman_to_deva,
)


# ─── 1. Character Elongation Collapse ────────────────────────────────────────

def test_collapse_elongations():
    assert collapse_elongations("bhaaaai") == "bhai"
    assert collapse_elongations("kyaaa") == "kya"
    assert collapse_elongations("soooo") == "so"
    assert collapse_elongations("bahuuuut") == "bahut"
    assert collapse_elongations("haaaan") == "haan"
    assert collapse_elongations("theeeek") == "theek"
    assert collapse_elongations("plzzzz") in ("plz", "please")


# ─── 2. Slang and Contraction Canonicalization ───────────────────────────────

def test_slang_canonicalization():
    assert normalize_word("nhi") == "nahi"
    assert normalize_word("nhii") == "nahi"
    assert normalize_word("nhn") == "nahi"
    assert normalize_word("bht") == "bahut"
    assert normalize_word("bhut") == "bahut"
    assert normalize_word("bhot") == "bahut"
    assert normalize_word("acha") == "achha"
    assert normalize_word("krrha") == "kar raha"
    assert normalize_word("dkh") == "dekh"
    assert normalize_word("plz") == "please"
    assert normalize_word("pls") == "please"
    assert normalize_word("thx") == "thanks"
    assert normalize_word("ty") == "thanks"
    assert normalize_word("btw") == "by the way"


# ─── 3. Capitalization / Case Preservation ───────────────────────────────────

def test_capitalization_preservation():
    assert normalize_word("Bhaaaai") == "Bhai"
    assert normalize_word("Nhi") == "Nahi"
    assert normalize_word("Bht") == "Bahut"
    assert normalize_word("Plz") == "Please"


# ─── 4. Full Sentence Normalization ──────────────────────────────────────────

def test_sentence_normalization():
    # Elongation + slang
    s1 = "bhaaaai kyaaa kar rhe ho?"
    assert normalize(s1) == "bhai kya kar rahe ho?"

    # Multiple colloquialisms
    s2 = "aj ka din bht jyada thk gya hu"
    assert normalize(s2) == "aaj ka din bahut zyada theek gaya hu"

    # Social media shorthand
    s3 = "plzz help me bro, thx"
    assert normalize(s3) == "please help me bhai, thanks"

    # Single-letter copula 'h' -> 'hai'
    s4 = "koi nhi bolega scripted h"
    assert normalize(s4) == "koi nahi bolega scripted hai"


# ─── 5. Model 1 LID Tagged Normalization ─────────────────────────────────────

def test_normalize_tokens_with_lid_labels():
    tokens_with_labels = [
        ("bht", "HI"),
        ("acha", "HI"),
        ("video", "EN"),
        ("hai", "HI"),
    ]
    result = normalize_tokens(tokens_with_labels)
    assert "bahut" in result
    assert "achha" in result
    assert "video" in result
    assert "hai" in result


# ─── 6. Devanagari Preservation ──────────────────────────────────────────────

def test_devanagari_untouched():
    deva = "भाई क्या कर रहा है आजकल?"
    assert normalize(deva) == deva
    assert normalize_and_transliterate(deva) == deva


# ─── 7. Transliteration to Devanagari ────────────────────────────────────────

def test_transliteration():
    assert is_roman_script("bhai kya kar rahe ho") is True
    assert is_roman_script("भाई क्या कर रहे हो") is False

    translit = normalize_and_transliterate("bhai kya kar raha hai")
    # Verify resulting text contains Devanagari characters
    assert any("\u0900" <= c <= "\u097F" for c in translit)


# ─── 8. Meaning Preservation (No Corruption) ─────────────────────────────────

def test_meaning_preservation_no_corruption():
    # Common English words should not be corrupted into accidental Hindi words
    assert normalize_word("rice", lang_tag="EN") == "rice"
    assert normalize_word("class", lang_tag="EN") == "class"
    assert normalize_word("teacher", lang_tag="EN") == "teacher"
    assert normalize_word("computer", lang_tag="EN") == "computer"

    # Numbers and symbols should remain unchanged
    assert normalize("100% genuine video") == "100% genuine video"

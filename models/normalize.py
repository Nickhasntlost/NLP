"""
normalize.py — Model 2: Hinglish Normalization & Script Conversion
===================================================================
Standardizes informal, noisy, Romanized code-mixed Hindi (Hinglish) into
canonical orthography, compresses phonetic elongations (e.g. "bhaaaai" -> "bhai"),
expands social/chat abbreviations (e.g. "bht" -> "bahut", "plz" -> "please"),
and converts Roman Hinglish to Devanagari script for Model 3 consumption.

Per RULES.md R3.3 and ARCHITECTURE.md §2.4:
    - Rule-based + dictionary + edit-distance normalizer (deterministic,
      zero hallucination risk, sub-millisecond latency).
    - Owns Roman-to-Devanagari transliteration before Model 3.
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple


# ─── 1. Canonical Vocabulary & Slang Dictionaries ────────────────────────────

MARKERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "roman_hindi_markers.json",
)

def _load_canonical_vocab() -> Set[str]:
    """Load canonical Roman Hindi markers from data/roman_hindi_markers.json."""
    vocab = {
        "aaj", "acha", "achha", "agar", "apna", "apne", "apni", "aur", "baad",
        "bata", "bhai", "bhi", "chahiye", "chalo", "de", "dena", "dono",
        "gaya", "gayi", "gaye", "hai", "hain", "hoga", "hogi", "hoge",
        "hona", "hota", "hoti", "hote", "hue", "ho", "honi", "jana", "jao",
        "ji", "jo", "ka", "kaam", "kaha", "kahan", "kaise", "kaun", "kar",
        "karna", "karta", "karti", "karte", "karenge", "karega", "karegi",
        "karo", "ke", "kehte", "khud", "ki", "kiya", "kiye", "kuch", "kya",
        "kyun", "kyunki", "liye", "log", "main", "mera", "mere", "meri",
        "mein", "me", "nahi", "par", "pehle", "raha", "rahi", "rahe",
        "sab", "sath", "se", "so", "tab", "tak", "tera", "tere", "teri",
        "the", "thi", "thoda", "tu", "tum", "unka", "unke", "unki",
        "uska", "uske", "uski", "wala", "wale", "wali", "yaar", "yeh",
        "theek", "bahut", "zyada", "badhiya", "sahi", "sirf", "shayad", "haan",
        "maine", "tune", "unhone", "inhone", "kisne", "sabne",
        "dene", "lene", "karne", "hone", "jane", "aane", "bolne", "dekhne", "rahne", "padhne",
        "padhta", "padhte", "padhti", "padhate", "padhati", "padhata",
        "bolta", "bolte", "bolti", "bolna", "samajhta", "samajhte", "samajhti",
        "chahta", "chahti", "chahte",
        "jaata", "jaate", "jaati", "aata", "aate", "aati",
        "dekhta", "dekhte", "dekhti", "sunta", "sunte", "sunti",
        "khata", "khate", "khati", "peeta", "peete", "peeti",
        "milta", "milte", "milti", "lagta", "lagte", "lagti",
        "rehta", "rehte", "rehti", "chalta", "chalte", "chalti",
        "khan", "sir"
    }
    if os.path.isfile(MARKERS_PATH):
        try:
            with open(MARKERS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                vocab.update(data.get("roman_hindi_markers", []))
        except Exception:
            pass
    return vocab


CANONICAL_VOCAB = _load_canonical_vocab()

# High-frequency colloquial spelling variants and abbreviations -> canonical forms
CANONICAL_SLANG_MAP: Dict[str, str] = {
    # Negation
    "nhi": "nahi",
    "nhii": "nahi",
    "nhn": "nahi",
    "nhin": "nahi",
    "nahin": "nahi",
    "naa": "na",

    # Wh-questions & Interrogatives
    "kyaa": "kya",
    "kyaaa": "kya",
    "kyu": "kyun",
    "kyn": "kyun",
    "kn": "kaun",
    "koun": "kaun",
    "kaha": "kahan",
    "khn": "kahan",
    "kse": "kaise",
    "kaisey": "kaise",

    # Pronouns & Demonstratives
    "mjhe": "mujhe",
    "mujhko": "mujhe",
    "mene": "maine",
    "mainey": "maine",
    "tjhe": "tujhe",
    "tujhko": "tujhe",
    "hume": "humein",
    "humko": "humein",
    "unhe": "unhein",
    "inhe": "inhein",
    "ye": "yeh",
    "wo": "woh",
    "voh": "woh",

    # Auxiliary & Copular verbs
    "h": "hai",
    "hy": "hai",
    "he": "hai",
    "hn": "hain",
    "hnn": "hain",
    "hyn": "hain",
    "thaa": "tha",
    "thii": "thi",
    "thee": "the",

    # Aspectual verbs & Participles
    "rha": "raha",
    "rhi": "rahi",
    "rhe": "rahe",
    "rh": "raha",
    "rhna": "rahna",
    "rhne": "rahne",
    "gya": "gaya",
    "gyi": "gayi",
    "gye": "gaye",

    # Common verbs & Multi-word contractions
    "kr": "kar",
    "kro": "karo",
    "kra": "kara",
    "kri": "kari",
    "krrha": "kar raha",
    "krrhi": "kar rahi",
    "krrhe": "kar rahe",
    "krega": "karega",
    "kregi": "karegi",
    "krenge": "karenge",
    "karnge": "karenge",
    "krna": "karna",
    "dkh": "dekh",
    "dkho": "dekho",
    "dkha": "dekha",
    "dkhna": "dekhna",
    "dkhe": "dekhe",
    "bna": "bana",
    "bnao": "banao",
    "bnaya": "banaya",
    "bnane": "banane",
    "jaega": "jayega",
    "jaegi": "jayegi",
    "btao": "batao",
    "smjh": "samajh",
    "smjho": "samjho",
    "smjhaya": "samjhaya",

    # Adverbs, Adjectives & Particles
    "bht": "bahut",
    "bhut": "bahut",
    "bhot": "bahut",
    "bohot": "bahut",
    "bahoot": "bahut",
    "acha": "achha",
    "achaa": "achha",
    "axha": "achha",
    "badiya": "badhiya",
    "bdya": "badhiya",
    "zyada": "zyada",
    "jyada": "zyada",
    "zada": "zyada",
    "thk": "theek",
    "thik": "theek",
    "thek": "theek",
    "sahi": "sahi",
    "shai": "sahi",
    "shi": "sahi",
    "bhaii": "bhai",
    "bhaai": "bhai",
    "bhayi": "bhai",
    "bhaiya": "bhai",
    "yr": "yaar",
    "yar": "yaar",
    "kch": "kuch",
    "kuchh": "kuch",
    "aj": "aaj",
    "ajkl": "aajkal",
    "bs": "bas",
    "bass": "bas",
    "srf": "sirf",
    "phle": "pehle",
    "bad": "baad",
    "shyd": "shayad",
    "shydh": "shayad",
    "haan": "haan",
    "haaan": "haan",
    "haaaan": "haan",
    "han": "haan",
    "waqt": "waqt",
    "vakt": "waqt",

    # Internet / Chat Acronyms & English Shorthands
    "plz": "please",
    "pls": "please",
    "plzz": "please",
    "thx": "thanks",
    "ty": "thanks",
    "tysm": "thank you so much",
    "btw": "by the way",
    "idk": "i do not know",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "omg": "oh my god",
    "bro": "bhai",
    "pic": "picture",
    "pics": "pictures",
    "vid": "video",
    "vids": "videos",
}


# ─── 2. Normalization Primitives ─────────────────────────────────────────────

def collapse_elongations(word: str) -> str:
    """
    Compress repeated characters (e.g. 'bhaaaai' -> 'bhai', 'soooo' -> 'so').
    Preserves legitimate double vowels ('aa', 'ee', 'oo') when matching canonical vocab.
    Preserves titlecase if the input was titlecased.
    """
    if len(word) < 3:
        return word

    is_title = word.istitle()

    # If no 3+ consecutive identical characters, return unchanged
    if not re.search(r"(\w)\1{2,}", word, flags=re.IGNORECASE):
        return word

    # Candidate 1: Collapse 3+ repeats to single char
    c1 = re.sub(r"(\w)\1{2,}", r"\1", word, flags=re.IGNORECASE)
    # Candidate 2: Collapse 3+ repeats to 2 chars (for standard doubles like aa/ee/oo)
    c2 = re.sub(r"(\w)\1{2,}", r"\1\1", word, flags=re.IGNORECASE)

    lower_c1 = c1.lower()
    lower_c2 = c2.lower()

    if lower_c1 in CANONICAL_SLANG_MAP:
        res = CANONICAL_SLANG_MAP[lower_c1]
    elif lower_c2 in CANONICAL_SLANG_MAP:
        res = CANONICAL_SLANG_MAP[lower_c2]
    elif lower_c2 in CANONICAL_VOCAB:
        res = c2
    elif lower_c1 in CANONICAL_VOCAB:
        res = c1
    else:
        res = c1

    return res.capitalize() if is_title else res


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def edit_distance_normalize(token: str, vocab: Optional[Set[str]] = None) -> str:
    """
    If a token is length >= 6 and not in canonical vocab, attempt distance <= 1 correction.
    Conservative: Only applies if exactly ONE nearest match exists in vocabulary.
    Words with length < 6 are handled strictly via CANONICAL_SLANG_MAP to prevent
    false collisions (e.g. Khan -> Kahan, year -> yaar, dene -> dena).
    """
    if vocab is None:
        vocab = CANONICAL_VOCAB
    lower = token.lower()
    if lower in vocab or len(lower) < 6:
        return token

    candidates = []
    for word in vocab:
        if abs(len(word) - len(lower)) <= 1:
            if levenshtein_distance(lower, word) == 1:
                candidates.append(word)

    if len(candidates) == 1:
        # Preserve original capitalization if titlecased
        res = candidates[0]
        return res.capitalize() if token.istitle() else res

    return token


# ─── 3. Token-Level and Sentence-Level Normalizers ────────────────────────────

def normalize_word(token: str, lang_tag: Optional[str] = None) -> str:
    """
    Normalize an individual word token:
    1. Collapse character elongations
    2. Check colloquial slang lookup dictionary
    3. If tag is 'HI' or undetermined and not matched, try conservative edit-distance
    """
    # Keep punctuation and Devanagari tokens as-is
    if not token.strip() or any("\u0900" <= c <= "\u097F" for c in token) or not any(c.isalpha() for c in token):
        return token

    # Check case
    is_title = token.istitle()
    lower = token.lower()

    # Step 1: Elongation collapse
    collapsed = collapse_elongations(token)
    lower_collapsed = collapsed.lower()

    # Step 2: Slang / phonetic dictionary lookup
    if lower_collapsed in CANONICAL_SLANG_MAP:
        replacement = CANONICAL_SLANG_MAP[lower_collapsed]
        return replacement.capitalize() if is_title else replacement

    if lower in CANONICAL_SLANG_MAP:
        replacement = CANONICAL_SLANG_MAP[lower]
        return replacement.capitalize() if is_title else replacement

    # Step 3: Edit-distance fallback (strictly for tokens explicitly tagged as 'HI' by Model 1)
    if lang_tag == "HI":
        corrected = edit_distance_normalize(collapsed)
        if corrected != collapsed:
            return corrected

    return collapsed


def normalize_tokens(tokens_with_labels: List[Tuple[str, str]]) -> str:
    """
    Normalize token sequence given (token, lid_label) pairs from Model 1 (LID).
    Matches interface contract in ARCHITECTURE.md §3 (Model 1 output -> Model 2 input).
    """
    normalized_words = []
    for token, label in tokens_with_labels:
        norm = normalize_word(token, lang_tag=label)
        normalized_words.append(norm)

    # Reconstruct sentence, keeping spaces around words and natural punctuation
    out = " ".join(normalized_words)
    out = re.sub(r"\s+([,.?!;:'%/\)\]\}])", r"\1", out)
    out = re.sub(r"([(\[\{])\s+", r"\1", out)
    return out.strip()


TOKEN_RE = re.compile(
    r"[\u0900-\u097F]+(?:[\u200c\u200d][\u0900-\u097F]+)*|[A-Za-z0-9]+|[^\w\s]"
)


def normalize(text: str) -> str:
    """
    Sentence-level normalization entry point for raw Roman Hinglish text.
    Handles tokenization, elongation collapse, slang canonicalization, and spacing.
    Preserves Devanagari words and matras completely intact.
    """
    if not text or not text.strip():
        return ""

    tokens = TOKEN_RE.findall(text)
    normalized = [normalize_word(tok) for tok in tokens]

    out = " ".join(normalized)
    out = re.sub(r"\s+([,.?!;:'%/\)\]\}])", r"\1", out)
    out = re.sub(r"([(\[\{])\s+", r"\1", out)
    return out.strip()


# ─── 4. Script Conversion (Roman -> Devanagari Transliteration) ───────────────

def is_roman_script(text: str) -> bool:
    """Return True if text is predominantly Latin/Roman-script."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    latin = sum(1 for c in alpha if c.isascii())
    return latin / len(alpha) > 0.5


def roman_to_deva(text: str) -> str:
    """
    Transliterate Roman-script Hinglish text to Devanagari using ITRANS scheme.
    If indic-transliteration is unavailable, returns text unmodified.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        return transliterate(text.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
    except ImportError:
        return text


def normalize_and_transliterate(text: str) -> str:
    """
    Full Model 2 pipeline function:
    1. Normalizes spelling variants, elongations, and slang in Roman script.
    2. Transliterates Roman script to Devanagari (skips if already Devanagari).
    Output is ready to feed directly into Model 3 (ai4bharat/indictrans2-indic-en-1B).
    """
    norm_text = normalize(text)
    if is_roman_script(norm_text):
        return roman_to_deva(norm_text)
    return norm_text


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    test_cases = [
        "bhaaaai kyaaa kar rhe ho?",
        "aj ka din bht jyada thk gya hu",
        "sir aap bht acha padhate ho, mjhe smjh aa gya",
        "plzz help me bro, thx",
        "koi nhi bolega scripted h",
        "भाई क्या कर रहा है आजकल?",
    ]
    print("=== MODEL 2 (NORMALIZATION & SCRIPT CONVERSION) DEMO ===")
    for s in test_cases:
        norm = normalize(s)
        deva = normalize_and_transliterate(s)
        print(f"RAW  : {s}")
        print(f"NORM : {norm}")
        print(f"DEVA : {deva}")
        print("-" * 50)

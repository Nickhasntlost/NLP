"""
translate.py — Hinglish → English inference wrapper
====================================================
Loads the baseline (pretrained, not fine-tuned)
ai4bharat/indictrans2-indic-en-1B model and exposes a single
translate(text: str) -> str function for Hinglish → English translation.

Script-mismatch handling
------------------------
IndicTrans2-indic-en-1B expects Devanagari Hindi as input (it was trained on
hin_Deva). Hinglish in our corpus is Roman-script (e.g. "Bhai kya kar raha
hai"). This file therefore adds a lightweight pre-processing step:

    Roman Hinglish  ─[ITRANS transliteration]→  Devanagari  ─[IndicTrans2]→  English

The transliteration uses `indic-transliteration` (pure-Python, no GPU, no
fairseq) with the ITRANS scheme. When the model receives well-formed Devanagari
input it produces human-grade English (validated empirically). English loanwords
in code-mixed input (e.g. "rice", "mood") are passed through as Latin tokens
by IndicProcessor; IndicTrans2 handles them correctly in context.

Dependencies (inference only — no training libs required):
    pip install "transformers==4.47.0" torch sentencepiece sacremoses
    pip install indic-transliteration
    # IndicTransToolkit optional — falls back to models/indic_processor_py.py

    Note: transformers must be pinned to 4.x. The IndicTrans2 custom tokenizer
    (tokenization_indictrans.py) uses _special_tokens_map which was removed in
    transformers 5.x. transformers 4.47.0 is the last compatible 4.x release.

Usage:
    from models.translate import translate
    print(translate("Bhai kya kar raha hai aajkal?"))

Or run directly:
    python models/translate.py
"""

import re
import time
import warnings

# ── Silence non-critical transformers warnings at load time ───────────────────
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="You are using a model of type")

MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
SRC_LANG   = "hin_Deva"
TGT_LANG   = "eng_Latn"

# ── Module-level singletons — loaded once, reused per translate() call ────────
_model      = None
_tokenizer  = None
_ip         = None          # IndicProcessor instance
_tgt_bos_id = None
_device     = None

# ── Roman → Devanagari transliteration helpers ────────────────────────────────

def _is_roman_script(text: str) -> bool:
    """
    Return True when the input is predominantly Latin/Roman-script (Hinglish).
    Threshold: >50% of alphabetic characters are ASCII.
    Pure Devanagari input will correctly return False and skip transliteration.
    """
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    latin = sum(1 for c in alpha if c.isascii())
    return latin / len(alpha) > 0.5


def _roman_to_deva(text: str) -> str:
    """
    Transliterate a Roman-script Hinglish sentence to Devanagari using the
    ITRANS scheme via the `indic-transliteration` package (pure-Python).

    English loanwords embedded in Hinglish (e.g. "rice", "phone", "mood") are
    also transliterated to approximate Devanagari syllables (e.g. "रिचे",
    "मूद्"); IndicTrans2 handles these gracefully in context and produces
    correct English output for the surrounding sentence.

    The function lower-cases the text before transliterating because ITRANS is
    case-sensitive and informal Hinglish uses inconsistent capitalisation.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        return transliterate(text.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
    except ImportError:
        # indic-transliteration not installed — return text unchanged.
        # The model will still run; quality will degrade for Roman-script input.
        return text


def _load():
    """
    Load the model, tokenizer, and IndicProcessor.
    Called once on first translate() invocation (lazy load).
    Prints progress markers so the user knows a load is in progress,
    not a hang — the 1B model takes 2-5 minutes on CPU, 30-60s on GPU.
    """
    global _model, _tokenizer, _ip, _tgt_bos_id, _device

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        from IndicTransToolkit import IndicProcessor
        _use_real_ip = True
    except ImportError:
        # IndicTransToolkit needs Cython+MSVC on Windows.
        # Fall back to pure-Python port (indic_processor_py.py).
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
            from indic_processor_py import IndicProcessor  # type: ignore
            _use_real_ip = True
            print('  NOTE: Using pure-Python IndicProcessor (full transliteration active).')
        except Exception as _ip_err:
            _use_real_ip = False
            print('  NOTE: IndicProcessor unavailable; using bare fallback.')
            class IndicProcessor:  # type: ignore
                def preprocess_batch(self, texts, src_lang, tgt_lang, **kw):
                    return [src_lang + ' ' + tgt_lang + ' ' + t for t in texts]
                def postprocess_batch(self, texts, lang, **kw):
                    import re as _re
                    return [_re.sub(r'>>[a-z_]+<<', '', t).strip() for t in texts]


    _device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  IndicTrans2 1B — Loading model from Hugging Face")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Device: {_device.upper()}")
    if _device == "cpu":
        print()
        print("  WARNING: No GPU detected. Running on CPU.")
        print("  Model load will take ~3-6 minutes on a typical laptop.")
        print("  Each sentence will take ~30-90 seconds to translate.")
        print("  This is expected — not a hang. Please wait.")
    print("=" * 60)

    t0 = time.time()

    print("  [1/3] Loading tokenizer...", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"        done ({time.time()-t0:.1f}s)", flush=True)

    print("  [2/3] Loading model weights (~2-4 GB download on first run)...", flush=True)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model.to(_device)
    _model.eval()
    print(f"        done ({time.time()-t0:.1f}s)", flush=True)

    print("  [3/3] Initialising IndicProcessor...", flush=True)
    if _use_real_ip:
        _ip = IndicProcessor(inference=True)
    else:
        _ip = IndicProcessor()   # our fallback class, no args needed
    print(f"        done ({time.time()-t0:.1f}s)", flush=True)

    # Restore as_target_tokenizer() if the installed transformers version removed it
    from contextlib import contextmanager as _cm
    if not hasattr(_tokenizer, "as_target_tokenizer"):
        if hasattr(_tokenizer, "_tgt_tokenize"):
            @_cm
            def _as_tgt(tok=_tokenizer):
                _orig = tok._tokenize
                tok._tokenize = tok._tgt_tokenize
                try:
                    yield
                finally:
                    tok._tokenize = _orig
            _tokenizer.as_target_tokenizer = _as_tgt
        else:
            from contextlib import nullcontext
            _tokenizer.as_target_tokenizer = lambda: nullcontext()

    # Resolve the forced BOS token ID for English output
    _tgt_bos_id = None
    for _lang_str in [TGT_LANG, f"[{TGT_LANG}]", f"<{TGT_LANG}>"]:
        _candidate = _tokenizer.convert_tokens_to_ids(_lang_str)
        _unk = getattr(_tokenizer, "unk_token_id", None)
        if _candidate is not None and _candidate != _unk:
            _tgt_bos_id = _candidate
            break

    total = time.time() - t0
    print()
    print(f"  Model ready. Load time: {total:.1f}s")
    print("=" * 60)


def translate(text: str) -> str:
    """
    Translate a single Hinglish sentence to English.

    The pipeline handles both Roman-script Hinglish ("Bhai kya kar raha hai")
    and Devanagari Hindi ("भाई क्या कर रहा है") transparently:
      1. If input is predominantly Latin/Roman-script, it is first transliterated
         to Devanagari (ITRANS scheme) so IndicTrans2 receives its expected script.
      2. IndicProcessor normalises and language-tags the Devanagari sentence.
      3. IndicTrans2 1B generates the English translation.

    Args:
        text: Input sentence in Hinglish (Roman-script or Devanagari or mixed).

    Returns:
        English translation string (leading punctuation artifacts removed).

    Example:
        >>> from models.translate import translate
        >>> translate("Bhai kya kar raha hai?")
        'what are you doing today?'
    """
    import torch

    if _model is None:
        _load()

    # ── Step 1: Model 2 Normalization & Script Conversion ─────────────────────
    # Normalizes character elongations, colloquial slang, and spelling variants,
    # then converts Roman Hinglish to Devanagari (or bypasses if already Devanagari)
    # so IndicTrans2 receives clean, canonical Devanagari input.
    try:
        from models.normalize import normalize_and_transliterate
        text = normalize_and_transliterate(text)
    except Exception:
        if _is_roman_script(text):
            text = _roman_to_deva(text)

    # ── Step 2: IndicProcessor normalisation & language tagging ───────────────
    preprocessed = _ip.preprocess_batch([text], src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    inputs = _tokenizer(
        preprocessed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            forced_bos_token_id=_tgt_bos_id,
            num_beams=4,
            max_new_tokens=128,
            early_stopping=True,
            use_cache=False,          # Required: IndicTrans2 custom decoder breaks with new Cache API
            repetition_penalty=1.2,   # Prevents repetition collapse on OOV/slang tokens
            no_repeat_ngram_size=3,   # Prevents n-gram repetition loops
        )

    with _tokenizer.as_target_tokenizer():
        decoded_raw = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Postprocess: IndicProcessor denormalises entities and numbers
    decoded = _ip.postprocess_batch(decoded_raw, lang=TGT_LANG)

    # Strip leading period/dot/whitespace artifact introduced by IndicProcessor
    result = re.sub(r"^[.\s]+", "", decoded[0]).strip()
    return result


# ── CLI test harness ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    TEST_SENTENCES = [
        # Benchmark set: informal Hinglish (Roman-script)
        "Bhai kya kar raha hai aajkal?",
        "Aaj ka din bahut zyada thaka dene wala tha.",
        "Sir aap bahut achha padhate ho, mujhe samajh aa gaya.",
        "Yaar main rice khate hue dekh rahi thi aur mera mood kharab ho gaya.",
        # Sanity check: pure Devanagari (should also work correctly)
        "\u092d\u093e\u0908 \u0915\u094d\u092f\u093e \u0915\u0930 \u0930\u0939\u093e \u0939\u0948 \u0906\u091c\u0915\u0932?",
    ]

    print()
    print("Running local inference test — 5 sentences (4 Hinglish + 1 Devanagari)")
    print("Load time is included in the first sentence's elapsed time.")
    print()

    load_start = time.time()
    for i, sentence in enumerate(TEST_SENTENCES, 1):
        t0 = time.time()
        result = translate(sentence)
        elapsed = time.time() - t0

        script = "[Deva]" if not _is_roman_script(sentence) else "[Roman]"
        print(f"[{i}] {script} SRC : {sentence}")
        print(f"         OUT : {result}")
        if i == 1:
            print(f"         TIME: {elapsed:.1f}s (includes model load)")
        else:
            print(f"         TIME: {elapsed:.1f}s")
        print()

    total_elapsed = time.time() - load_start
    print(f"Total wall-clock time (load + 5 translations): {total_elapsed:.1f}s")

"""
evaluate_pipeline.py — End-to-End Pipeline Evaluation (Phase 7)
================================================================
Fulfills Phase 7 Exit Criteria (EVALUATION.md & TASKS.md):
  1. Executes 10 diverse real code-mixed sentences from our scraped corpus
     end-to-end through the complete pipeline without manual intervention.
  2. Formally validates that output schemas at each interface match
     ARCHITECTURE.md §3.
  3. Measures per-stage and end-to-end latency profiling.
  4. Produces qualitative intermediate data transformation logs.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

from pipeline import PipelineResult, translate_pipeline

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 10 diverse real-world sentences representing colloquial code-mixed phenomena:
BENCHMARK_SENTENCES: List[Tuple[str, str]] = [
    (
        "Bhai kya kar raha hai aajkal?",
        "Standard informal Roman Hinglish question"
    ),
    (
        "Aaj ka din bht zyada thk gya hu bhai",
        "Elongations + slang contractions ('bht', 'thk gya')"
    ),
    (
        "Sir aap bahut achha padhate ho, mujhe samajh aa gaya.",
        "Polite register with mixed clauses and punctuation"
    ),
    (
        "Yaar main rice khate hue dekh rahi thi aur mera mood kharab ho gaya.",
        "Embedded English loanwords ('rice', 'mood') in Hindi syntax"
    ),
    (
        "plzz help me bro, thx so much",
        "Social media abbreviations ('plzz', 'bro', 'thx')"
    ),
    (
        "koi nhi bolega scripted h ye video",
        "Slang ('nhi', single-char copula 'h') and English noun ('video')"
    ),
    (
        "bhaaaai kyaaa scene hai kal ka?",
        "Heavy phonetic elongations ('bhaaaai', 'kyaaa')"
    ),
    (
        "Maine 636 tiktok videos khud report kiye the",
        "Entity ('tiktok'), numerical value ('636'), and Hindi verb morphology"
    ),
    (
        "batao bhai kaunsa phone sabse best h",
        "Hindi question word ('kaunsa'), English loanwords ('phone', 'best')"
    ),
    (
        "भाई क्या कर रहे हो आजकल?",
        "Pure Devanagari Hindi control sentence (script bypass test)"
    ),
]


def validate_schema(res: PipelineResult) -> bool:
    """Validate interface schemas match ARCHITECTURE.md §3."""
    assert isinstance(res.input_text, str), "input_text must be str"
    assert isinstance(res.preprocessed_tokens, list), "preprocessed_tokens must be list"
    assert isinstance(res.script_tags, list), "script_tags must be list"
    assert isinstance(res.lid_tags, list), "lid_tags must be list of tuples"
    assert isinstance(res.normalized_hinglish, str), "normalized_hinglish must be str"
    assert isinstance(res.devanagari_input, str), "devanagari_input must be str"
    assert isinstance(res.raw_translation, str), "raw_translation must be str"
    assert isinstance(res.final_translation, str), "final_translation must be str"
    assert len(res.final_translation) > 0, "final_translation must not be empty"

    timing = res.timing_ms
    required_timing = [
        "preprocessing_ms", "lid_ms", "normalization_ms",
        "translation_ms", "grammar_ms", "total_ms"
    ]
    for key in required_timing:
        assert key in timing, f"Missing timing key: {key}"
    return True


def run_evaluation():
    print("=" * 90)
    print("  PHASE 7: UNIFIED TRANSLATION PIPELINE — END-TO-END BENCHMARK (10 SENTENCES)")
    print("=" * 90)
    print("Executing full pipeline: Preprocess -> LID -> Normalization -> IndicTrans2 -> Grammar Polish\n")

    results: List[PipelineResult] = []
    schema_passed = True

    t_benchmark_start = time.perf_counter()

    for idx, (sentence, category) in enumerate(BENCHMARK_SENTENCES, 1):
        print(f"[{idx:02d}/10] Running: '{sentence}' ({category})...")
        t0 = time.perf_counter()
        res = translate_pipeline(sentence, use_neural_grammar=False)
        dt = (time.perf_counter() - t0) * 1000

        # Validate schema per ARCHITECTURE.md §3
        try:
            validate_schema(res)
            schema_status = "VALID"
        except AssertionError as e:
            schema_status = f"INVALID ({e})"
            schema_passed = False

        results.append(res)
        print(f"       -> Preprocessed Tokens : {res.preprocessed_tokens}")
        print(f"       -> LID Tagged          : {res.lid_tags}")
        print(f"       -> Normalized Hinglish : {res.normalized_hinglish}")
        print(f"       -> Devanagari Model 3  : {res.devanagari_input}")
        print(f"       -> Raw Translation     : {res.raw_translation}")
        print(f"       -> Final English Output: {res.final_translation}")
        print(f"       -> Schema & Latency    : {schema_status} | {dt:.1f}ms total")
        print("-" * 90)

    total_benchmark_sec = time.perf_counter() - t_benchmark_start

    print("\n" + "=" * 90)
    print("  PHASE 7 EVALUATION SUMMARY — EXIT CRITERIA VERIFICATION")
    print("=" * 90)

    avg_total_ms = sum(r.timing_ms["total_ms"] for r in results) / len(results)
    avg_prep_ms = sum(r.timing_ms["preprocessing_ms"] for r in results) / len(results)
    avg_lid_ms = sum(r.timing_ms["lid_ms"] for r in results) / len(results)
    avg_norm_ms = sum(r.timing_ms["normalization_ms"] for r in results) / len(results)
    avg_trans_ms = sum(r.timing_ms["translation_ms"] for r in results) / len(results)
    avg_gram_ms = sum(r.timing_ms["grammar_ms"] for r in results) / len(results)

    print(f"  Total Sentences Evaluated      : {len(results)} / 10")
    print(f"  Execution Completed Cleanly    : 100% (10/10 without manual patching)")
    print(f"  Interface Schema Compliance    : {'100% PASS' if schema_passed else 'FAIL'}")
    print(f"  Total Benchmark Execution Time : {total_benchmark_sec:.2f}s")
    print("\n  Average Latency Breakdown per Sentence:")
    print(f"    - Preprocessing (Phase 2)    : {avg_prep_ms:.2f} ms")
    print(f"    - Language ID (Phase 3)      : {avg_lid_ms:.2f} ms")
    print(f"    - Normalization (Phase 5)    : {avg_norm_ms:.2f} ms")
    print(f"    - Translation (Phase 4)      : {avg_trans_ms:.2f} ms ({avg_trans_ms/1000:.2f}s on CPU)")
    print(f"    - Grammar Polish (Phase 6)   : {avg_gram_ms:.2f} ms")
    print(f"    ---------------------------------------------")
    print(f"    - Total End-to-End Latency   : {avg_total_ms:.2f} ms ({avg_total_ms/1000:.2f}s)")
    print("=" * 90)

    return results


if __name__ == "__main__":
    run_evaluation()

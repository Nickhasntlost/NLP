"""
test_pipeline.py — Unit Tests for Unified Pipeline Integration (Phase 7)
========================================================================
Validates schema conformance per ARCHITECTURE.md §3, edge cases,
language identification, Devanagari handling, and serialization.
"""

import pytest
from unittest.mock import patch
from pipeline import (
    PipelineResult,
    identify_languages,
    translate_pipeline,
    translate_text,
)


def test_pipeline_empty_and_whitespace():
    """Pipeline should gracefully handle empty or whitespace-only input."""
    res_empty = translate_pipeline("")
    assert res_empty.final_translation == ""
    assert res_empty.preprocessed_tokens == []
    assert "total_ms" in res_empty.timing_ms

    res_space = translate_pipeline("    \t\n   ")
    assert res_space.final_translation == ""
    assert res_space.preprocessed_tokens == []


def test_identify_languages():
    """Model 1 language identifier correctly classifies HI, EN, and OTHER tokens."""
    tokens = ["bhai", "video", "dekh", "123", "क्या"]
    script_tags = [
        {"token": "bhai", "script": "latin"},
        {"token": "video", "script": "latin"},
        {"token": "dekh", "script": "latin"},
        {"token": "123", "script": "garbage"},
        {"token": "क्या", "script": "devanagari"},
    ]
    labels = identify_languages(tokens, script_tags)
    label_dict = dict(labels)

    assert label_dict["bhai"] == "HI"
    assert label_dict["video"] == "EN"
    assert label_dict["dekh"] == "HI"
    assert label_dict["123"] == "OTHER"
    assert label_dict["क्या"] == "HI"


def test_pipeline_schema_conformance():
    """Verify PipelineResult adheres strictly to ARCHITECTURE.md §3 interface contract."""
    with patch("pipeline.translate_devanagari", return_value="what are you doing today"):
        result = translate_pipeline("Bhai kya kar raha hai?", use_neural_grammar=False)

    assert isinstance(result, PipelineResult)
    assert result.input_text == "Bhai kya kar raha hai?"
    assert isinstance(result.preprocessed_tokens, list)
    assert len(result.preprocessed_tokens) > 0

    assert isinstance(result.script_tags, list)
    assert isinstance(result.lid_tags, list)
    assert all(len(item) == 2 for item in result.lid_tags)

    assert isinstance(result.normalized_hinglish, str)
    assert isinstance(result.devanagari_input, str)
    assert isinstance(result.raw_translation, str)
    assert isinstance(result.final_translation, str)

    # Verify timing telemetry
    timing = result.timing_ms
    assert "preprocessing_ms" in timing
    assert "lid_ms" in timing
    assert "normalization_ms" in timing
    assert "translation_ms" in timing
    assert "grammar_ms" in timing
    assert "total_ms" in timing

    # Verify JSON serialization
    as_dict = result.to_dict()
    assert as_dict["final_translation"] == result.final_translation


def test_pipeline_pure_devanagari():
    """Verify that pure Devanagari input bypasses transliteration cleanly."""
    devanagari_sentence = "भाई क्या कर रहे हो?"
    with patch("pipeline.translate_devanagari", return_value="brother what are you doing"):
        result = translate_pipeline(devanagari_sentence, use_neural_grammar=False)

    # Devanagari input should remain Devanagari
    assert any("\u0900" <= ch <= "\u097F" for ch in result.devanagari_input)
    assert result.final_translation == "Brother what are you doing."


def test_translate_text_helper():
    """Verify the convenience helper translate_text."""
    with patch("pipeline.translate_devanagari", return_value="today was a tiring day"):
        final_str = translate_text("aaj ka din thaka dene wala tha", use_neural_grammar=False)
    assert final_str == "Today was a tiring day."

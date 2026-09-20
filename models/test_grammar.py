"""
test_grammar.py — Unit Tests for Model 4 Grammar & Fluency Correction
======================================================================
Tests rule-based polish, capitalization, punctuation formatting,
meaning-drift protection, and interface contracts for Model 4.
"""

import pytest
from models.grammar import (
    _check_meaning_drift,
    _rule_based_polish,
    correct_grammar,
)


def test_rule_based_polish_capitalization():
    # Capitalize first letter of sentence
    assert _rule_based_polish("today was a very tiring day") == "Today was a very tiring day."
    # Capitalize standalone "i" and contractions
    assert _rule_based_polish("i understand and i'm happy") == "I understand and I'm happy."
    assert _rule_based_polish("i was looking at rich and i was dumbstruck") == "I was looking at rich and I was dumbstruck."


def test_rule_based_polish_punctuation():
    # Strip leading dot/period artifacts from decoder
    assert _rule_based_polish(".. who is watching today") == "Who is watching today?"
    assert _rule_based_polish("... hello world") == "Hello world."
    # Fix spacing before commas and periods
    assert _rule_based_polish("Sir , you are good .") == "Sir, you are good."
    # Auto-add question mark for interrogative starters
    assert _rule_based_polish("what are you doing today") == "What are you doing today?"
    assert _rule_based_polish("how are you doing") == "How are you doing?"


def test_meaning_drift_guard():
    raw = "The students were listening carefully to Khan sir."

    # Exact or minor phrasing differences preserve meaning
    safe_cand = "The students were listening carefully to Khan sir."
    assert _check_meaning_drift(raw, safe_cand) is True

    # Complete semantic swap / hallucination fails drift check
    drift_cand = "The government announced a new economic policy."
    assert _check_meaning_drift(raw, drift_cand) is False

    # Severe length collapse fails
    collapsed_cand = "Students."
    assert _check_meaning_drift(raw, collapsed_cand) is False


def test_correct_grammar_interface():
    # Empty / whitespace input
    assert correct_grammar("") == ""
    assert correct_grammar("   ") == ""

    # Rule-based mode (deterministic)
    out = correct_grammar("sir you are very good i understand", use_neural=False)
    assert out.startswith("Sir")
    assert " I understand" in out
    assert out.endswith(".")


def test_proper_nouns_and_numbers_preserved():
    text = "Khan sir taught 36 students today"
    out = correct_grammar(text, use_neural=False)
    assert "Khan sir" in out or "Khan Sir" in out or "Khan" in out
    assert "36" in out

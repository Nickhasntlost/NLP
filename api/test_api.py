"""
api/test_api.py — Unit & Integration Tests for FastAPI Application
===================================================================
Tests API routes, schemas, error states, and CORS headers.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root():
    """Root endpoint returns 200 with service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs_url" in data
    assert "health_url" in data


def test_health():
    """Health check returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "beyond-words-api"


def test_models_metadata():
    """Models endpoint explicitly declares status for all models (R3.1)."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    models = data["models"]
    assert "model_1_lid" in models
    assert "model_2_normalization" in models
    assert "model_3_translation" in models
    assert "model_4_grammar" in models
    assert "off-the-shelf" in models["model_3_translation"]["status"]


def test_translate_endpoint_mocked():
    """Translate endpoint accepts Hinglish text and returns full PipelineResult schema."""
    with patch("pipeline.translate", return_value="what are you doing today"):
        payload = {
            "text": "Bhai kya kar raha hai?",
            "use_neural_grammar": False,
        }
        response = client.post("/translate", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["input_text"] == "Bhai kya kar raha hai?"
        assert isinstance(data["preprocessed_tokens"], list)
        assert isinstance(data["lid_tags"], list)
        assert isinstance(data["normalized_hinglish"], str)
        assert isinstance(data["devanagari_input"], str)
        assert isinstance(data["raw_translation"], str)
        assert data["final_translation"] == "What are you doing today?"
        assert "timing_ms" in data
        assert "total_ms" in data["timing_ms"]


def test_translate_endpoint_empty():
    """Translate endpoint handles empty string gracefully."""
    payload = {"text": "", "use_neural_grammar": False}
    response = client.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["final_translation"] == ""
    assert data["preprocessed_tokens"] == []


def test_cors_headers():
    """CORS headers are present on responses for browser access."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/translate", headers=headers)
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers

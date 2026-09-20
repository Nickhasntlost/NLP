"""
api/main.py — FastAPI Application for Beyond Words Hinglish Translation
========================================================================
Exposes the end-to-end translation pipeline to web clients and browser extensions.

Endpoints:
  - POST /translate : Translates Hinglish / Devanagari text to fluent English.
  - GET  /health    : Health check and service readiness status.
  - GET  /models    : Model metadata and deployment transparency (RULES.md R3.1).

CORS:
  Configured to permit cross-origin requests from React dev servers and Chrome extensions.
"""

from typing import Any, Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import translate_pipeline, PipelineResult

app = FastAPI(
    title="Beyond Words — Hinglish Translation API",
    description="Context-Aware Hinglish (Romanized Hindi-English) to English Translation Pipeline",
    version="1.0.0",
)

# ── Cross-Origin Resource Sharing (CORS) ─────────────────────────────────────
# Allows React frontend (localhost:5173 / localhost:3000) and Chrome Extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request & Response Schemas ──────────────────────────────────────

class TranslationRequest(BaseModel):
    text: str = Field(
        ...,
        description="Input text in Roman Hinglish, Devanagari Hindi, or mixed code-mesh.",
        example="Bhai kya kar raha hai aajkal?",
    )
    use_neural_grammar: bool = Field(
        False,
        description="If True, applies T5 neural grammar correction. If False (default), applies sub-millisecond rule-based polish.",
    )


class TranslationResponse(BaseModel):
    input_text: str
    preprocessed_tokens: List[str]
    script_tags: List[Dict[str, str]]
    lid_tags: List[Tuple[str, str]]
    normalized_hinglish: str
    devanagari_input: str
    raw_translation: str
    final_translation: str
    timing_ms: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ── Route Handlers ───────────────────────────────────────────────────────────

@app.get("/", summary="Root Endpoint")
def read_root() -> Dict[str, str]:
    return {
        "message": "Beyond Words Hinglish Translation API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
        "models_url": "/models",
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check() -> HealthResponse:
    """Returns service health status and ready state."""
    return HealthResponse(
        status="ok",
        service="beyond-words-api",
        version="1.0.0",
    )


@app.get("/models", summary="Model Descriptions & Deployment Status")
def get_models_metadata() -> Dict[str, Any]:
    """
    Returns deployment declarations for all 4 models conforming to RULES.md R3.1:
    Explicitly distinguishes fine-tuned models from off-the-shelf baselines.
    """
    return {
        "pipeline_version": "1.0.0",
        "models": {
            "model_1_lid": {
                "name": "Token-Level Language Identification",
                "type": "Heuristic token classifier with marker vocab & Devanagari detection",
                "labels": ["HI", "EN", "OTHER"],
                "status": "in-pipeline",
            },
            "model_2_normalization": {
                "name": "Hinglish Normalization & Script Conversion",
                "type": "Rule-based slang dictionary, elongation compressor, and ITRANS transliterator",
                "status": "deterministic off-the-shelf (R3.3 documented fallback)",
            },
            "model_3_translation": {
                "name": "IndicTrans2 1B (hin_Deva -> eng_Latn)",
                "base_model": "ai4bharat/indictrans2-indic-en-1B",
                "status": "off-the-shelf (baseline retained due to domain mismatch on fine-tune per R3.1)",
            },
            "model_4_grammar": {
                "name": "Grammar & Fluency Post-Processor",
                "base_model": "vennify/t5-base-grammar-correction with rule-assisted polish and semantic drift guard",
                "status": "off-the-shelf (R3.1 documented)",
            },
        },
    }


@app.post("/translate", response_model=TranslationResponse, summary="Translate Hinglish to English")
def translate_endpoint(payload: TranslationRequest) -> TranslationResponse:
    """
    Execute the full 5-stage translation pipeline:
    1. Preprocessing (noise removal, mixed-script tokenization)
    2. Language Identification (token-level HI/EN/OTHER tagging)
    3. Normalization (elongation collapse, slang map, Devanagari transliteration)
    4. Machine Translation (IndicTrans2 1B)
    5. Grammar & Fluency Correction (orthography polish, casing, drift guard)
    """
    try:
        result: PipelineResult = translate_pipeline(
            text=payload.text,
            use_neural_grammar=payload.use_neural_grammar,
        )
        return TranslationResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Translation pipeline error: {str(exc)}",
        )

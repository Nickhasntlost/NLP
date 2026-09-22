# Architecture — Beyond Words: Context-Aware Translation of Code-Mixed Indian Languages

## 1. High-Level Pipeline

```
┌─────────────────┐   ┌───────────────┐   ┌────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│  Web Scraping    │──▶│ Preprocessing │──▶│  Model 1   │──▶│   Model 2    │──▶│   Model 3     │──▶│   Model 4    │
│  (raw corpus)    │   │ (clean/token) │   │  LID       │   │ Normalization│   │  Translation  │   │  Grammar Fix │
└─────────────────┘   └───────────────┘   └────────────┘   └──────────────┘   └───────────────┘   └──────────────┘
                                                                                                          │
                                                                                                          ▼
                                                                                                  ┌──────────────┐
                                                                                                  │ Final Output  │
                                                                                                  └──────────────┘
```

Each arrow is a hard interface boundary: the output schema of one stage is the input schema of the next. No stage may skip ahead or assume access to a later stage's internals.

## 2. Component Breakdown

### 2.1 Data Collection Layer
- **Sources:** Reddit (via `praw`), YouTube comments (via YouTube Data API / `youtube-comment-downloader`)
- **Output:** raw JSON/CSV — `{source, text, timestamp, source_id}`
- **Constraint:** data must remain in its raw, unedited form at this stage. No cleaning happens here — this is the "raw dataset" required by the project brief.

### 2.2 Preprocessing Layer (Term Work #2 & #3)
- **Input:** raw text from 2.1
- **Architectural requirement:** the preprocessing stage is split into two explicit tracks. They are both derived from the same cleaned, tokenized, script-tagged sentence, but they serve different downstream uses.

#### Track A — Term-work demo track
- **Purpose:** demonstrate the syllabus-required NLP techniques in a classroom setting
- **Steps (must run in this order):**
  1. Noise removal — strip HTML, emojis, URLs, @mentions, excess punctuation
  2. Tokenization — mixed-script aware (Devanagari + Latin)
  3. Script validation — tag each token's script (Devanagari / Latin / mixed / garbage); drop garbage tokens
  4. Stopword removal — bilingual stopword lists (English + Hindi, extendable to other pairs) for demonstration purposes
  5. Lemmatization / stemming — spaCy (English), Indic NLP Library (Hindi) when available
- **Output:** cleaned, normalized token stream for teaching and analysis — `{sentence_id, filtered_tokens[], lemmatized[]}`
- **Rule:** this is the demo track only; it must not be used to define the production model input contract.

#### Track B — Model-input track (for later Model 3 translation pipeline)
- **Purpose:** preserve the full grammatical meaning of a sentence when feeding the translation stage later
- **Step sequence:**
  1. Noise removal
  2. Tokenization
  3. Script validation
  4. Stop here — no stopword removal, no lemmatization, no grammatical stripping
- **Output:** preserved sentence representation — `{sentence_id, cleaned_text, tokens[], script_tags[]}`
- **Critical rule:** negation markers and question words must not be stripped before translation. Examples include `nahi`, `kya`, `kahan`, `kaun`, `kyun`, `kaise`, and English equivalents `not`, `no`, `what`, `where`, `who`, `why`, `how`. Translation quality depends on preserving the original sentence semantics.

The project must keep both tracks explicitly separated. The demo track may continue to use the 5-step term-work pipeline, but the model-input track is the contract for all later translation work.

### 2.3 Model 1 — Language Identification (word-level)
- **Base:** `ai4bharat/IndicBERT` or `xlm-roberta-base`, fine-tuned as token classifier
- **Input:** tokenized sentence
- **Output:** per-token language label (HI/EN/OTHER)
- **Training data:** LinCE / GLUECoS + scraped+labeled subset

### 2.4 Model 2 — Normalization & Orthographic Standardization
- **Base:** Rule-based (canonical dictionary + vowel/consonant elongation collapse + edit-distance fallback per R3.3)
- **Input:** tokenized sentence + LID tags
- **Output:** normalized sentence (spelling variants collapsed to canonical form) **in standardized Roman Hinglish**
- **Design rationale:** Preserves original Roman script so English loanwords (e.g. "project", "video", "subscribe") remain unaltered. Phonetic Devanagari transliteration is bypassed to eliminate loanword corruption.

### 2.5 Model 3 — Context-Aware Translation (core model)
- **Base:** `rudrashah/RLM-hinglish-translator` (2B Gemma-based causal language model)
- **Production deployment status:** Base model deployed with automatic detection for custom QLoRA adapter (`finetune/outputs/rlm_hinglish_lora/`).
- **Input:** Clean, normalized Roman Hinglish sentence
- **Output:** Fluent English translation
- **Inference entry point:** `models/translate.py` — exposes `translate(text: str) -> str`
- **This is the project's primary contribution — highest priority for quality and evaluation.**

#### 2.5.1 Historical Finding: Script-Mismatch & Transliteration Limitations
Initially, `ai4bharat/indictrans2-indic-en-1B` was evaluated. Because IndicTrans2 expects Devanagari script, an intermediate Roman-to-Devanagari transliteration stage was introduced. While this succeeded on basic Hindi clauses, it introduced severe **phonetic corruption on English loanwords** embedded in code-mixed text (e.g., *"project"* transliterated and translated as *"projected s."*; *"rice"* translated as *"Rich"*).

Fine-tuning IndicTrans2 on Roman Twitter text (PHINC) also produced negative transfer on YouTube comments (-6.65 BLEU regression).

#### 2.5.2 Upgraded Architecture: Direct Roman Code-Mixed MT
To eliminate the transliteration bottleneck entirely, Model 3 was upgraded to a native Roman code-mixed causal LM:

```
Normalized Roman Hinglish  ──▶  [RLM-Hinglish-Translator (Base / + LoRA)]  ──▶  English Translation
```

- **Prompt format:** `Hinglish:\n{text}\n\nEnglish:\n`
- **Loanword preservation:** Native understanding of English loanwords interleaved with Romanized Hindi.
- **LoRA integration:** Supports drop-in loading of fine-tuned LoRA weights trained on the domain-specific corpus.

### 2.6 Model 4 — Grammar / Fluency Post-Processing
- **Base:** off-the-shelf grammar correction model (e.g. `vennify/t5-base-grammar-correction`)
- **Input:** raw translation from Model 3
- **Output:** grammatically corrected final text
- **Note:** may be used as-is with no fine-tuning; this must be stated plainly in documentation.

## 3. System-Level Interfaces

| Stage | Input format | Output format |
|---|---|---|
| Scraper | N/A | JSON (raw) |
| Preprocessor | JSON (raw) | JSON (tokens + tags) |
| Model 1 | JSON (tokens) | JSON (tokens + lang labels) |
| Model 2 | JSON (tokens + labels) | Plain text (normalized Roman Hinglish) |
| Model 3 | Plain text (normalized Roman Hinglish) | Plain text (translated English) |
| Model 4 | Plain text (translated English) | Plain text (final polished English) |

Any component that changes its input/output schema must update this table in the same commit/change.

## 4. Application Layer (Deliverables)

- **Backend:** FastAPI, exposes `/translate` endpoint wrapping the full pipeline
- **Frontend:** React + Tailwind, calls backend API
- **Browser Extension:** Chrome Manifest V3, calls the same backend API — no duplicated pipeline logic
- **Database:** PostgreSQL/MongoDB — stores scraped corpus, preprocessed data, and evaluation logs only. Not part of the inference path.

## 5. Non-Goals (explicitly out of scope)

- Real-time speech-to-speech translation (future scope only)
- Support for all 22 scheduled languages (future scope only)
- On-device/offline inference (future scope only)

Anything not in this document is out of scope until this document is updated.

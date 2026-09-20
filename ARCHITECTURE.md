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

### 2.4 Model 2 — Normalization & Script Conversion
- **Base:** `mt5-small` fine-tuned seq2seq, OR rule-based (edit-distance + lookup table) if time-constrained
- **Input:** tokenized sentence + LID tags
- **Output:** normalized sentence (spelling variants collapsed to canonical form) **in Devanagari script**
- **Additional responsibility (production):** Roman-to-Devanagari transliteration of Hinglish input before Model 3. See §2.5.2.
- **Fallback rule:** if fine-tuning is infeasible within timeline, rule-based is an acceptable substitute — must be documented as such, not silently swapped in.

### 2.5 Model 3 — Context-Aware Translation (core model)
- **Base:** `ai4bharat/indictrans2-indic-en-1B`
- **Production deployment status:** Used **as-is in pretrained/baseline form** (`trust_remote_code=True`). No fine-tuning is applied in the current production pipeline.
- **Input:** Devanagari-script sentence (see §2.5.2 for how Roman Hinglish is converted before this stage)
- **Output:** fluent English translation
- **Inference entry point:** `models/translate.py` — exposes `translate(text: str) -> str`
- **This is the project's primary contribution — highest priority for quality and evaluation.**

#### 2.5.1 Fine-Tuning Experiment — Documented Negative Result (RULES.md R3.1)

A PHINC-based fine-tuning experiment was conducted (Phase 4) and produced the following result:

| Evaluation Set | Baseline BLEU | Post Fine-Tune BLEU | Verdict |
|---|---|---|---|
| PHINC-val (in-domain, Twitter register) | 19.42 | 36.45 | +16.7 — in-domain improvement |
| YouTube-eval (out-of-domain, target domain) | 14.85 | 8.20 (corrected) | **-6.65 — regression on target domain** |

**Decision:** The baseline pretrained 1B model is retained for production. Fine-tuning on PHINC narrowed the model's register toward Twitter-style text, degrading performance on the YouTube comment domain this project targets. Per RULES.md R3.1: model is **used off-the-shelf** and must be described as such in the final report.

**Parallel track:** A second fine-tuning attempt on higher-VRAM hardware may be revisited. It is not blocking the current pipeline. If that attempt yields positive out-of-domain results, this section will be updated before any architecture change is applied.

#### 2.5.2 Script-Mismatch Fix — Roman-to-Devanagari Transliteration

**Problem identified:** `indictrans2-indic-en-1B` was trained exclusively on Devanagari-script Hindi (`hin_Deva`). When fed Roman-script Hinglish (e.g. `"Bhai kya kar raha hai"`), the model echoes or hallucinates rather than translating. This is a **script-mismatch**, not a domain-mismatch.

**Fix implemented** (production, in `models/translate.py`):

```
Roman Hinglish  ─[ITRANS transliteration]→  Devanagari  ─[IndicProcessor]→  [IndicTrans2]→  English
```

- **Library:** `indic-transliteration` (pure-Python, no GPU, no fairseq dependency)
- **Scheme:** ITRANS — standard Roman-to-Devanagari phonetic mapping
- **Detection:** `_is_roman_script(text)` — if >50% of alphabetic chars are ASCII, text is treated as Hinglish and transliterated; pure Devanagari bypasses this step
- **Known limitation:** English loanwords embedded in Hinglish (e.g. `"rice"`, `"mood"`) are also transliterated to approximate Devanagari syllables (e.g. `रिचे`, `मूद्`). In most cases the model correctly infers meaning from sentence context; rare failures occur with short ambiguous loanwords (e.g. `"rice"` → model reads `रिचे` as the proper noun "Rich").
- **Validated empirically:** All 4 benchmark Hinglish sentences produced fluent, accurate English output post-fix.

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
| Model 2 | JSON (tokens + labels) | Plain text (normalized) |
| Model 3 | Plain text (normalized) | Plain text (translated) |
| Model 4 | Plain text (translated) | Plain text (final) |

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

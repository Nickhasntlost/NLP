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

### 2.4 Model 2 — Normalization
- **Base:** `mt5-small` fine-tuned seq2seq, OR rule-based (edit-distance + lookup table) if time-constrained
- **Input:** tokenized sentence + LID tags
- **Output:** normalized sentence (spelling variants collapsed to canonical form)
- **Fallback rule:** if fine-tuning is infeasible within timeline, rule-based is an acceptable substitute — must be documented as such, not silently swapped in.

### 2.5 Model 3 — Context-Aware Translation (core model)
- **Base:** `ai4bharat/indictrans2-indic-en-1B`, fine-tuned on code-mixed parallel data
- **Input:** normalized sentence
- **Output:** fluent translation in target language
- **Training data:** PHINC, CALCS, or self-built parallel set from scraped data
- **This is the project's primary contribution — highest priority for quality and evaluation.**

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

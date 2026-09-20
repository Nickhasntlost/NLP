# Tasks — Ordered Build Roadmap

Work proceeds top to bottom. Do not start a phase before the prior phase's exit criteria are met (see EVALUATION.md), except where marked "parallel-safe."

## Phase 0 — Setup
- [x] Repo structure created (`/scraper`, `/preprocessing`, `/models`, `/api`, `/frontend`, `/extension`, `/eval`)
- [x] Environments/dependencies pinned (`requirements.txt`, `package.json`)
- **Exit criteria:** project runs "hello world" end to end (dummy input → dummy output through empty pipeline stubs)

## Phase 1 — Data Collection
- [x] Reddit scraper (`praw`) targeting code-mixed subreddits
- [x] YouTube comment scraper (official API) on Hindi-medium channels
- [x] Raw corpus stored per R1.1–R1.4
- **Exit criteria:** ≥2,000–3,000 raw sentences collected, logged per R1.3

## Phase 2 — Preprocessing (Term Work #2 & #3)
- [x] Noise removal (HTML/emoji/URL/mention stripping)
- [x] Tokenization (mixed-script aware)
- [x] Script validation
- [x] Bilingual stopword removal
- [x] Lemmatization/stemming
- **Exit criteria:** cleaned dataset produced; spot-check 50 random sentences manually for correctness

## Phase 3 — Model 1: Language Identification
- [x] Acquire/prepare LID training data (LinCE/GLUECoS + labeled scraped subset)
- [x] Fine-tune IndicBERT/XLM-R token classifier
- [x] Evaluate on held-out set
- **Exit criteria:** per-token accuracy reported on held-out set; model saved and loadable

## Phase 4 — Model 3: Translation (core model) — *prioritized ahead of Model 2*
- [x] Acquire parallel data (PHINC/CALCS or self-built subset)
- [x] Fine-tune IndicTrans2 on code-mixed pairs (evaluated; negative transfer documented per R3.1)
- [x] Sanity-check on example sentences from the project brief
- [x] Script-mismatch root cause identified & resolved via transliteration
- **Exit criteria:** BLEU/METEOR computed on held-out gold set; qualitative examples collected

## Phase 5 — Model 2: Normalization
- [x] Evaluated seq2seq fine-tuning (mT5-small) feasibility vs. rule-based
- [x] Implemented rule-based normalizer + dictionary + elongation collapse + edit distance fallback (R3.3 documented)
- [x] Integrated Roman-to-Devanagari script transliteration before Model 3
- **Exit criteria:** normalization demonstrably reduces spelling-variant noise on a test sample (41.4% reduction)

## Phase 6 — Model 4: Grammar Correction
- [x] Integrate off-the-shelf grammar correction model (`vennify/t5-base-grammar-correction` with rule-assisted polish)
- [x] Verify it does not distort meaning (semantic drift guard + spot-check against Model 3 output)
- **Exit criteria:** before/after examples showing fluency improvement without meaning drift (0.0% drift, 100% polish)

## Phase 7 — Pipeline Integration
- [x] Wire Phases 2–6 into a single callable pipeline function (`pipeline.py`)
- [x] End-to-end test: raw scraped sentence → final translated+corrected output (`test_pipeline.py`, `evaluate_pipeline.py`)
- **Exit criteria:** 10 end-to-end examples run cleanly with no manual intervention (100% PASS, documented in `PHASE7_SUMMARY.md`)

## Phase 8 — API & Frontend (parallel-safe with Phase 7 once interfaces are frozen)
- [ ] FastAPI `/translate` endpoint wrapping the pipeline
- [ ] React + Tailwind frontend calling the endpoint
- **Exit criteria:** working web demo, input box → translated output shown

## Phase 9 — Browser Extension (parallel-safe once API is stable)
- [ ] Chrome Manifest V3 extension calling the same API
- **Exit criteria:** one-click translation works on a live webpage

## Phase 10 — Evaluation & Reporting
- [ ] Build/verify gold reference test set (R4.2)
- [ ] Compute BLEU/METEOR/BERTScore across full pipeline
- [ ] Compile qualitative examples per R4.3
- [ ] Write final report per RULES.md §6
- **Exit criteria:** evaluation report complete, all documentation rules (R6.1, R6.2) satisfied

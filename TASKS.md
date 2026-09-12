# Tasks — Ordered Build Roadmap

Work proceeds top to bottom. Do not start a phase before the prior phase's exit criteria are met (see EVALUATION.md), except where marked "parallel-safe."

## Phase 0 — Setup
- [ ] Repo structure created (`/scraper`, `/preprocessing`, `/models`, `/api`, `/frontend`, `/extension`, `/eval`)
- [ ] Environments/dependencies pinned (`requirements.txt`, `package.json`)
- **Exit criteria:** project runs "hello world" end to end (dummy input → dummy output through empty pipeline stubs)

## Phase 1 — Data Collection
- [ ] Reddit scraper (`praw`) targeting code-mixed subreddits
- [ ] YouTube comment scraper (official API) on Hindi-medium channels
- [ ] Raw corpus stored per R1.1–R1.4
- **Exit criteria:** ≥2,000–3,000 raw sentences collected, logged per R1.3

## Phase 2 — Preprocessing (Term Work #2 & #3)
- [ ] Noise removal (HTML/emoji/URL/mention stripping)
- [ ] Tokenization (mixed-script aware)
- [ ] Script validation
- [ ] Bilingual stopword removal
- [ ] Lemmatization/stemming
- **Exit criteria:** cleaned dataset produced; spot-check 50 random sentences manually for correctness

## Phase 3 — Model 1: Language Identification
- [ ] Acquire/prepare LID training data (LinCE/GLUECoS + labeled scraped subset)
- [ ] Fine-tune IndicBERT/XLM-R token classifier
- [ ] Evaluate on held-out set
- **Exit criteria:** per-token accuracy reported on held-out set; model saved and loadable

## Phase 4 — Model 3: Translation (core model) — *prioritized ahead of Model 2*
- [ ] Acquire parallel data (PHINC/CALCS or self-built subset)
- [ ] Fine-tune IndicTrans2 on code-mixed pairs
- [ ] Sanity-check on example sentences from the project brief
- **Exit criteria:** BLEU/METEOR computed on held-out gold set; qualitative examples collected

## Phase 5 — Model 2: Normalization
- [ ] Attempt seq2seq fine-tuning (mT5-small) on noisy/clean pairs
- [ ] If infeasible in timeline, fall back to rule-based normalizer (R3.3) — document the decision
- **Exit criteria:** normalization demonstrably reduces spelling-variant noise on a test sample

## Phase 6 — Model 4: Grammar Correction
- [ ] Integrate off-the-shelf grammar correction model
- [ ] Verify it does not distort meaning (spot-check against Model 3 output)
- **Exit criteria:** before/after examples showing fluency improvement without meaning drift

## Phase 7 — Pipeline Integration
- [ ] Wire Phases 2–6 into a single callable pipeline function
- [ ] End-to-end test: raw scraped sentence → final translated+corrected output
- **Exit criteria:** 10 end-to-end examples run cleanly with no manual intervention

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

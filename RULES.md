# Rules — Beyond Words Project

These rules are binding for all work on this project. If a request or a piece of work conflicts with a rule here, the rule wins unless this document is explicitly updated first.

## 1. Data Rules

- R1.1 — Raw scraped data must be stored unmodified in its own layer before any cleaning happens. Never overwrite raw data with cleaned data.
- R1.2 — Only scrape from sources with an accessible official API or public scraping policy (Reddit via `praw`, YouTube via official API/allowed tools). Do not scrape platforms that require bypassing paywalls or authentication walls (e.g. X/Twitter without paid API access).
- R1.3 — Every dataset used (scraped or third-party: LinCE, PHINC, CALCS, etc.) must be logged with source, size, license, and date collected.
- R1.4 — No personally identifying information (usernames, emails, phone numbers) may be retained in the stored corpus beyond what's needed for deduplication. Strip or hash before storage.

## 2. Preprocessing Rules

- R2.1 — Preprocessing steps must run in the fixed order defined in ARCHITECTURE.md §2.2. Do not reorder without updating that document.
- R2.2 — Every preprocessing function must be independently testable on a small sample input before being run on the full corpus.
- R2.3 — Script validation must never silently convert a script tag — ambiguous tokens are flagged, not guessed.

## 3. Model Rules

- R3.1 — State explicitly, for every model, whether it is (a) used as-is / off-the-shelf, (b) fine-tuned, or (c) trained from scratch. Never claim (c) unless it is literally true — for a project of this scope, (c) should not occur.
- R3.2 — Model 3 (Translation) is the primary contribution. It receives priority for time, data quality, and evaluation effort over Models 1, 2, and 4.
- R3.3 — If a fine-tuning approach for Model 2 (Normalization) proves infeasible within the timeline, fall back to rule-based normalization — but this substitution must be recorded in the project log, not silently swapped.
- R3.4 — No model may be evaluated only on its own training data. A held-out test set is mandatory for every model that is fine-tuned.

## 4. Evaluation Rules

- R4.1 — Every translation output reported as a result must be reproducible: same input, same model version, same output.
- R4.2 — BLEU/METEOR/BERTScore require a gold reference set. This reference set must be built (manually translated/verified) before evaluation claims are made — it cannot be skipped.
- R4.3 — Report both quantitative metrics (BLEU/METEOR/BERTScore) and qualitative examples (2–3 sentences showing before/after) in the final report.

## 5. Process / Ordering Rules

- R5.1 — Follow the build order in TASKS.md. Do not start a downstream stage (e.g. Model 3) before its upstream dependency (e.g. preprocessing) produces usable output, except for early prototyping on a small sample.
- R5.2 — Each stage must be evaluated against EVALUATION.md's checklist before being marked complete.
- R5.3 — Scope changes must be reflected in ARCHITECTURE.md before implementation — no undocumented scope creep (e.g. adding a 5th model, adding a new language mid-project) without updating the architecture doc first.

## 6. Documentation Rules

- R6.1 — Every model, dataset, and API endpoint must have a one-paragraph description in the final report: what it does, what it's built on, and its known limitations.
- R6.2 — Claims about "context-aware" or "novel" behavior must be backed by a concrete example (input → naive output → system output) in the report.

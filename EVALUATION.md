# Evaluation — Completion Rubric

Used to check whether a phase in TASKS.md is genuinely done, not just attempted. A phase is NOT complete until every applicable item below is checked.

## Data Collection (Phase 1)
- [ ] Source, method, and date logged for every batch scraped (R1.3)
- [ ] No PII retained (R1.4)
- [ ] Minimum viable volume reached (≥2,000 sentences) — if not, state why and what's blocking it

## Preprocessing (Phase 2)
- [ ] Manual spot-check of ≥50 sentences confirms tokenization/script tagging is correct
- [ ] Stopword removal doesn't strip content-bearing words (spot-check)
- [ ] Order of operations matches ARCHITECTURE.md §2.2

## Model 1 — LID (Phase 3)
- [ ] Held-out test accuracy reported (not training accuracy)
- [ ] Confusion between HI/EN/OTHER analyzed, not just aggregate accuracy
- [ ] Model artifact saved and reloadable

## Model 3 — Translation (Phase 4)
- [ ] BLEU/METEOR/BERTScore reported on a gold reference set built independently of training data (R4.2)
- [ ] At least 5 qualitative before/after examples included, showing naive word-for-word vs. system output
- [ ] Known failure cases documented (e.g. rare slang, long sentences, ambiguous switches)

## Model 2 — Normalization (Phase 5)
- [ ] Decision recorded: fine-tuned vs. rule-based (R3.3)
- [ ] Before/after examples showing spelling variants collapsed correctly
- [ ] No evidence of normalization changing meaning (spot-check)

## Model 4 — Grammar Correction (Phase 6)
- [ ] Confirmed off-the-shelf vs. fine-tuned status stated (R3.1)
- [ ] Before/after examples show fluency improvement
- [ ] No meaning drift introduced (spot-check against Model 3 output)

## Pipeline Integration (Phase 7)
- [ ] 10 end-to-end runs complete without manual patching
- [ ] Output schema at each interface matches ARCHITECTURE.md §3

## API / Frontend / Extension (Phases 8–9)
- [ ] API returns correct output for a known test input
- [ ] Frontend displays it correctly, including error states (e.g. empty input, API timeout)
- [ ] Extension calls the same API — no duplicated pipeline logic (R... consistency check)

## Final Report (Phase 10)
- [ ] Every model has a one-paragraph description: base, fine-tuned or not, limitations (R6.1)
- [ ] Every "context-aware"/"novel" claim has a concrete example (R6.2)
- [ ] Metrics and qualitative results both present (R4.3)
- [ ] Data sources and licenses listed

## Scoring Guide (for self-assessment before submission)
| Rating | Meaning |
|---|---|
| **Complete** | All checklist items for the phase are checked, with evidence (numbers, examples, logs) |
| **Partial** | Core function works but 1+ checklist items unaddressed — must be disclosed in report, not hidden |
| **Blocked** | Cannot proceed — document the blocker and the fallback taken (e.g. R3.3 fallback) |

No phase should be marked "Complete" in the final report unless it meets the bar above.

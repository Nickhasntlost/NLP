# Phase 4 Handoff Summary

Status: Phase 4 (Model 3 — Translation) — Data preparation COMPLETE. Fine-tuning script ready. **Upload files to Colab and run** (see `models/README_translation.md`).

---

## Step 1 — Dataset Availability Findings

### PHINC (Parallel Hindi-English Code-Mixed Corpus)

| Item | Finding |
|---|---|
| **Status** | ✅ Freely downloadable — no login, no approval gate |
| **Source** | HuggingFace: `veezbo/phinc` (community mirror of original PHINC) |
| **Original paper** | Srivastava & Singh, 2020 — "PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine Translation" |
| **Format** | `Sentence` (code-mixed Hindi-English) → `English_Translation` |
| **Raw pair count** | 13,738 |
| **Domain** | Social media / Twitter posts |
| **Script** | Latin-script Romanized Hindi + English (matches our corpus) |
| **Access method** | `datasets.load_dataset("veezbo/phinc")` — no authentication needed |

### CALCS (Code-switching Across Languages as a Career Strategy)

| Item | Finding |
|---|---|
| **Status** | BLOCKED — No accessible download |
| **Finding** | The CALCS shared task website (ritual.uh.edu/calcs-2021) is unreachable (connection timeout). All GitHub repository URLs attempted returned 404. No active HuggingFace mirror found. |
| **Action** | CALCS is ruled out for Phase 4. PHINC alone is sufficient given 11,177 clean pairs available. |

---

## Step 2 — PHINC Data Preparation Results

Script: `scripts/prepare_phinc.py`

### Quality Filters Applied

| Filter | Dropped |
|---|---|
| Empty source or target | 0 |
| Either field < 3 chars | 228 |
| Source == target (untranslated) | 287 |
| Source > 120 chars (extreme outliers) | 2,045 |
| Exact duplicate pairs | 1 |
| **TOTAL KEPT** | **11,177** |

### Train / Validation Split (seed=42, 90/10)

| Split | Pairs | File |
|---|---|---|
| **Train** | **10,059** | `data/phinc/phinc_train.jsonl` |
| **Validation** | **1,118** | `data/phinc/phinc_validation.jsonl` |
| Full filtered | 11,177 | `data/phinc/phinc_raw.jsonl` |

### Corpus Statistics (post-filter)

| Metric | Source | Target |
|---|---|---|
| Avg length (chars) | 64.6 | 63.9 |
| Min / Max length | 3 / 120 | 3 / 152 |

### Spot-Check Notes

The 10-sample spot check reveals:
- **Domain**: Twitter/social media — mix of news commentary, personal statements, celebrity mentions. Different register from our YouTube corpus (more conversational, less educational), but this is a known characteristic of PHINC.
- **Translation quality**: Generally accurate. Minor quirks: some rows keep hashtags/handles in target unchanged (acceptable); one apparent source/target swap observed (isolated).
- **Code-mixing level**: Genuine Hinglish — Roman-script Hindi interleaved with English at word and clause level. Matches our project target domain.

> [!WARNING]
> PHINC is Twitter-sourced; our inference corpus is YouTube comments. The domain mismatch is real and should be disclosed in the final report (Section R6.1). The primary fine-tuning data comes from Twitter; the demo/evaluation sentences come from YouTube. This gap will be partially bridged by the 30 manually-translated YouTube sentences.

---

## Step 3 — Manual Translation Candidates

Script: `scripts/sample_for_translation.py`

- Corpus scanned: 85 raw_filtered_*.jsonl files (6,583 comments)
- Qualifying candidates (5+ tokens, <=35 tokens, >=1 Roman-Hindi marker, no duplicates): **5,309**
- Selected for manual translation: **30** (random seed=42, sorted ascending by token count)
- Token range: **5 – 20 tokens**
- Output (plain text for hand-translation): `data/manual_translation_candidates.txt`
- Output (JSONL for later pipeline use): `data/manual_translation_candidates.jsonl`

### Flagged Sentences Requiring Attention Before Translating

| # | Sentence | Flag |
|---|---|---|
| 6 | `6 36 tiktok giraya hai mene` | Numbers `36` are garbled (rendering artifact from source). Usable. |
| 8 | `PII 9dc62185978fe77f tu pee mara lol@` | Contains PII hash placeholder. **Skip this one.** |
| 10 | `36 Years old year old ki hongi` | Similar garbled number artifact. Marginal — translate if meaning is clear. |
| 16 | `3 36 chal be panch foot ke baune was epic` | Garbled number + CarryMinati slang ("5-foot dwarf"). Very colloquial but translatable. |
| 17 | `Btw bro Navasi 89 hota hai aur 79 unyasi hota h` | Hindi number correction joke. Good code-mixed example. |
| 18 | `17 18 19 20 physics Walah ko de sab log kiss` | Garbled number prefix. Meaningful part: "sab log Physics Wallah ko kiss de." |
| 20 | `Par Adarsh Ki ja to me lipta Parivar to yaha hai` | Unclear referent (vlog inside-joke). Translate as best understood or skip. |

**Recommendation**: Skip sentence 8 (PII hash) — leaves 29 clean sentences.

### YouTube Evaluation Set — Disclosed Limitation (R6.1)

> **AI-assisted translations**: The 29 English translations in `data/youtube_eval.jsonl` were
> produced with LLM assistance rather than independent human translation, due to project time
> constraints. Some source sentences with garbled text (item IDs 6, 10, 16, 18, 20) required
> best-guess interpretation and are flagged with a `notes` field in the JSONL.
> 
> **Required language in the final report**: Describe this set as "AI-assisted reference translations."
> Do NOT claim it as a gold-standard human evaluation set. The YouTube BLEU score is indicative,
> not authoritative. This is an acceptable, disclosed limitation under RULES.md R6.1.

Evaluation set produced: `data/youtube_eval.jsonl` — **29 pairs, all non-empty, ID 8 excluded.**

---

## Files Produced This Phase

| File | Purpose |
|---|---|
| `data/phinc/phinc_raw.jsonl` | 11,177 quality-filtered PHINC pairs |
| `data/phinc/phinc_train_cleaned.jsonl` | **8,694 training pairs** (swap/ENG-src filtered, 90%) |
| `data/phinc/phinc_validation_cleaned.jsonl` | **967 validation pairs** (swap/ENG-src filtered, 10%) |
| `data/phinc/phinc_swap_investigation.txt` | Full swap/English-source investigation report |
| `data/phinc/phinc_stats.txt` | Original quality filter stats |
| `data/youtube_eval.jsonl` | **29-pair domain-matched eval set** (AI-assisted translations, see disclosure) |
| `data/manual_translation_candidates.txt` | 30 source sentences (original sampling output) |
| `models/translation_colab.py` | Fine-tuning + dual evaluation script (Colab-ready) |
| `models/README_translation.md` | Model card, run instructions, known limitations |
| `scripts/prepare_phinc.py` | PHINC download + clean + split |
| `scripts/investigate_phinc_swaps.py` | Swap/ENG-src detection and cleaning |
| `scripts/sample_for_translation.py` | Corpus sampling script (seed=42) |
| `scripts/parse_translations.py` | Translation file → JSONL parser |

---

## Colab Run Summary

Files to upload to Colab:
- `models/translation_colab.py`
- `data/phinc/phinc_train_cleaned.jsonl`
- `data/phinc/phinc_validation_cleaned.jsonl`
- `data/youtube_eval.jsonl`

See `models/README_translation.md` for exact commands, expected runtimes, and OOM fallbacks.

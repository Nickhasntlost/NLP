Phase 4 (Model 3 — Translation) — Colab training notes and model card

## Evaluation Data Disclosure (R6.1)

> **Limitation — AI-assisted translations (must be stated in final report):**
> The domain-matched YouTube evaluation set (`data/youtube_eval.jsonl`) contains 29 source sentences
> sampled from our Phase 1 YouTube corpus. The English translations were **produced with AI assistance
> (LLM-generated)** rather than independent human translation, due to project time constraints.
> Some source sentences contained garbled or ambiguous text (original item IDs 6, 10, 16, 18, 20)
> where translation required best-guess interpretation; these pairs are flagged with a `notes` field
> in the JSONL. This is a disclosed limitation: the YouTube BLEU score is indicative, not authoritative.
> It should be described as "AI-assisted reference translations" in the final report, not as a
> gold-standard human evaluation set.

---

## Model

- Base model: `ai4bharat/indictrans2-indic-en-1B`
- Task: Code-mixed Hinglish → English machine translation (seq2seq)
- Fine-tuning script: `models/translation_colab.py`
- Training environment: Google Colab (T4 GPU recommended)

### What this model is and is not

Per RULES.md R3.1: this model is **fine-tuned**, not trained from scratch.
IndicTrans2 is a pre-trained multilingual translation model from AI4Bharat.
We fine-tune it on PHINC, a code-mixed Hinglish→English parallel corpus.
The model is not trained from scratch, and does not claim to be.

### Known limitations (R6.1 obligations)

1. **Domain mismatch**: PHINC is Twitter-sourced; our inference corpus is YouTube comments.
   This must be stated in the final report. The YouTube-eval BLEU score quantifies this gap.
2. **Source language ambiguity**: IndicTrans2's `hin_Latn` tag covers Romanized Hindi.
   For mixed Devanagari+Latin text (some of our corpus), this tag may cause suboptimal tokenization.
   Document as an accepted limitation.
3. **Weak-label LID dependency**: The pipeline feeds LID-tagged output into normalization before translation.
   LID errors from Phase 3 propagate here. End-to-end evaluation (Phase 7) will expose this.
4. **PHINC data quality**: After cleaning, 1,516 rows were removed (13.6%) due to English-only sources
   or swapped pairs. This is documented in `data/phinc/phinc_swap_investigation.txt`.

---

## Training data — dataset log (R1.3 compliance)

| Dataset | Source | Size (clean) | License | Date collected | Format |
|---|---|---|---|---|---|
| PHINC | `veezbo/phinc` (HuggingFace) | 11,177 → 9,661 after swap filter | CC-BY (see original paper) | 2026-09-13 | JSONL |
| YouTube eval | **AI-assisted translations** of Phase 1 corpus sample (see disclosure above) | 29 pairs | Proprietary (YouTube ToS) | 2026-09-13 | JSONL |

Original PHINC citation:
> Srivastava, S., & Singh, D. (2020). PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine Translation. *EMNLP 2020 Findings*.

---

## Dataset pipeline

```
data/phinc/
├── phinc_raw.jsonl                  ← 11,177 pairs after initial quality filter
├── phinc_train_cleaned.jsonl        ← 8,694 training pairs (after swap filter, 90%)
├── phinc_validation_cleaned.jsonl   ← 967 validation pairs (after swap filter, 10%)
├── phinc_swap_investigation.txt     ← Full swap/ENG-SRC investigation report
└── phinc_stats.txt                  ← Original quality filter stats

data/
├── manual_translation_candidates.txt  ← 30 source sentences with hand translations
├── manual_translation_candidates.jsonl
└── youtube_eval.jsonl               ← Final domain-matched evaluation set (29 pairs)
```

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/prepare_phinc.py` | Download PHINC from HuggingFace, quality filter, split |
| `scripts/investigate_phinc_swaps.py` | Detect and remove swapped/English-only pairs |
| `scripts/sample_for_translation.py` | Sample 30 YouTube sentences for manual translation |
| `scripts/parse_translations.py` | Parse completed translation file → youtube_eval.jsonl |
| `models/translation_colab.py` | Fine-tuning + evaluation script (Colab-ready) |

---

## Colab quick-start

```bash
# 1. Install deps
!pip install -q transformers datasets sacrebleu sentencepiece accelerate bitsandbytes

# 2. Upload files to /content:
#    - models/translation_colab.py
#    - data/phinc/phinc_train_cleaned.jsonl
#    - data/phinc/phinc_validation_cleaned.jsonl
#    - data/youtube_eval.jsonl

# 3. Run (3 epochs, batch 8 — expect ~2-4h on T4)
!python translation_colab.py \
    --train_path /content/phinc_train_cleaned.jsonl \
    --val_path   /content/phinc_validation_cleaned.jsonl \
    --yt_eval_path /content/youtube_eval.jsonl \
    --output_dir /content/translation_model \
    --epochs 3 \
    --batch_size 8

# 4. If out-of-memory: reduce batch or use smaller model
!python translation_colab.py ... --batch_size 4 --fp16
# Or: --model_name ai4bharat/indictrans2-indic-en-dist-200M

# 5. Download model after training
from google.colab import files
import shutil
shutil.make_archive('/content/translation_model_export', 'zip', '/content/translation_model')
files.download('/content/translation_model_export.zip')
```

---

## Evaluation — what to report (R4.3)

Both scores must be reported separately in the final report. Do NOT average them.

| Metric | Evaluation set | Meaning |
|---|---|---|
| BLEU (PHINC-val) | 967 Twitter pairs | In-domain generalization on PHINC |
| BLEU (YouTube-eval) | 29 YouTube pairs | Domain-matched, out-of-distribution test |
| Pre-training baseline | Both sets | IndicTrans2 performance with no fine-tuning |

The delta between PHINC-val and YouTube-eval BLEU quantifies the domain mismatch.
A negative delta (YouTube < PHINC) is expected and should be discussed in the report.

---

## Model persistence

- Trained checkpoint is saved to `/content/translation_model` during training.
- Colab storage is ephemeral — download or copy to Drive after each epoch.
- The repository does not store model weights (too large for Git).
- For pipeline reuse (Phase 7), load from saved checkpoint with:
  ```python
  from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
  model = AutoModelForSeq2SeqLM.from_pretrained("/path/to/translation_model")
  tokenizer = AutoTokenizer.from_pretrained("/path/to/translation_model")
  ```

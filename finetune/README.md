# Fine-tuning IndicTrans2 — GPU Laptop Guide

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run fine-tuning (~1-2 hours on GPU)
python finetune_indictrans2.py

# 4. Evaluate results
python evaluate_model.py
```

## Folder Structure

```
finetune/
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
├── finetune_indictrans2.py      ← Main fine-tuning script (LoRA)
├── evaluate_model.py            ← Evaluation (BLEU/METEOR/BERTScore)
├── data/
│   ├── phinc_train.jsonl        ← 8,694 Hinglish→English training pairs
│   ├── phinc_val.jsonl          ← 967 validation pairs
│   ├── youtube_eval_gold.jsonl  ← 29 hand-curated test pairs
│   ├── manual_candidates.jsonl  ← 30 additional test pairs
│   └── eval_candidates.jsonl    ← 200 candidates (NEED English translations)
└── output/                      ← Created after training
    └── lora_adapter/            ← Copy this back to main laptop
```

## What This Does

1. **Loads** PHINC data (Romanized Hinglish → English parallel pairs)
2. **Transliterates** Hinglish source → Devanagari (using our dictionary)
3. **Fine-tunes** IndicTrans2 1B using **LoRA** (only ~1-2% of weights updated)
4. **Evaluates** on held-out validation set with BLEU/METEOR/BERTScore
5. **Saves** LoRA adapter (~50MB) to `output/lora_adapter/`

## After Training

Copy the `output/lora_adapter/` folder back to your main project and load it:

```python
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

base_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B")
model = PeftModel.from_pretrained(base_model, "path/to/lora_adapter")
```

## GPU Requirements

- **Minimum**: 6GB VRAM (RTX 3060, RTX 4060, etc.)
- **Recommended**: 8GB+ VRAM
- Training time: ~1-2 hours on modern GPU

## TODO Before Training

- [ ] Fill in English translations for `data/eval_candidates.jsonl` (200 pairs)
- [ ] This expands the test set from 29 → 229 pairs for credible metrics (R4.2)

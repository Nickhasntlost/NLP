"""
Phase 4 — IndicTrans2 Fine-Tuning Script for Google Colab

Base model: ai4bharat/indictrans2-indic-en-1B
Task: Seq2seq translation, code-mixed Hinglish → English
Training data: PHINC cleaned train split (phinc_train_cleaned.jsonl)
Validation data: PHINC cleaned validation split (phinc_validation_cleaned.jsonl)
Domain-matched eval: YouTube evaluation set (youtube_eval.jsonl)

== HOW TO RUN IN COLAB ==

Step 1: Create a new Colab notebook with GPU runtime
  Runtime → Change runtime type → GPU (T4 or better)

Step 2: Upload these files to Colab (or copy from Drive):
  - models/translation_colab.py          ← this script
  - data/phinc/phinc_train_cleaned.jsonl
  - data/phinc/phinc_validation_cleaned.jsonl
  - data/youtube_eval.jsonl              ← your domain-matched eval set

Step 3: Install dependencies in a Colab cell:
  !pip install -q --upgrade peft accelerate
  !pip install -q sacrebleu sentencepiece bitsandbytes
  !pip install -q IndicTransToolkit
  # Then RESTART the Colab runtime before proceeding to Step 4.
  # (Runtime → Restart runtime)
  # Do NOT try to downgrade transformers or huggingface-hub — Colab's system
  # packages prevent it. The script patches the IndicTrans2 compatibility
  # breaks at runtime instead.

Step 4: Run the script:
  !python translation_colab.py \
      --train_path /content/phinc_train_cleaned.jsonl \
      --val_path   /content/phinc_validation_cleaned.jsonl \
      --yt_eval_path /content/youtube_eval.jsonl \
      --output_dir /content/translation_model \
      --epochs 3 \
      --batch_size 8

Step 5: After training completes, download the model:
  from google.colab import files
  import shutil
  shutil.make_archive('/content/translation_model_export', 'zip', '/content/translation_model')
  files.download('/content/translation_model_export.zip')

== ESTIMATED RUNTIME ==
  Dataset: ~8,694 training pairs
  Model: indictrans2-indic-en-1B (large — ~1B params)
  Colab T4: expect 2–4 hours for 3 epochs at batch_size=8
  If OOM: reduce batch_size to 4 and/or add --fp16 flag
  Alternative: indictrans2-indic-en-dist-200M (smaller, faster, less accurate)

== IMPORTANT NOTES ==
  1. IndicTrans2 uses its own tokenizer (IndicProcessor + SentencePiece).
     The input must be prefixed with the source language tag.
     For Hinglish (Hindi script mixed Latin): use "hin_Latn" as source lang.
     For English output: use "eng_Latn" as target lang.
  2. The model is large. If Colab repeatedly crashes, switch to the 200M distilled
     variant: ai4bharat/indictrans2-indic-en-dist-200M
  3. BLEU is computed via sacrebleu on detokenized output.
     Both PHINC-validation BLEU and YouTube BLEU are reported separately.
  4. Model weights are saved to output_dir after each epoch.
     If Colab resets, download checkpoint after each epoch.
"""

import argparse
import json
import os
import sys
import types

# ── Patch 1: transformers.onnx stub ──────────────────────────────────────────
# IndicTrans2's configuration_indictrans.py imports OnnxConfig,
# OnnxSeq2SeqConfigWithPast, and compute_effective_axis_dimension from
# transformers.onnx, which was removed in transformers >= 4.40.
# Register a fully permissive stub BEFORE transformers is imported.

class _DummyBase:
    """Subclassable, callable stub returned for any attribute on the fake module."""
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return _DummyBase()
    def __call__(self, *args, **kwargs): return _DummyBase()

def _make_stub_module(name):
    mod = types.ModuleType(name)
    mod.__file__ = "<stub>"
    mod.__loader__ = None
    mod.__spec__ = None
    mod.__package__ = name
    def __getattr__(attr_name):
        return _DummyBase  # return CLASS so `class Foo(OnnxSeq2SeqConfigWithPast)` works
    mod.__getattr__ = __getattr__
    return mod

if "transformers.onnx" not in sys.modules:
    _onnx_stub = _make_stub_module("transformers.onnx")
    _onnx_utils_stub = _make_stub_module("transformers.onnx.utils")
    _onnx_stub.utils = _onnx_utils_stub
    sys.modules["transformers.onnx"] = _onnx_stub
    sys.modules["transformers.onnx.utils"] = _onnx_utils_stub
# ── End Patch 1 ───────────────────────────────────────────────────────────────

# ── Patch 3: huggingface_hub.HfFolder stub ────────────────────────────────────
# HfFolder was removed from huggingface_hub >= 0.24. Colab's system hub is
# 1.31.0 (no HfFolder), but older datasets versions (which transformers'
# trainer.py imports unconditionally) still try to import it.
# Inject a minimal stub so the import chain doesn't crash.
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
        @staticmethod
        def delete_token(): pass
    _hf_hub.HfFolder = _HfFolder
# ── End Patch 3 ───────────────────────────────────────────────────────────────

# ── Patch 4: custom IndicTrans tie_weights() vs newer transformers callers ──
# IndicTransForConditionalGeneration (loaded dynamically via trust_remote_code)
# DEFINES ITS OWN tie_weights(self) that only accepts `self`. Because it's
# defined directly on the subclass, Python calls it instead of the base
# PreTrainedModel.tie_weights() — patching the base class has no effect.
# Newer transformers versions call self.tie_weights(...) internally from
# several places (post_init, _finalize_model_loading, etc.) and have
# progressively added new kwargs over time (recompute_mapping, missing_keys,
# ...), so the custom method raises:
#   TypeError: IndicTransForConditionalGeneration.tie_weights() got an
#   unexpected keyword argument '<whatever kwarg this transformers version added>'
# Fix: try loading normally; if it fails with this specific error, find the
# dynamically-imported IndicTrans module in sys.modules, wrap its
# tie_weights method to swallow all args/kwargs, and retry the load. This
# covers any current or future extra kwarg without needing to know its name.
import sys as _sys

def load_indictrans_model(model_name, **kwargs):
    from transformers import AutoModelForSeq2SeqLM
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    except TypeError as e:
        if "tie_weights() got an unexpected keyword argument" not in str(e):
            raise
        patched = False
        for mod_name, mod in list(_sys.modules.items()):
            if not mod_name.startswith("transformers_modules"):
                continue
            for attr in list(vars(mod).values()):
                if isinstance(attr, type) and "tie_weights" in attr.__dict__:
                    _orig_tw = attr.__dict__["tie_weights"]
                    def _make_wrapper(orig_fn):
                        def _wrapper(self, *args, **kwargs):
                            return orig_fn(self)
                        return _wrapper
                    attr.tie_weights = _make_wrapper(_orig_tw)
                    patched = True
        if not patched:
            raise
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
# ── End Patch 4 ───────────────────────────────────────────────────────────────

# ── Patch 6: legacy tuple-based KV cache vs newer transformers Cache objects ─
# Newer transformers passes past_key_values as an EncoderDecoderCache object
# during generation, but IndicTrans2's custom decoder
# (modeling_indictrans.py) still does the old tuple-style access:
#   past_key_values[0][0].shape[2]
# which raises:
#   TypeError: 'EncoderDecoderCache' object is not subscriptable
# This affects both direct model.generate() calls AND Trainer's internal
# generate() calls during evaluation (predict_with_generate=True), so we
# disable KV caching globally right after loading the model rather than
# patching every call site. This makes generation recompute each step
# instead of reusing cached attention states — slower, but avoids the
# broken code path entirely and is fine for these eval set sizes.
def disable_kv_cache(model):
    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    return model
# ── End Patch 6 ───────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import sacrebleu

# ── Patch 2: _special_tokens_map bootstrap ───────────────────────────────────
# IndicTransTokenizer sets self.unk_token (and other special tokens) before
# calling super().__init__(). In newer transformers, __setattr__ for special
# tokens immediately tries to update self._special_tokens_map — but that dict
# is only created inside PreTrainedTokenizerBase.__init__(), so it doesn't
# exist yet, causing: AttributeError: IndicTransTokenizer has no attribute
# _special_tokens_map.
# Fix: lazily bootstrap the dict on first use.

from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTTB
_orig_pttb_setattr = _PTTB.__setattr__

def _patched_pttb_setattr(self, key, value):
    if "_special_tokens_map" not in self.__dict__:
        object.__setattr__(self, "_special_tokens_map", {})
    _orig_pttb_setattr(self, key, value)

_PTTB.__setattr__ = _patched_pttb_setattr
# ── End Patch 2 ───────────────────────────────────────────────────────────────


# ─── IndicTrans2 language tags ────────────────────────────────────────────────
# Source: code-mixed Hinglish (romanized Hindi + English, Latin script).
# IMPORTANT: IndicTrans2's tokenizer only recognizes a fixed set of BCP-47-style
# tags (see LANGUAGE_TAGS in the model's tokenization_indictrans.py). Romanized
# Hindi has NO tag of its own — the only Latin-script tags supported are
# kha_Latn (Khasi) and lus_Latn (Mizo), which are unrelated languages. Using
# "hin_Latn" raises: AssertionError: Invalid source language tag: hin_Latn
# There is no clean fix here without a full transliteration pipeline
# (romanized Hindi -> Devanagari via a dedicated transliteration model, plus
# word-level language ID to know which words are Hindi vs. English) — a much
# bigger undertaking than this model conditioning step.
# Pragmatic choice: tag the source as "eng_Latn" (the closest valid tag to
# what's actually on the page — Latin script) and let fine-tuning on the
# paired Hinglish->English data teach the actual mapping. The tag mainly
# needs to be valid and applied consistently; it doesn't need to be
# linguistically "correct" once the model is fine-tuned end-to-end on your data.
SRC_LANG = "eng_Latn"
TGT_LANG = "eng_Latn"

MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
# Fallback if T4 OOMs on 1B:
# MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

# ── IndicTrans2 preprocessing ─────────────────────────────────────────────────
# IMPORTANT: IndicTrans2 does NOT use manual ">>lang<<" style tags (that is an
# mBART/M2M100 convention). It requires sentences to be run through
# IndicProcessor.preprocess_batch(), which applies the correct language tag
# format internally along with normalization (numbers, punctuation, etc.).
# Passing raw ">>hin_Latn<<"-tagged strings directly to the tokenizer raises:
#   AssertionError: Invalid source language tag: >>hin_Latn<<
# because that literal string is never a tag the tokenizer's vocabulary knows.
#
# Patch 5: IndicTransToolkit's collator.py does
#   `from transformers.tokenization_utils import PreTrainedTokenizerBase`
# but newer transformers moved that class to tokenization_utils_base and
# no longer re-exports it from tokenization_utils. IndicTransToolkit's own
# __init__.py unconditionally imports its collator module (even though we
# only need IndicProcessor from it), so this breaks `import IndicTransToolkit`
# entirely unless we restore the old re-export first.
import transformers.tokenization_utils as _tok_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
if not hasattr(_tok_utils, "PreTrainedTokenizerBase"):
    _tok_utils.PreTrainedTokenizerBase = _PTB
# ── End Patch 5 ───────────────────────────────────────────────────────────────
from IndicTransToolkit import IndicProcessor
IP = IndicProcessor(inference=True)
# ── End IndicTrans2 preprocessing ────────────────────────────────────────────


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_pairs(path):
    """Load (source, target) pairs from a JSONL file.

    Tolerates extra fields (e.g. 'notes', 'id') in the YouTube eval JSONL.
    Accepts 'target' or 'translation' as the reference column name.
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("source", "").strip()
            # Accept 'target' (PHINC format) or 'translation' (alt format)
            tgt = (rec.get("target") or rec.get("translation") or "").strip()
            if src and tgt:
                pairs.append((src, tgt))
    print(f"  Loaded {len(pairs)} pairs from {os.path.basename(path)}")
    return pairs


class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_src_len=128, max_tgt_len=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        # IndicTrans2 requires IndicProcessor for tagging/normalization —
        # see "IndicTrans2 preprocessing" note near SRC_LANG/TGT_LANG above.
        src_with_tag = IP.preprocess_batch([src], src_lang=SRC_LANG, tgt_lang=TGT_LANG)[0]
        model_inputs = self.tokenizer(
            src_with_tag,
            max_length=self.max_src_len,
            truncation=True,
            padding=False,
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt,
                max_length=self.max_tgt_len,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ─── Evaluation helpers ────────────────────────────────────────────────────────

def generate_translations(model, tokenizer, pairs, batch_size=16, device="cuda"):
    """Run beam-search generation on a list of (src, tgt) pairs. Returns (hypotheses, references)."""
    model.eval()
    hypotheses = []
    references = []

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        srcs = [s for s, _ in batch_pairs]
        refs = [t for _, t in batch_pairs]

        # IndicTrans2 requires IndicProcessor for tagging/normalization —
        # see "IndicTrans2 preprocessing" note near SRC_LANG/TGT_LANG above.
        preprocessed_srcs = IP.preprocess_batch(srcs, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

        inputs = tokenizer(
            preprocessed_srcs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[TGT_LANG]
                if hasattr(tokenizer, "lang_code_to_id")
                else None,
                num_beams=4,
                max_new_tokens=128,
                early_stopping=True,
                use_cache=False,  # see Patch 6: IndicTrans2 custom code breaks on newer Cache objects
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # Postprocess (denormalizes numbers/entities IndicProcessor normalized
        # during preprocessing) to get final, comparable output text.
        decoded = IP.postprocess_batch(decoded, lang=TGT_LANG)
        hypotheses.extend(decoded)
        references.extend(refs)

    return hypotheses, references


def compute_bleu(hypotheses, references):
    """Compute corpus BLEU using sacrebleu."""
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score


def evaluate_on_set(model, tokenizer, pairs, set_name, device):
    print(f"\n  Evaluating on {set_name} ({len(pairs)} pairs)...")
    hyps, refs = generate_translations(model, tokenizer, pairs, device=device)
    bleu = compute_bleu(hyps, refs)
    print(f"  {set_name} BLEU: {bleu:.2f}")

    # Show 5 examples
    print(f"\n  {set_name} — 5 example translations:")
    for i in range(min(5, len(hyps))):
        print(f"    SRC: {pairs[i][0]}")
        print(f"    REF: {refs[i]}")
        print(f"    HYP: {hyps[i]}")
        print()

    return bleu, hyps, refs


# ─── Training compute_metrics ──────────────────────────────────────────────────

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Replace -100 (padding labels) with pad_token_id
        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in label]
            for label in labels
        ]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        return {"bleu": bleu}

    return compute_metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune IndicTrans2 on PHINC + evaluate on YouTube set")
    p.add_argument("--train_path",   required=True, help="Path to phinc_train_cleaned.jsonl")
    p.add_argument("--val_path",     required=True, help="Path to phinc_validation_cleaned.jsonl")
    p.add_argument("--yt_eval_path", required=True, help="Path to youtube_eval.jsonl")
    p.add_argument("--output_dir",   default="/content/translation_model")
    p.add_argument("--model_name",   default=MODEL_NAME)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--fp16",         action="store_true", help="Use fp16 mixed precision (faster, less VRAM)")
    p.add_argument("--max_src_len",  type=int,   default=128)
    p.add_argument("--max_tgt_len",  type=int,   default=128)
    p.add_argument("--warmup_steps", type=int,   default=200)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. Training will be extremely slow on CPU.")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_pairs = load_pairs(args.train_path)
    val_pairs   = load_pairs(args.val_path)
    yt_pairs    = load_pairs(args.yt_eval_path)
    print(f"  Train: {len(train_pairs)} | Val (PHINC): {len(val_pairs)} | YT eval: {len(yt_pairs)}")

    # ── Load model & tokenizer ─────────────────────────────────────────────────
    print(f"\nLoading model: {args.model_name}")
    print("  This may take a few minutes on first run (downloads ~2–4 GB)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = disable_kv_cache(load_indictrans_model(args.model_name))
    model.to(device)

    # ── Build datasets ─────────────────────────────────────────────────────────
    train_dataset = TranslationDataset(train_pairs, tokenizer, args.max_src_len, args.max_tgt_len)
    val_dataset   = TranslationDataset(val_pairs,   tokenizer, args.max_src_len, args.max_tgt_len)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    # ── Training arguments ─────────────────────────────────────────────────────
    # NOTE: transformers renamed `evaluation_strategy` -> `eval_strategy`
    # (deprecated/removed depending on version). Detect which name the
    # installed Seq2SeqTrainingArguments actually accepts so this works
    # across versions without pinning transformers.
    import inspect as _inspect
    _ta_params = _inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    _eval_strategy_kwarg = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        fp16=args.fp16,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",  # disable W&B/tensorboard
        **{_eval_strategy_kwarg: "epoch"},
    )

    compute_metrics = make_compute_metrics(tokenizer)

    # NOTE: transformers renamed the `tokenizer` kwarg on Trainer/Seq2SeqTrainer
    # to `processing_class` (tokenizer= was removed entirely in v5.0.0).
    # Detect which name the installed Seq2SeqTrainer actually accepts.
    _trainer_params = _inspect.signature(Seq2SeqTrainer.__init__).parameters
    _tokenizer_kwarg = "processing_class" if "processing_class" in _trainer_params else "tokenizer"

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        **{_tokenizer_kwarg: tokenizer},
    )

    # ── Baseline evaluation (before fine-tuning) ───────────────────────────────
    print("\n" + "=" * 60)
    print("PRE-TRAINING BASELINE (off-the-shelf IndicTrans2, no fine-tuning)")
    print("=" * 60)
    baseline_phinc_bleu, _, _ = evaluate_on_set(model, tokenizer, val_pairs[:200], "PHINC-val (baseline, first 200)", device)
    baseline_yt_bleu,   _, _ = evaluate_on_set(model, tokenizer, yt_pairs,         "YouTube-eval (baseline)",          device)

    # ── Fine-tuning ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINE-TUNING")
    print("=" * 60)
    train_result = trainer.train()
    print(f"\nTraining complete. Loss: {train_result.training_loss:.4f}")

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")

    # ── Post-training evaluation ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    # PHINC validation BLEU
    phinc_bleu, phinc_hyps, phinc_refs = evaluate_on_set(
        model, tokenizer, val_pairs, "PHINC-val (fine-tuned)", device
    )

    # YouTube domain-matched BLEU — SEPARATE, NOT AVERAGED
    yt_bleu, yt_hyps, yt_refs = evaluate_on_set(
        model, tokenizer, yt_pairs, "YouTube-eval (fine-tuned)", device
    )

    # ── Summary report ─────────────────────────────────────────────────────────
    report_lines = [
        "",
        "=" * 60,
        "FINAL EVALUATION SUMMARY",
        "=" * 60,
        "",
        "Training data:  PHINC cleaned train split",
        f"  Pairs: {len(train_pairs)}",
        f"  Epochs: {args.epochs}",
        f"  Batch size: {args.batch_size}",
        f"  LR: {args.lr}",
        "",
        "BLEU Scores (sacrebleu, corpus-level):",
        f"  PRE-training baseline:",
        f"    PHINC-val  (first 200): {baseline_phinc_bleu:.2f}",
        f"    YouTube-eval:           {baseline_yt_bleu:.2f}",
        f"  POST-training:",
        f"    PHINC-val:              {phinc_bleu:.2f}",
        f"    YouTube-eval:           {yt_bleu:.2f}",
        "",
        "Domain gap analysis:",
        f"  PHINC → YouTube BLEU delta: {yt_bleu - phinc_bleu:+.2f}",
        "  (Negative delta = domain mismatch hurts YouTube performance)",
        "",
        "NOTE: PHINC-val and YouTube-eval BLEU are reported separately.",
        "  Do NOT average these — they test different domains.",
        "  Per RULES.md R4.3: report both in the final report.",
    ]

    print("\n".join(report_lines))

    # Save report
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nEvaluation report saved: {report_path}")

    # Save YouTube hypotheses for qualitative review
    yt_examples_path = os.path.join(args.output_dir, "youtube_eval_translations.jsonl")
    with open(yt_examples_path, "w", encoding="utf-8") as f:
        for src_tgt, hyp in zip(yt_pairs, yt_hyps):
            f.write(json.dumps({
                "source": src_tgt[0],
                "reference": src_tgt[1],
                "hypothesis": hyp,
            }, ensure_ascii=False) + "\n")
    print(f"YouTube translation examples saved: {yt_examples_path}")


if __name__ == "__main__":
    main()

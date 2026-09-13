"""Phase 4 — Local GPU Fine-Tuning: IndicTrans2 (Hinglish -> English)

Run from the repo root (any machine with NVIDIA GPU):

    pip install -r models/requirements_translation.txt
    python models/train_translation.py

All data paths default to the correct repo-relative locations automatically.
No Colab required.
"""

import argparse
import inspect
import json
import os
import sys
import types
from contextlib import contextmanager

# ─── 1. Repo-relative default paths ──────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN = os.path.join(REPO_ROOT, "data", "phinc", "phinc_train_cleaned.jsonl")
DEFAULT_VAL   = os.path.join(REPO_ROOT, "data", "phinc", "phinc_validation_cleaned.jsonl")
DEFAULT_YT    = os.path.join(REPO_ROOT, "data", "youtube_eval.jsonl")
DEFAULT_OUT   = os.path.join(REPO_ROOT, "models", "translation_model")

MODEL_1B   = "ai4bharat/indictrans2-indic-en-1B"
MODEL_200M = "ai4bharat/indictrans2-indic-en-dist-200M"

SRC_LANG = "eng_Latn"
TGT_LANG = "eng_Latn"

_tgt_bos_id = None   # resolved after tokenizer loads


# ─── 2. Compatibility patches (same as Colab version, required for IndicTrans2) ─

# Patch 1: transformers.onnx removed in transformers >= 4.40
class _DummyBase:
    def __init__(self, *a, **kw): pass
    def __getattr__(self, n): return _DummyBase()
    def __call__(self, *a, **kw): return _DummyBase()

def _stub_module(name):
    m = types.ModuleType(name)
    m.__file__ = "<stub>"; m.__loader__ = None
    m.__spec__ = None; m.__package__ = name
    m.__getattr__ = lambda _: _DummyBase
    return m

if "transformers.onnx" not in sys.modules:
    _s = _stub_module("transformers.onnx")
    _su = _stub_module("transformers.onnx.utils")
    _s.utils = _su
    sys.modules["transformers.onnx"] = _s
    sys.modules["transformers.onnx.utils"] = _su

# Patch 3: huggingface_hub.HfFolder removed in hub >= 0.24
try:
    import huggingface_hub as _hfh
    if not hasattr(_hfh, "HfFolder"):
        class _HFF:
            get_token = staticmethod(lambda: None)
            save_token = staticmethod(lambda t: None)
            delete_token = staticmethod(lambda: None)
        _hfh.HfFolder = _HFF
except ImportError:
    pass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import sacrebleu

# Patch 2: _special_tokens_map bootstrap (IndicTransTokenizer sets special
# tokens before super().__init__, so the dict doesn't exist yet)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTTB
_orig_setattr = _PTTB.__setattr__
def _safe_setattr(self, k, v):
    if "_special_tokens_map" not in self.__dict__:
        object.__setattr__(self, "_special_tokens_map", {})
    _orig_setattr(self, k, v)
_PTTB.__setattr__ = _safe_setattr

# Patch 5: IndicTransToolkit imports PreTrainedTokenizerBase from
# transformers.tokenization_utils (moved to tokenization_utils_base)
import transformers.tokenization_utils as _tu
if not hasattr(_tu, "PreTrainedTokenizerBase"):
    _tu.PreTrainedTokenizerBase = _PTTB

try:
    from IndicTransToolkit import IndicProcessor
except ImportError:
    print("ERROR: IndicTransToolkit not installed.")
    print("  Run:  pip install IndicTransToolkit")
    sys.exit(1)

IP = IndicProcessor(inference=True)


# Patch 4: tie_weights() signature mismatch in newer transformers
def _load_model(name, **kw):
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True, **kw)
    except TypeError as e:
        if "tie_weights() got an unexpected keyword argument" not in str(e):
            raise
        for mn, mod in list(sys.modules.items()):
            if not mn.startswith("transformers_modules"):
                continue
            for attr in vars(mod).values():
                if isinstance(attr, type) and "tie_weights" in attr.__dict__:
                    orig = attr.__dict__["tie_weights"]
                    attr.tie_weights = (lambda f: lambda self, *a, **kw: f(self))(orig)
        return AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True, **kw)


# Patch 6: IndicTrans2 custom decoder breaks on newer Cache objects -> disable KV cache
def _disable_kv_cache(model):
    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    return model


# ─── 3. Data helpers ─────────────────────────────────────────────────────────

def load_pairs(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src = r.get("source", "").strip()
            tgt = (r.get("target") or r.get("translation") or "").strip()
            if src and tgt:
                pairs.append((src, tgt))
    print(f"  {len(pairs):,} pairs  ← {os.path.basename(path)}")
    return pairs


class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_src=128, max_tgt=128):
        self.pairs, self.tok = pairs, tokenizer
        self.max_src, self.max_tgt = max_src, max_tgt

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        tagged_src = IP.preprocess_batch([src], src_lang=SRC_LANG, tgt_lang=TGT_LANG)[0]
        enc = self.tok(tagged_src, max_length=self.max_src, truncation=True, padding=False)
        with _tgt_context(self.tok):
            lbl = self.tok(tgt, max_length=self.max_tgt, truncation=True, padding=False)
        enc["labels"] = lbl["input_ids"]
        return enc


# ─── 4. Target-tokenizer context helper ──────────────────────────────────────

@contextmanager
def _tgt_context(tok):
    """Safely enter as_target_tokenizer(), works across transformers versions."""
    if hasattr(tok, "as_target_tokenizer"):
        with tok.as_target_tokenizer():
            yield
    elif hasattr(tok, "_tgt_tokenize"):
        orig = tok._tokenize
        tok._tokenize = tok._tgt_tokenize
        try:
            yield
        finally:
            tok._tokenize = orig
    else:
        yield


# ─── 5. Generation & evaluation ──────────────────────────────────────────────

def generate(model, tokenizer, pairs, batch_size, device):
    model.eval()
    hyps, refs = [], []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i: i + batch_size]
        srcs  = IP.preprocess_batch([s for s, _ in chunk], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        inputs = tokenizer(srcs, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=_tgt_bos_id,
                num_beams=4,
                max_new_tokens=128,
                early_stopping=True,
                use_cache=False,
            )
        with _tgt_context(tokenizer):
            raw = tokenizer.batch_decode(out, skip_special_tokens=True)
        decoded = IP.postprocess_batch(raw, lang=TGT_LANG)
        hyps.extend(decoded)
        refs.extend(t for _, t in chunk)
    return hyps, refs


def eval_set(model, tokenizer, pairs, label, device, show_n=5):
    print(f"\n  [{label}]  {len(pairs)} pairs...")
    hyps, refs = generate(model, tokenizer, pairs, batch_size=16, device=device)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    print(f"  BLEU = {bleu:.2f}")
    for i in range(min(show_n, len(pairs))):
        print(f"    SRC: {pairs[i][0]}")
        print(f"    REF: {refs[i]}")
        print(f"    HYP: {hyps[i]}\n")
    return bleu, hyps, refs


def make_metrics(tokenizer):
    def _fn(eval_pred):
        preds, labels = eval_pred
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in row]
                  for row in labels]
        p = [x.strip() for x in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        r = [x.strip() for x in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return {"bleu": sacrebleu.corpus_bleu(p, [r]).score}
    return _fn


# ─── 6. Argument parser ──────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Beyond Words — Phase 4 local GPU training")
    ap.add_argument("--train_path",    default=DEFAULT_TRAIN)
    ap.add_argument("--val_path",      default=DEFAULT_VAL)
    ap.add_argument("--yt_eval_path",  default=DEFAULT_YT)
    ap.add_argument("--output_dir",    default=DEFAULT_OUT)
    ap.add_argument("--model_name",    default=MODEL_1B,
                    help="Use --model_name ai4bharat/indictrans2-indic-en-dist-200M for lower VRAM")
    ap.add_argument("--epochs",        type=int,   default=3)
    ap.add_argument("--batch_size",    type=int,   default=4,
                    help="Per-device batch size. Reduce to 2 if OOM.")
    ap.add_argument("--grad_accum",    type=int,   default=2,
                    help="Gradient accumulation steps. Effective batch = batch_size * grad_accum")
    ap.add_argument("--lr",            type=float, default=5e-5)
    ap.add_argument("--warmup_steps",  type=int,   default=200)
    ap.add_argument("--max_src_len",   type=int,   default=128)
    ap.add_argument("--max_tgt_len",   type=int,   default=128)
    ap.add_argument("--no_fp16",       action="store_true",
                    help="Disable fp16 (use if your GPU does not support it)")
    ap.add_argument("--bf16",          action="store_true",
                    help="Use bfloat16 — recommended on RTX 30xx/40xx (Ampere+)")
    return ap.parse_args()


# ─── 7. Main ─────────────────────────────────────────────────────────────────

def main():
    global _tgt_bos_id
    args = parse_args()

    print("\n" + "=" * 60)
    print("  BEYOND WORDS — PHASE 4  |  LOCAL GPU TRAINING")
    print("=" * 60)

    # Hardware report
    has_cuda = torch.cuda.is_available()
    device   = "cuda" if has_cuda else "cpu"
    use_fp16 = use_bf16 = False

    if has_cuda:
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        major  = torch.cuda.get_device_capability(0)[0]
        print(f"GPU  : {name}  ({vram:.1f} GB VRAM)")

        # Patch 7: single-GPU guard (IndicTrans2 breaks with DataParallel)
        if torch.cuda.device_count() > 1:
            torch.cuda.device_count = lambda: 1
            print("Note : Multiple GPUs detected — restricted to GPU 0 (IndicTrans2 guard).")

        if args.bf16 or (major >= 8 and not args.no_fp16):
            use_bf16 = True
            print("Prec : bfloat16 (Ampere/Ada)")
        elif not args.no_fp16:
            use_fp16 = True
            print("Prec : fp16")
    else:
        print("Device: CPU  ⚠ No CUDA GPU found — training will be extremely slow.")
        print("Make sure you installed the CUDA-enabled PyTorch wheel.")

    # Load data
    print("\nLoading data...")
    train_pairs = load_pairs(args.train_path)
    val_pairs   = load_pairs(args.val_path)
    yt_pairs    = load_pairs(args.yt_eval_path)

    # Load model
    print(f"\nLoading  {args.model_name}  (first run downloads ~2-4 GB)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model     = _disable_kv_cache(_load_model(args.model_name))
    model.to(device)

    # Patch 8: restore as_target_tokenizer
    if not hasattr(tokenizer, "as_target_tokenizer"):
        if hasattr(tokenizer, "_tgt_tokenize"):
            @contextmanager
            def _cm(tok=tokenizer):
                o = tok._tokenize
                tok._tokenize = tok._tgt_tokenize
                try: yield
                finally: tok._tokenize = o
            tokenizer.as_target_tokenizer = _cm
            print("Patched: tokenizer.as_target_tokenizer() restored.")
        else:
            tokenizer.as_target_tokenizer = contextmanager(lambda: (yield))

    # Resolve target BOS ID
    for tag in [TGT_LANG, f"[{TGT_LANG}]", f"<{TGT_LANG}>"]:
        cand = tokenizer.convert_tokens_to_ids(tag)
        unk  = getattr(tokenizer, "unk_token_id", None)
        if cand is not None and cand != unk:
            _tgt_bos_id = cand
            break
    print(f"BOS ID: {TGT_LANG!r} → {_tgt_bos_id}")

    # Datasets & collator
    train_ds = TranslationDataset(train_pairs, tokenizer, args.max_src_len, args.max_tgt_len)
    val_ds   = TranslationDataset(val_pairs,   tokenizer, args.max_src_len, args.max_tgt_len)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100,
        pad_to_multiple_of=8 if (use_fp16 or use_bf16) else None,
    )

    # Training args (handles renamed eval_strategy kwarg)
    ta_params  = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    eval_kwarg = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    tr_params  = inspect.signature(Seq2SeqTrainer.__init__).parameters
    tok_kwarg  = "processing_class" if "processing_class" in tr_params else "tokenizer"

    tr_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        fp16=use_fp16, bf16=use_bf16,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=25,
        save_total_limit=2,
        report_to="none",
        **{eval_kwarg: "epoch"},
    )

    trainer = Seq2SeqTrainer(
        model=model, args=tr_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=make_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        **{tok_kwarg: tokenizer},
    )

    # ── Baseline (zero-shot) ──
    print("\n" + "=" * 60)
    print("STEP 1 — ZERO-SHOT BASELINE (no fine-tuning yet)")
    print("=" * 60)
    b_phinc, _, _ = eval_set(model, tokenizer, val_pairs[:200], "PHINC-val  baseline (200)", device)
    b_yt,    _, _ = eval_set(model, tokenizer, yt_pairs,        "YouTube-eval baseline",     device)

    # ── Fine-tune ──
    print("\n" + "=" * 60)
    print(f"STEP 2 — FINE-TUNING  ({args.epochs} epochs | "
          f"batch {args.batch_size} × accum {args.grad_accum} = "
          f"effective {args.batch_size * args.grad_accum})")
    print("=" * 60)
    result = trainer.train()
    print(f"\nTraining complete.  Final loss: {result.training_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved → {args.output_dir}")

    # ── Post-training eval ──
    print("\n" + "=" * 60)
    print("STEP 3 — POST-TRAINING EVALUATION")
    print("=" * 60)
    phinc_bleu, phinc_hyps, phinc_refs = eval_set(model, tokenizer, val_pairs, "PHINC-val  fine-tuned", device)
    yt_bleu,    yt_hyps,    yt_refs    = eval_set(model, tokenizer, yt_pairs,  "YouTube-eval fine-tuned", device)

    # ── Report ──
    gpu_label = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    lines = [
        "=" * 60,
        "PHASE 4 EVALUATION REPORT",
        "=" * 60,
        f"Model   : {args.model_name}",
        f"Device  : {gpu_label}",
        f"Epochs  : {args.epochs}  |  Batch: {args.batch_size}  |  Grad Accum: {args.grad_accum}",
        f"Train   : {len(train_pairs):,} PHINC pairs",
        "",
        "BLEU (SacreBLEU corpus-level):",
        f"  Zero-shot baseline  — PHINC-val (200): {b_phinc:.2f}",
        f"  Zero-shot baseline  — YouTube-eval:     {b_yt:.2f}",
        f"  Fine-tuned          — PHINC-val:        {phinc_bleu:.2f}",
        f"  Fine-tuned          — YouTube-eval:     {yt_bleu:.2f}",
        "",
        f"Domain gap (YouTube - PHINC):  {yt_bleu - phinc_bleu:+.2f}",
        "(Negative = domain mismatch; expected and disclosed per RULES.md R6.1)",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(args.output_dir, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    with open(os.path.join(args.output_dir, "youtube_eval_translations.jsonl"), "w", encoding="utf-8") as f:
        for (src, ref), hyp in zip(yt_pairs, yt_hyps):
            f.write(json.dumps({"source": src, "reference": ref, "hypothesis": hyp},
                               ensure_ascii=False) + "\n")

    print(f"\n[DONE]  Phase 4 complete.  See {args.output_dir}/")


if __name__ == "__main__":
    main()

"""
evaluate_model.py — Evaluate fine-tuned IndicTrans2 with BLEU/METEOR/BERTScore
================================================================================
Run AFTER fine-tuning to get final metrics for the project report.

Usage:
    python evaluate_model.py
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import evaluate

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
ADAPTER_PATH = OUTPUT_DIR / "lora_adapter"


def load_eval_data(filepath: str):
    """Load evaluation JSONL with {source, target} pairs."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            source = entry.get("source", "").strip()
            target = entry.get("target", "").strip()
            if source and target:
                data.append({"source": source, "target": target})
    return data


def main():
    print("=" * 70)
    print("Model Evaluation — BLEU / METEOR / BERTScore")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer & base model
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Load fine-tuned adapter
    if ADAPTER_PATH.exists():
        print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        model_name = "IndicTrans2 + LoRA (fine-tuned)"
    else:
        print("No adapter found — evaluating BASE model")
        model = base_model
        model_name = "IndicTrans2 (baseline)"

    model = model.to(device)
    model.eval()

    # Load eval datasets
    eval_files = {
        "youtube_eval_gold": DATA_DIR / "youtube_eval_gold.jsonl",
        "phinc_val": DATA_DIR / "phinc_val.jsonl",
    }

    # Load metrics
    sacrebleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    for eval_name, eval_path in eval_files.items():
        if not eval_path.exists():
            print(f"\nSkipping {eval_name} — file not found")
            continue

        data = load_eval_data(str(eval_path))
        if not data:
            continue

        print(f"\n{'─' * 50}")
        print(f"Evaluating on: {eval_name} ({len(data)} pairs)")
        print(f"Model: {model_name}")
        print(f"{'─' * 50}")

        predictions = []
        references = []
        t0 = time.time()

        for i, item in enumerate(data):
            inputs = tokenizer(
                item["source"],
                return_tensors="pt",
                max_length=128,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128, num_beams=5)

            pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            predictions.append(pred)
            references.append(item["target"].strip())

            if i < 5:
                print(f"  [{i+1}] {item['source'][:60]}...")
                print(f"       Pred: {pred}")
                print(f"       Gold: {item['target'][:60]}")

        elapsed = time.time() - t0
        print(f"\n  Inference time: {elapsed:.1f}s ({elapsed/len(data)*1000:.0f}ms/sample)")

        # Compute metrics
        bleu = sacrebleu.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        met = meteor.compute(predictions=predictions, references=references)
        bert = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
        )

        print(f"\n  ┌─────────────────────────────┐")
        print(f"  │ BLEU:      {bleu['score']:>6.2f}          │")
        print(f"  │ METEOR:    {met['meteor']*100:>6.2f}%         │")
        print(f"  │ BERTScore: {sum(bert['f1'])/len(bert['f1'])*100:>6.2f}% (F1)    │")
        print(f"  └─────────────────────────────┘")

        # Save results
        results = {
            "model": model_name,
            "eval_set": eval_name,
            "num_samples": len(data),
            "bleu": round(bleu["score"], 2),
            "meteor": round(met["meteor"] * 100, 2),
            "bertscore_f1": round(sum(bert["f1"]) / len(bert["f1"]) * 100, 2),
        }

        results_path = OUTPUT_DIR / f"eval_results_{eval_name}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()

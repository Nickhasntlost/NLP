"""
evaluate_model.py — Evaluate Hinglish Translation Model with BLEU / chrF
========================================================================
Evaluates either the base RLM model or the fine-tuned LoRA adapter on
the held-out YouTube Gold evaluation set.

Usage:
    python evaluate_model.py
"""

import sys
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVAL_FILE = DATA_DIR / "youtube_eval_gold.jsonl"
ADAPTER_PATH = BASE_DIR / "outputs" / "rlm_hinglish_lora"
BASE_MODEL_NAME = "rudrashah/RLM-hinglish-translator"

INFERENCE_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n"


def load_eval_data(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            src = item.get("source", "").strip()
            tgt = item.get("target", "").strip()
            if src and tgt:
                data.append({"id": item.get("id", len(data) + 1), "source": src, "target": tgt})
    return data


def main():
    print("=" * 65)
    print("  Hinglish -> English Model Evaluation")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    # 1. Load Tokenizer & Model
    print(f"\nLoading Tokenizer for {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # 2. Check for LoRA Adapter
    if ADAPTER_PATH.exists() and (ADAPTER_PATH / "adapter_config.json").exists():
        print(f"\n[+] Found fine-tuned LoRA adapter at: {ADAPTER_PATH}")
        print("    Loading adapter weights on top of base model...")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
        eval_label = "Fine-Tuned Model (RLM + Custom LoRA)"
    else:
        print(f"\n[-] No LoRA adapter found at {ADAPTER_PATH}")
        print("    Evaluating BASE model out-of-the-box...")
        eval_label = "Base Model (rudrashah/RLM-hinglish-translator)"

    model.eval()

    # 3. Load Evaluation Data
    if not EVAL_FILE.exists():
        print(f"ERROR: Eval file not found: {EVAL_FILE}")
        return

    eval_data = load_eval_data(EVAL_FILE)
    print(f"\nLoaded {len(eval_data)} gold pairs from {EVAL_FILE.name}")

    print("\n" + "=" * 65)
    print(f"  RUNNING INFERENCE ({eval_label})")
    print("=" * 65)

    predictions = []
    references = []
    t_start = time.time()

    for idx, item in enumerate(eval_data):
        src = item["source"]
        ref = item["target"]
        prompt = INFERENCE_TEMPLATE.format(source=src)

        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "English:\n" in full_output:
            pred = full_output.split("English:\n")[-1].strip()
        else:
            pred = full_output.replace(prompt, "").strip()

        # Clean single trailing artifact if present
        pred = pred.split("\n")[0].strip()

        predictions.append(pred)
        references.append(ref)

        if idx < 6:
            print(f"[{idx+1}/{len(eval_data)}]")
            print(f"  Source : {src}")
            print(f"  Target : {ref}")
            print(f"  Output : {pred}")
            print("-" * 40)

    elapsed = time.time() - t_start
    print(f"\nInference completed in {elapsed:.1f}s ({elapsed/len(eval_data):.2f}s per sentence)")

    # 4. Compute Metrics
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [[r] for r in references])
        chrf = sacrebleu.corpus_chrf(predictions, [[r] for r in references])

        print("\n" + "=" * 65)
        print("  EVALUATION RESULTS")
        print("=" * 65)
        print(f"  Model Evaluated : {eval_label}")
        print(f"  Corpus BLEU     : {bleu.score:.2f}")
        print(f"  chrF++ Score    : {chrf.score:.2f}")
        print("=" * 65)

        # Save to file
        out_file = BASE_DIR / "outputs" / "evaluation_report.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": eval_label,
                "dataset": EVAL_FILE.name,
                "num_samples": len(eval_data),
                "bleu": round(bleu.score, 2),
                "chrf": round(chrf.score, 2),
                "time_seconds": round(elapsed, 2),
                "samples": [
                    {"id": eval_data[i]["id"], "source": eval_data[i]["source"], "reference": references[i], "prediction": predictions[i]}
                    for i in range(len(eval_data))
                ]
            }, f, indent=2)
        print(f"\nDetailed evaluation report saved to: {out_file}")

    except ImportError:
        print("\nInstall sacrebleu (`pip install sacrebleu`) to compute automatic BLEU/chrF metrics.")


if __name__ == "__main__":
    main()

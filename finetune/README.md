# RLM Hinglish → English Fine-Tuning Guide (GPU Laptop)

This folder contains everything needed to fine-tune the **`rudrashah/RLM-hinglish-translator`** model using **QLoRA (4-bit)** on your GPU laptop.

---

## 🚀 Quick Start on GPU Laptop

### 1. Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Fine-Tuning (~30–45 mins on RTX 3050)
```bash
python finetune_rlm.py
```

### 4. Evaluate (BLEU & chrF on YouTube Gold Set)
```bash
python evaluate_model.py
```

---

## 📁 Folder Structure

```
finetune/
├── README.md               ← Instructions (this file)
├── requirements.txt        ← All required libraries (PyTorch, Transformers, PEFT, BitsAndBytes)
├── finetune_rlm.py         ← Main QLoRA fine-tuning script (Hinglish -> English)
├── evaluate_model.py       ← BLEU / chrF evaluation script
├── data/
│   ├── phinc_train.jsonl   ← 8,695 Roman Hinglish → English training pairs
│   ├── phinc_val.jsonl     ← 967 validation pairs
│   └── youtube_eval_gold.jsonl ← Gold reference test set for BLEU evaluation
└── outputs/                ← Created after training
    └── rlm_hinglish_lora/  ← Saved LoRA adapter weights (~50MB) + eval report
```

---

## ⚙️ Hardware & Training Details

* **Base Model:** `rudrashah/RLM-hinglish-translator` (2B Gemma causal LM)
* **Optimization:** 4-bit QLoRA (`bitsandbytes` NF4)
* **Target Hardware:** NVIDIA GeForce RTX 3050 (4GB or 6GB VRAM)
* **VRAM Usage:** ~3.2 GB to 3.8 GB (fits comfortably within 6GB)
* **Training Time:** ~30 to 45 minutes for 2 epochs on 8.7K pairs

---

## 📦 What to Bring Back to Your Main Laptop

After `python finetune_rlm.py` finishes:
1. You will have a folder: `finetune/outputs/rlm_hinglish_lora/` (~50 MB).
2. You will have evaluation metrics: `finetune/outputs/evaluation_report.json`.
3. Simply commit or copy `finetune/outputs/` back to your main project repository!

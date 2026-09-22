"""
finetune_indictrans2.py — LoRA Fine-Tuning IndicTrans2 on PHINC Data
=====================================================================
Self-contained script for GPU laptop. Fine-tunes ai4bharat/indictrans2-indic-en-1B
using LoRA (parameter-efficient) on the PHINC Hinglish→English parallel corpus.

Usage:
    python finetune_indictrans2.py

Requirements:
    pip install -r requirements.txt

The script will:
1. Load PHINC train/val data (Romanized Hinglish → English)
2. Transliterate Hinglish source to Devanagari using our dictionary
3. Fine-tune IndicTrans2 with LoRA on (Devanagari, English) pairs
4. Save the fine-tuned adapter to ./output/
5. Evaluate on validation set with BLEU/METEOR/BERTScore
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"

# ── Hinglish-to-Devanagari Dictionary (from normalize.py) ───────────────────
HINGLISH_DEVA_DICT: Dict[str, str] = {
    # Pronouns & Demonstratives
    "main": "मैं", "mein": "में", "mai": "मैं", "maine": "मैंने",
    "tu": "तू", "tum": "तुम", "tune": "तूने", "aap": "आप",
    "yeh": "यह", "ye": "यह", "woh": "वो", "wo": "वो",
    "mera": "मेरा", "mere": "मेरे", "meri": "मेरी",
    "tera": "तेरा", "tere": "तेरे", "teri": "तेरी",
    "uska": "उसका", "uske": "उसके", "uski": "उसकी",
    "unka": "उनका", "unke": "उनके", "unki": "उनकी",
    "apna": "अपना", "apne": "अपने", "apni": "अपनी",
    "hum": "हम", "humein": "हमें", "humko": "हमको",
    "mujhe": "मुझे", "tujhe": "तुझे", "unhein": "उन्हें", "inhein": "इन्हें",
    "tumhe": "तुम्हें", "tumhein": "तुम्हें",
    "unhone": "उन्होंने", "inhone": "इन्होंने", "kisne": "किसने", "sabne": "सबने",
    "koi": "कोई", "kuch": "कुछ", "sab": "सब",
    "khud": "खुद", "dono": "दोनों",
    # Interrogatives
    "kya": "क्या", "kahan": "कहाँ", "kaun": "कौन", "kaise": "कैसे",
    "kyun": "क्यों", "kyunki": "क्योंकि", "kab": "कब", "kitna": "कितना",
    "kitne": "कितने", "kitni": "कितनी",
    # Auxiliary verbs
    "hai": "है", "hain": "हैं", "tha": "था", "thi": "थी", "the": "थे",
    "ho": "हो", "hoga": "होगा", "hogi": "होगी", "hoge": "होगे",
    "hona": "होना", "hota": "होता", "hoti": "होती", "hote": "होते",
    "honi": "होनी", "hue": "हुए",
    # Common verbs
    "kar": "कर", "karo": "करो", "karna": "करना", "karta": "करता",
    "karti": "करती", "karte": "करते", "karega": "करेगा", "karegi": "करेगी",
    "karenge": "करेंगे", "kara": "करा", "kari": "करी", "karne": "करने",
    "de": "दे", "dena": "देना", "dene": "देने", "deta": "देता",
    "deti": "देती", "dete": "देते", "diya": "दिया",
    "le": "ले", "lena": "लेना", "lene": "लेने", "leta": "लेता",
    "leti": "लेती", "lete": "लेते", "liya": "लिया", "liye": "लिए",
    "ja": "जा", "jao": "जाओ", "jana": "जाना", "jane": "जाने",
    "jaata": "जाता", "jaate": "जाते", "jaati": "जाती",
    "jayega": "जाएगा", "jayegi": "जाएगी",
    "aa": "आ", "aao": "आओ", "aana": "आना", "aane": "आने",
    "aata": "आता", "aate": "आते", "aati": "आती",
    "aaya": "आया", "aayi": "आयी", "aaye": "आए",
    "bol": "बोल", "bolo": "बोलो", "bola": "बोला", "bolna": "बोलना",
    "bolne": "बोलने", "bolta": "बोलता", "bolte": "बोलते", "bolti": "बोलती",
    "dekh": "देख", "dekho": "देखो", "dekha": "देखा", "dekhna": "देखना",
    "dekhne": "देखने", "dekhta": "देखता", "dekhte": "देखते", "dekhti": "देखती",
    "sun": "सुन", "suno": "सुनो", "suna": "सुना", "sunna": "सुनना",
    "sune": "सुने", "sunta": "सुनता", "sunte": "सुनते", "sunti": "सुनती",
    "padh": "पढ़", "padho": "पढ़ो", "padhna": "पढ़ना", "padhne": "पढ़ने",
    "padhta": "पढ़ता", "padhte": "पढ़ते", "padhti": "पढ़ती",
    "padhate": "पढ़ाते", "padhati": "पढ़ाती", "padhata": "पढ़ाता",
    "samajh": "समझ", "samjho": "समझो", "samajhta": "समझता",
    "samajhte": "समझते", "samajhti": "समझती", "samjhaya": "समझाया",
    "bata": "बता", "batao": "बताओ", "batana": "बताना",
    "bana": "बना", "banao": "बनाओ", "banaya": "बनाया", "banane": "बनाने",
    "kha": "खा", "khao": "खाओ", "khana": "खाना", "khaya": "खाया",
    "khata": "खाता", "khate": "खाते", "khati": "खाती",
    "pi": "पी", "peeyo": "पीयो", "peeta": "पीता", "peete": "पीते", "peeti": "पीती",
    "mil": "मिल", "milta": "मिलता", "milte": "मिलते", "milti": "मिलती",
    "lag": "लग", "lagta": "लगता", "lagte": "लगते", "lagti": "लगती",
    "chal": "चल", "chalo": "चलो", "chalta": "चलता", "chalte": "चलते", "chalti": "चलती",
    "ruk": "रुक", "ruko": "रुको",
    "kiya": "किया", "kiye": "किए",
    # Aspectual markers
    "raha": "रहा", "rahi": "रही", "rahe": "रहे", "rahna": "रहना", "rahne": "रहने",
    "rehta": "रहता", "rehte": "रहते", "rehti": "रहती",
    "gaya": "गया", "gayi": "गयी", "gaye": "गए",
    # Postpositions & Particles
    "ka": "का", "ke": "के", "ki": "की", "se": "से",
    "me": "में", "par": "पर", "tak": "तक", "ko": "को",
    "ne": "ने", "bhi": "भी", "hi": "ही", "to": "तो",
    "na": "ना", "ya": "या", "aur": "और",
    # Adverbs & Adjectives
    "bahut": "बहुत", "zyada": "ज़्यादा", "thoda": "थोड़ा",
    "achha": "अच्छा", "acha": "अच्छा", "badhiya": "बढ़िया",
    "achhi": "अच्छी", "acchi": "अच्छी", "achhe": "अच्छे",
    "sahi": "सही", "theek": "ठीक", "bura": "बुरा",
    "pehle": "पहले", "baad": "बाद", "aaj": "आज", "kal": "कल",
    "aajkal": "आजकल", "abhi": "अभी", "tab": "तब",
    "haan": "हाँ", "nahi": "नहीं", "bas": "बस", "sirf": "सिर्फ़",
    "shayad": "शायद", "zaroor": "ज़रूर", "bilkul": "बिल्कुल",
    "waqt": "वक़्त", "din": "दिन", "raat": "रात",
    # Common nouns
    "bhai": "भाई", "yaar": "यार", "dost": "दोस्त",
    "kaam": "काम", "ghar": "घर", "log": "लोग",
    "paisa": "पैसा", "paise": "पैसे", "wala": "वाला",
    "wale": "वाले", "wali": "वाली", "sir": "सर",
    "ji": "जी", "maza": "मज़ा",
    # Social / conversational
    "chahiye": "चाहिए", "chahta": "चाहता", "chahti": "चाहती", "chahte": "चाहते",
    "agar": "अगर", "jo": "जो", "sath": "साथ",
    # Numbers
    "ek": "एक", "do": "दो", "teen": "तीन", "char": "चार",
    "paanch": "पाँच", "chhe": "छे", "saat": "सात",
    "aath": "आठ", "nau": "नौ", "das": "दस",
    # Greetings
    "namaste": "नमस्ते", "namaskar": "नमस्कार",
    "dhanyavaad": "धन्यवाद", "shukriya": "शुक्रिया",
    # Misc
    "kahin": "कहीं", "idhar": "इधर", "udhar": "उधर",
    "andar": "अंदर", "bahar": "बाहर", "upar": "ऊपर", "neeche": "नीचे",
    "sach": "सच", "pyar": "प्यार", "zindagi": "ज़िन्दगी", "duniya": "दुनिया",
    "dil": "दिल", "mann": "मन", "pata": "पता", "matlab": "मतलब",
    "problem": "प्रॉब्लम", "video": "वीडियो", "class": "क्लास",
    "exam": "एग्ज़ाम", "school": "स्कूल", "college": "कॉलेज",
    "hal": "हाल", "haal": "हाल", "sa": "सा", "si": "सी",
    "accha": "अच्छा", "bada": "बड़ा", "chota": "छोटा",
    "naya": "नया", "purana": "पुराना", "taiyari": "तैयारी",
}


# ── Phonetic Fallback ────────────────────────────────────────────────────────
_PHONETIC_MAP = [
    ("ksh", "क्ष"), ("gya", "ज्ञ"), ("tra", "त्र"), ("shr", "श्र"),
    ("kh", "ख"), ("gh", "घ"), ("chh", "छ"), ("ch", "च"),
    ("jh", "झ"), ("th", "थ"), ("dh", "ध"),
    ("ph", "फ"), ("bh", "भ"), ("sh", "श"),
    ("tt", "ट्ट"), ("dd", "ड्ड"),
    ("ng", "ंग"), ("nn", "न्न"), ("mm", "म्म"),
    ("k", "क"), ("g", "ग"), ("c", "च"),
    ("j", "ज"), ("t", "त"), ("d", "द"),
    ("n", "न"), ("p", "प"), ("b", "ब"),
    ("m", "म"), ("y", "य"), ("r", "र"),
    ("l", "ल"), ("v", "व"), ("w", "व"),
    ("s", "स"), ("h", "ह"), ("z", "ज़"),
    ("f", "फ़"), ("q", "क़"), ("x", "क्स"),
    ("aa", "ा"), ("ee", "ी"), ("oo", "ू"),
    ("ai", "ै"), ("au", "ौ"), ("ei", "ै"), ("ou", "ौ"),
    ("a", ""), ("e", "े"), ("i", "ि"), ("o", "ो"), ("u", "ु"),
]

_VOWEL_MATRAS = {"ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "ं", "ः", "ँ", ""}
_INDEPENDENT_VOWELS = {
    "": "अ", "ा": "आ", "ि": "इ", "ी": "ई", "ु": "उ", "ू": "ऊ",
    "े": "ए", "ै": "ऐ", "ो": "ओ", "ौ": "औ",
}


def _transliterate_word_phonetic(word: str) -> str:
    word = word.lower()
    result = []
    i = 0
    last_was_consonant = False
    while i < len(word):
        matched = False
        for roman, deva in _PHONETIC_MAP:
            if word[i:i+len(roman)] == roman:
                is_vowel = deva in _VOWEL_MATRAS
                if is_vowel:
                    if not result:
                        result.append(_INDEPENDENT_VOWELS.get(deva, deva))
                    elif last_was_consonant:
                        if deva != "":
                            result.append(deva)
                    else:
                        result.append(_INDEPENDENT_VOWELS.get(deva, deva))
                    last_was_consonant = False
                else:
                    if last_was_consonant:
                        result.append("्")
                    result.append(deva)
                    last_was_consonant = True
                i += len(roman)
                matched = True
                break
        if not matched:
            result.append(word[i])
            last_was_consonant = False
            i += 1
    return "".join(result)


def roman_to_deva(text: str) -> str:
    """Transliterate Romanized Hinglish to Devanagari."""
    if not text or not text.strip():
        return text
    words = re.split(r'(\s+|[^\w]+)', text.lower())
    result = []
    for word in words:
        if not word or not word.strip():
            result.append(word)
            continue
        if not any(c.isalpha() for c in word):
            result.append(word)
            continue
        if any("\u0900" <= c <= "\u097F" for c in word):
            result.append(word)
            continue
        lower = word.lower().strip()
        if lower in HINGLISH_DEVA_DICT:
            result.append(HINGLISH_DEVA_DICT[lower])
        else:
            result.append(_transliterate_word_phonetic(lower))
    return "".join(result)


def is_roman_script(text: str) -> bool:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    latin = sum(1 for c in alpha if c.isascii())
    return latin / len(alpha) > 0.5


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_phinc_data(filepath: str) -> List[Dict[str, str]]:
    """Load PHINC JSONL: {source: hinglish, target: english}"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            source = entry.get("source", "").strip()
            target = entry.get("target", "").strip()
            if source and target:
                # Transliterate Roman Hinglish to Devanagari
                if is_roman_script(source):
                    deva_source = roman_to_deva(source)
                else:
                    deva_source = source
                data.append({"source": deva_source, "target": target, "original": source})
    return data


class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_len=128, max_target_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = item["source"]
        target = item["target"]

        # IndicTrans2 requires language tags set before tokenization
        # IndicTrans2 requires source and target language tags
        source = f"hin_Deva eng_Latn {source}"

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(
                target,
                max_length=self.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


# ── Main Fine-Tuning ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("IndicTrans2 LoRA Fine-Tuning on PHINC Data")
    print("=" * 70)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected! Fine-tuning on CPU will be very slow.")
        print("Consider using Google Colab or a GPU machine.")

    # ── Load Data ──
    print("\n[1/5] Loading training data...")
    train_data = load_phinc_data(str(DATA_DIR / "phinc_train.jsonl"))
    val_data = load_phinc_data(str(DATA_DIR / "phinc_val.jsonl"))
    print(f"  Train: {len(train_data)} pairs")
    print(f"  Val:   {len(val_data)} pairs")

    # Show samples
    print("\n  Sample (original → devanagari → english):")
    for i, item in enumerate(train_data[:3]):
        print(f"    {i+1}. {item['original']}")
        print(f"       → {item['source']}")
        print(f"       → {item['target']}")

    # ── Load Model & Tokenizer ──
    print("\n[2/5] Loading IndicTrans2 model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    print(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # ── Apply LoRA ──
    print("\n[3/5] Applying LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,                    # LoRA rank
        lora_alpha=32,           # Scaling factor
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.0f}M ({100*trainable/total:.2f}%)")

    # ── Create Datasets ──
    print("\n[4/5] Preparing datasets...")
    train_dataset = TranslationDataset(train_data, tokenizer)
    val_dataset = TranslationDataset(val_data, tokenizer)

    # ── Training Arguments ──
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=8,     # Adjust based on GPU VRAM
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=200,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=(device == "cuda"),
        predict_with_generate=True,
        generation_max_length=128,
        report_to="tensorboard",
        dataloader_num_workers=0,  # Must be 0 on Windows to avoid multiprocessing errors
    )

    # ── Data Collator ──
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # ── Metrics ──
    sacrebleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in labels
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # BLEU
        bleu_result = sacrebleu.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
        )
        # METEOR
        meteor_result = meteor.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        return {
            "bleu": round(bleu_result["score"], 2),
            "meteor": round(meteor_result["meteor"] * 100, 2),
        }

    # ── Train ──
    print("\n[5/5] Starting fine-tuning...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"\nTraining complete! Time: {elapsed/60:.1f} minutes")

    # ── Save ──
    model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    print(f"LoRA adapter saved to: {OUTPUT_DIR / 'lora_adapter'}")

    # ── Final Evaluation ──
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    eval_results = trainer.evaluate()
    print(f"  Loss:   {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"  BLEU:   {eval_results.get('eval_bleu', 'N/A')}")
    print(f"  METEOR: {eval_results.get('eval_meteor', 'N/A')}")

    # ── Test on sample inputs ──
    print("\n── Sample Translations (Fine-tuned) ──")
    test_inputs = [
        "क्या कर रहा है",
        "भाई बहुत मज़ा आया",
        "मेरा काम हो गया",
        "मुझे नहीं पता",
        "सर आप बहुत अच्छा पढ़ाते हो",
    ]
    model.eval()
    for text in test_inputs:
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=5)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  {text} → {translation}")

    print("\n" + "=" * 70)
    print("DONE! Copy the output/lora_adapter folder back to your main laptop.")
    print("=" * 70)


if __name__ == "__main__":
    main()

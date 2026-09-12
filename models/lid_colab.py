"""
Colab-ready script to fine-tune a token-classification LID model.

Usage in Colab (quick):
1. Upload `lid_train.jsonl` and `lid_holdout.jsonl` to Colab (or place in Drive and mount).
2. Run the cells or execute this script with Python after installing packages.

This script is intended to be run inside Google Colab with a GPU (T4).
Replace `model_name` with 'ai4bharat/indic-bert' if preferred and available.
"""

# Install required packages (run in a Colab cell):
# !pip install -q transformers datasets torch sentencepiece accelerate seqeval

import os
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


model_name = "xlm-roberta-base"
label_list = ["HI", "EN", "OTHER"]
label2id = {l: i for i, l in enumerate(label_list)}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def prepare_hf_dataset(jsonl_path):
    rows = read_jsonl(jsonl_path)
    # convert labels to ids
    for r in rows:
        r["label_ids"] = [label2id.get(x, label2id["OTHER"]) for x in r["labels"]]
    return Dataset.from_list(rows)


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding=False)
    labels = []
    for i, label in enumerate(examples["label_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # For wordpieces, use -100 to ignore or repeat same label
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    true_labels = []
    pred_labels = []
    for i in range(len(labels)):
        lb = [l for l in labels[i] if l != -100]
        pr = [predictions[i][j] for j, l in enumerate(labels[i]) if l != -100]
        true_labels.extend(lb)
        pred_labels.extend(pr)
    acc = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(label_list))))
    return {"accuracy": acc, "confusion_matrix": cm.tolist()}


def main(train_path="/content/lid_train.jsonl", hold_path="/content/lid_holdout.jsonl", independent_valid_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
    model.config.label2id = label2id
    model.config.id2label = {i: label for i, label in enumerate(label_list)}

    train_ds = prepare_hf_dataset(train_path)
    eval_ds = prepare_hf_dataset(hold_path)

    tokenized_train = train_ds.map(lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir="/content/lid_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Weak-holdout Eval metrics:", metrics)

    # If an independent validation set is provided, evaluate separately
    if independent_valid_path:
        print('Loading independent validation set:', independent_valid_path)
        ind_ds = prepare_hf_dataset(independent_valid_path)
        tokenized_ind = ind_ds.map(lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True, remove_columns=ind_ds.column_names)
        ind_metrics = trainer.evaluate(eval_dataset=tokenized_ind)
        print("Independent Eval metrics:", ind_metrics)

    # Save model
    trainer.save_model("/content/lid_model")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Colab-ready LID training/eval script")
    p.add_argument('--train_path', default='/content/lid_train.jsonl')
    p.add_argument('--hold_path', default='/content/lid_holdout.jsonl')
    p.add_argument('--independent_valid_path', default=None,
                   help='Path to auto/heuristic independent validation (optional)')
    p.add_argument('--independent_manual_path', default=None,
                   help='Path to manually-labeled independent validation (preferred primary metric)')
    args = p.parse_args()

    print("This script is intended for Colab. Running with provided paths...")
    # If a manual independent file is provided, use it as the primary independent eval.
    main(train_path=args.train_path, hold_path=args.hold_path, independent_valid_path=args.independent_valid_path)
    # If manual independent file exists, evaluate it and report as primary
    if args.independent_manual_path:
        print('\nPrimary (manual) independent validation provided; evaluating now:')
        # Run a second evaluation pass using the manual file and print clearly
        # Load tokenizer/model via main's internals by calling a small wrapper
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
        from datasets import Dataset
        def _prepare(path):
            rows = read_jsonl(path)
            for r in rows:
                r['label_ids'] = [label2id.get(x, label2id['OTHER']) for x in r['labels']]
            return Dataset.from_list(rows)
        try:
            ind_ds = _prepare(args.independent_manual_path)
            tokenized_ind = ind_ds.map(lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True, remove_columns=ind_ds.column_names)
            # create a Trainer for evaluation only
            from transformers import Trainer, TrainingArguments
            data_collator = DataCollatorForTokenClassification(tokenizer)
            tr_args = TrainingArguments(output_dir='/content/lid_model_eval_tmp', per_device_eval_batch_size=8)
            trainer = Trainer(model=model, args=tr_args, data_collator=data_collator, tokenizer=tokenizer, compute_metrics=compute_metrics)
            ind_metrics = trainer.evaluate(eval_dataset=tokenized_ind)
            print('Primary (manual) Independent Eval metrics:', ind_metrics)
        except Exception as e:
            print('Failed to evaluate manual independent file:', e)

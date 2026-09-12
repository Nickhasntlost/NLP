Phase 3 (Model 1 — LID) dataset & Colab instructions

Files generated locally:
- `models/lid_full.jsonl` — full weak-labeled dataset (token-level labels)
- `models/lid_sample_100.jsonl` — random 100-sample for small-sample testing
- `models/lid_train.jsonl` — 80% train split
- `models/lid_holdout.jsonl` — 20% held-out split for evaluation
- `models/lid_dataset.zip` — packaged dataset for upload

Local preparation notes:
- Labels were produced with a deterministic heuristic using `preprocessing/preprocess_text(track='model_input')` tokens and script tags.
- Heuristic rules: Devanagari tokens -> `HI`; Latin tokens in `data/roman_hindi_markers.json` -> `HI`; Latin tokens otherwise -> `EN`; mixed/garbage -> `OTHER`.
- This is a weak-label baseline and should be reviewed/curated if you need higher-quality training labels.

Evaluation Limitations:
- The held-out evaluation set was created using the same weak / heuristic labeling scheme as the training data (`roman_hindi_markers.json` lookup + Devanagari detection), not an independently verified or manually labeled gold set.
- In other words, the reported held-out accuracy is an optimistic upper bound on real-world performance, not a guarantee of true generalization.
- This is a known limitation of the current Phase 3 setup and should be described clearly in the final report. The weak-holdout metric is useful as a sanity check, but it is not a fully validated measure of production performance.
- A small manually-reviewed validation set (for example, 50 sentences) would be a natural next step and would provide a much more trustworthy evaluation signal.

Colab usage (quick):
1. Open a new Colab notebook with GPU runtime (Runtime → Change runtime type → GPU).
2. Upload `models/lid_train.jsonl` and `models/lid_holdout.jsonl` (or upload `models/lid_dataset.zip` and unzip).
   - Alternatively, mount your Google Drive and copy the files there. Example:

```python
from google.colab import drive
drive.mount('/content/drive')
# then copy from Drive to /content or point the script to the Drive path
```

3. Install dependencies in a Colab cell:

```bash
!pip install -q transformers datasets torch sentencepiece accelerate scikit-learn
```

4. Run the training script in notebook cells or invoke the provided script:

```bash
# If uploading the files directly to Colab's /content:
python /content/lid_colab.py --train_path /content/lid_train.jsonl --hold_path /content/lid_holdout.jsonl

# Or, open `models/lid_colab.py` in an editor cell and run the functions directly
```

5. After training, the model and checkpoints will be saved to `/content/lid_model` — download or copy to Drive.

Optional secondary signal (not authoritative):
- The file `lid_independent_valid_auto.jsonl` is kept as a secondary, heuristic-only validation signal.
- It is not an authoritative ground truth set and should only be treated as a rough sanity check, not a final metric.

Results (important):
- 99.93% accuracy reflects performance against weak / heuristic-generated labels, not verified ground truth.
- This number should be read as: "how well the model reproduced the labeling heuristic," not as real-world LID accuracy.
- A qualitative sanity check on unseen sentences was used instead of full manual evaluation due to project time constraints.

Estimated training time on a Google Colab T4 (very approximate):
- Dataset size: ~6.5k sentences (~83k tokens)
- Fine-tuning `xlm-roberta-base` for token classification for 3 epochs with `per_device_train_batch_size=8` on a single T4: expect ~30–90 minutes depending on IO and exact batch sizing. Using `ai4bharat/IndicBERT` may be similar or faster depending on model size; the T4 resource limits and colab background variability can extend runtime.

Notes & next steps:
- The dataset is weak-labeled; if you prefer higher-quality labels, we can produce a small manually-corrected subset (e.g., 500 sentences) to use for validation or fine-tuning.
- I did not run training locally because the machine lacks a GPU; the Colab script is ready to run as-is.

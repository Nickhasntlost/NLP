import json, random
random.seed(99)
rows = []
with open('data/phinc/phinc_train.jsonl', encoding='utf-8') as f:
    for line in f:
        rows.append(json.loads(line))

sample = random.sample(rows, 10)
for i, r in enumerate(sample, 1):
    src = r["source"]
    tgt = r["target"]
    print(f"{i}. SRC: {src}")
    print(f"   TGT: {tgt}")
    print()

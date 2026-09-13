import json

rows = []
with open('data/youtube_eval.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

for r in rows:
    rid = r["id"]
    src = r["source"][:70]
    tgt = r["target"][:70]
    print(f"[{rid:2d}] SRC: {src}")
    print(f"       TGT: {tgt}")

print(f"\nTotal pairs: {len(rows)}")
flagged = [r for r in rows if r.get("notes", "")]
print(f"Flagged (notes non-empty): {len(flagged)} — IDs: {[r['id'] for r in flagged]}")

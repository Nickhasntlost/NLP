import os
import json
import glob
from collections import Counter

from qa_spotcheck import classify_text

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')

def main():
    pattern = os.path.join(RAW_DIR, 'raw_filtered_*.jsonl')
    files = glob.glob(pattern)
    counts = Counter()
    total = 0
    examples = {k: [] for k in ['emoji-only','pure english','pure devanagari','code-mixed','other/unclear']}

    if not files:
        print('No filtered files found matching', pattern)
        return

    for path in files:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get('text','')
                cat = classify_text(text)
                counts[cat] += 1
                total += 1
                if len(examples[cat]) < 5:
                    examples[cat].append(text)

    print('Files scanned:', len(files))
    print('Total comments scanned:', total)
    for cat in ['emoji-only','pure english','pure devanagari','code-mixed','other/unclear']:
        c = counts.get(cat,0)
        pct = (c/total*100) if total>0 else 0
        print(f"{cat}: {c} ({pct:.1f}%)")
        if examples[cat]:
            print('  examples:')
            for ex in examples[cat]:
                print('   -', ex.replace('\n',' ')[:200])

if __name__ == '__main__':
    main()

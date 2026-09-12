import os
import json
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(__file__), '..')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
HOLDOUT_PATH = os.path.join(RAW_DIR, 'holdout_unbiased_sample.jsonl')

def scan():
    files = [f for f in os.listdir(RAW_DIR) if f.startswith('raw_filtered_') and f.endswith('.jsonl')]
    per_video = Counter()
    per_file_counts = {}
    category_counts = Counter()
    samples_educ = []
    samples_big = []

    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        cnt = 0
        # category inference from filename
        if '_educational_' in fname:
            cat = 'educational'
        elif '_big_creator_' in fname or '_bigcreator_' in fname:
            cat = 'big_creator'
        elif '_seed' in fname:
            cat = 'seed'
        else:
            cat = 'other'

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                src = rec.get('source','')
                # source like youtube/{video_id}
                vid = src.split('/')[-1] if src else fname
                per_video[vid] += 1
                cnt += 1
                category_counts[cat] += 1
                # collect samples
                if cat == 'educational' and len(samples_educ) < 10:
                    samples_educ.append(rec.get('text',''))
                if cat in ('big_creator','other') and len(samples_big) < 10:
                    samples_big.append(rec.get('text',''))

        per_file_counts[fname] = cnt

    total = sum(per_video.values())
    # per-source distribution (top 20)
    top20 = per_video.most_common(20)

    # check per-video cap violations (cap 200)
    violations = {v:c for v,c in per_video.items() if c > 200}

    # holdout check
    holdout_count = 0
    if os.path.exists(HOLDOUT_PATH):
        with open(HOLDOUT_PATH, 'r', encoding='utf-8') as f:
            for _ in f:
                holdout_count += 1

    return {
        'total_comments': total,
        'top20_per_video': top20,
        'per_file_counts': per_file_counts,
        'category_counts': dict(category_counts),
        'violations': violations,
        'holdout_count': holdout_count,
        'samples_educ': samples_educ[:10],
        'samples_big': samples_big[:10]
    }

if __name__ == '__main__':
    import pprint
    report = scan()
    pprint.pprint(report)

import os
import json
import random
import re
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), '..')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
ARCHIVE_DIR = os.path.join(RAW_DIR, 'raw_archive_phase1a')
os.makedirs(ARCHIVE_DIR, exist_ok=True)

ROMAN_HINDI_MARKERS = set([
    'hai','hain','kya','nahi','bhai','yaar','acha','achha',
    'kar','karo','ka','ke','ki','tu','tum','main','mein','kaun','kaha','kahan',
    'kuch','bhi','raha','rahi','gaya','gayi','log','sab'
])

TOKEN_RE = re.compile(r"[A-Za-z']+")

def load_all_filtered_records():
    files = [f for f in os.listdir(RAW_DIR) if f.startswith('raw_filtered_') and f.endswith('.jsonl')]
    records = []  # list of (rec, fname)
    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        if os.path.isdir(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                records.append((rec, fname))
    return records

def passes_strict_check(text: str) -> bool:
    if not text or not text.strip():
        return False
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    if not tokens:
        return False
    has_marker = any(t in ROMAN_HINDI_MARKERS for t in tokens)
    has_non_marker = any(t not in ROMAN_HINDI_MARKERS for t in tokens)
    return has_marker and has_non_marker

def sample_and_audit(n=100, seed=42):
    records = load_all_filtered_records()
    total = len(records)
    random.seed(seed)
    sample = random.sample(records, min(n, total))
    fails = []
    for rec, fname in sample:
        text = rec.get('text','')
        if not passes_strict_check(text):
            fails.append((text, fname))
    fail_rate = len(fails)/len(sample) if sample else 0
    return {'sampled': len(sample), 'fails': fails, 'fail_rate': fail_rate, 'total_filtered': total}

def clean_corpus():
    # read each filtered file and split passed vs failed, write pass back, move failed to archive
    files = [f for f in os.listdir(RAW_DIR) if f.startswith('raw_filtered_') and f.endswith('.jsonl')]
    moved_count = 0
    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        if os.path.isdir(path):
            continue
        passed = []
        failed = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get('text','')
                if passes_strict_check(text):
                    passed.append(rec)
                else:
                    failed.append(rec)
        # overwrite filtered file with passed only
        with open(path, 'w', encoding='utf-8') as f:
            for r in passed:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        if failed:
            archive_fname = f'archived_falsepos_{fname}'
            archive_path = os.path.join(ARCHIVE_DIR, archive_fname)
            with open(archive_path, 'w', encoding='utf-8') as af:
                for r in failed:
                    af.write(json.dumps(r, ensure_ascii=False) + '\n')
            moved_count += len(failed)
    return moved_count

if __name__ == '__main__':
    report = sample_and_audit(100)
    print('Sampled:', report['sampled'], 'Total filtered:', report['total_filtered'])
    print('Fail count:', len(report['fails']), 'Fail rate:', f"{report['fail_rate']*100:.1f}%")
    if report['fails']:
        print('Examples of fails:')
        for t, fname in report['fails'][:10]:
            print('-', t.replace('\n',' ')[:200], '...from', fname)

    # if fail rate > 5% return code indicating cleaning recommended
    if report['fail_rate'] > 0.05:
        moved = clean_corpus()
        print('Cleaned corpus: moved', moved, 'false positives to archive')
        # recompute total
        records_after = load_all_filtered_records()
        print('Total after cleaning:', len(records_after))
    else:
        print('Fail rate acceptable; no cleaning performed')

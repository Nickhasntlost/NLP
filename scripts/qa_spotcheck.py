import os
import json
import re
from collections import Counter

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

ROMAN_HINDI_MARKERS = set([
    'hai','hain','kya','nahi','bhai','yaar','acha','achha',
    'kar','karo','ka','ke','ki','tu','tum','main','mein','kaun','kaha','kahan',
    'kuch','bhi','raha','rahi','gaya','gayi','log','sab'
])

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
LATIN_RE = re.compile(r'[A-Za-z]')

def classify_text(text: str) -> str:
    if not text or not text.strip():
        return 'other/unclear'
    has_latin = bool(re.search(LATIN_RE, text))
    has_deva = bool(re.search(DEVANAGARI_RE, text))
    # emoji-only / no alphabetic content
    if not has_latin and not has_deva:
        return 'emoji-only'

    # pure Devanagari (contains Devanagari and no Latin letters)
    if has_deva and not has_latin:
        return 'pure devanagari'

    # tokenise latin words
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    has_roman_hindi = any(t in ROMAN_HINDI_MARKERS for t in tokens)
    # pure English: has Latin, no Roman Hindi markers and no Devanagari
    if has_latin and not has_roman_hindi and not has_deva:
        return 'pure english'

    # genuinely code-mixed: has Latin words and romanized hindi markers
    if has_latin and has_roman_hindi:
        return 'code-mixed'

    # other/unclear
    return 'other/unclear'

def scan_raw_files():
    counts = Counter()
    total = 0
    examples = {k: [] for k in ['emoji-only','pure english','pure devanagari','code-mixed','other/unclear']}
    if not os.path.isdir(RAW_DIR):
        print('No raw dir', RAW_DIR)
        return
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(RAW_DIR, fname)
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
    scan_raw_files()

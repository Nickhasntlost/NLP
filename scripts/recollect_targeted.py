import os
import json
import time
import shutil
from collections import Counter
from scraper.youtube_scraper import search_videos, scrape_video_comments, save_raw_batch

ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
BACKUP_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_low_quality')
LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'datasets_log.jsonl')

ROMAN_HINDI_MARKERS = set(['hai','hain','kya','nahi','bhai','yaar','acha','achha','kar','karo','ka','ke','ki','tu','tum','main','mein','kaun','kaha','kahan','kuch','bhi','raha','rahi','gaya','gayi','log','sab'])

def read_env(key):
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k == key:
                    return v.strip()
    return os.environ.get(key)

def classify_text_simple(text):
    import re
    if not text or not text.strip():
        return 'other/unclear'
    has_latin = bool(re.search(r'[A-Za-z]', text))
    has_deva = bool(re.search(r'[\u0900-\u097F]', text))
    if not has_latin and not has_deva:
        return 'emoji-only'
    if has_deva and not has_latin:
        return 'pure devanagari'
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    has_roman_hindi = any(t in ROMAN_HINDI_MARKERS for t in tokens)
    if has_latin and not has_roman_hindi and not has_deva:
        return 'pure english'
    if has_latin and has_roman_hindi:
        return 'code-mixed'
    return 'other/unclear'

def analyze_raw_files():
    per_file = {}
    totals = Counter()
    total_count = 0
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(RAW_DIR, fname)
        c = Counter()
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                cat = classify_text_simple(rec.get('text',''))
                c[cat] += 1
                totals[cat] += 1
                count += 1
        per_file[fname] = {'counts': dict(c), 'total': count}
        total_count += count
    return per_file, totals, total_count

def move_low_quality(per_file, threshold=0.25):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    moved = []
    for fname, info in per_file.items():
        total = info['total']
        code_mixed = info['counts'].get('code-mixed', 0)
        frac = (code_mixed/total) if total>0 else 0
        if frac < threshold:
            src = os.path.join(RAW_DIR, fname)
            dst = os.path.join(BACKUP_DIR, fname)
            shutil.move(src, dst)
            moved.append((fname, total, code_mixed, frac))
            # Append a log entry noting the move
            with open(LOG_PATH, 'a', encoding='utf-8') as lf:
                lf.write(json.dumps({'action':'move_low_quality','file':dst,'orig_file':fname,'code_mixed_fraction':frac}) + '\n')
    return moved

def recollect(api_key, target_min=2000, per_video_limit=500, target_code_mixed_frac=0.5, max_rounds=5):
    # targeted queries likely to yield code-mixing
    queries = ['vlog hindi english mix','Hinglish','college life vlog India','daily vlog India','study with me Hindi','relationship advice Hindi','career advice India','personal vlog hindi english']
    collected_files = []
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        print('Recollection round', rounds)
        # discover videos
        candidates = []
        for q in queries:
            try:
                res = search_videos(api_key, q, max_results=5)
                for vid, title in res:
                    candidates.append((vid, title, q))
                time.sleep(0.5)
            except Exception as e:
                print('search failed', e)
        # dedupe
        seen = set()
        videos = []
        for vid, title, q in candidates:
            if vid not in seen:
                seen.add(vid)
                videos.append((vid, title, q))

        for vid, title, q in videos:
            print('Fetching', vid, title)
            try:
                records = scrape_video_comments([vid], api_key=api_key, max_results=per_video_limit)
            except Exception as e:
                print('fetch failed', e)
                continue
            if not records:
                continue
            path = save_raw_batch(records, f'recollect_{vid}', method='youtube_api')
            collected_files.append(path)
            # analyze after each batch
            per_file, totals, total_count = analyze_raw_files()
            code_mixed = totals.get('code-mixed',0)
            frac = (code_mixed/total_count) if total_count>0 else 0
            print('After collection: total comments', total_count, 'code-mixed fraction', frac)
            if total_count >= target_min and frac >= target_code_mixed_frac:
                return collected_files, per_file, totals, total_count

    # return whatever collected
    per_file, totals, total_count = analyze_raw_files()
    return collected_files, per_file, totals, total_count

if __name__ == '__main__':
    api_key = read_env('YOUTUBE_API_KEY')
    if not api_key:
        print('YOUTUBE_API_KEY missing')
        raise SystemExit(1)

    per_file, totals, total_count = analyze_raw_files()
    print('Before recollect: total', total_count, 'code-mixed', totals.get('code-mixed',0))
    moved = move_low_quality(per_file, threshold=0.25)
    print('Moved low-quality files:', moved)
    collected_files, per_file_after, totals_after, total_after = recollect(api_key)
    print('Recollection finished. New totals:', total_after, totals_after)

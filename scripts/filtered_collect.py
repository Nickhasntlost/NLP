"""Filtered collection: create unbiased holdout, archive existing raw, then collect only code-mixed comments.

Steps:
- Create a random unbiased holdout sample (N=400) from existing active raw files and save to data/raw/holdout_unbiased_sample.jsonl
- Move existing active raw JSONL files to data/raw/raw_archive_phase1a/ (excluding raw_low_quality and the holdout)
- Run targeted collection: seed with high-yield videos and search queries; for each video's comments keep only those containing roman-hindi markers
- Stop when code-mixed saved count reaches target (5000)
- Log methodology entry to logs/datasets_log.jsonl
"""
import os
import json
import random
import time
from datetime import datetime
from scraper.youtube_scraper import search_videos, scrape_video_comments, save_raw_batch

ROOT = os.path.join(os.path.dirname(__file__), '..')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
LOW_QUAL_DIR = os.path.join(RAW_DIR, 'raw_low_quality')
ARCHIVE_DIR = os.path.join(RAW_DIR, 'raw_archive_phase1a')
HOLDOUT_PATH = os.path.join(RAW_DIR, 'holdout_unbiased_sample.jsonl')
LOG_PATH = os.path.join(ROOT, 'logs', 'datasets_log.jsonl')
ENV_PATH = os.path.join(ROOT, '.env')

ROMAN_HINDI_MARKERS = set(['hai','hain','kya','nahi','bhai','yaar','acha','achha','kar','karo','ka','ke','ki','tu','tum','main','mein','kaun','kaha','kahan','kuch','bhi','raha','rahi','gaya','gayi','log','sab','acha','acha'])

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

def reservoir_sample_holdout(n=400):
    # reservoir sampling across all active raw files (exclude low_quality dir and archive)
    reservoir = []
    count = 0
    for fname in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, fname)
        if not fname.endswith('.jsonl'):
            continue
        if fname == os.path.basename(HOLDOUT_PATH):
            continue
        if os.path.isdir(path):
            continue
        # skip files in low_quality dir (those are in RAW_DIR/raw_low_quality)
        if path.startswith(LOW_QUAL_DIR):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                count += 1
                if len(reservoir) < n:
                    reservoir.append(rec)
                else:
                    s = random.randint(1, count)
                    if s <= n:
                        reservoir[s-1] = rec

    # save reservoir to HOLDOUT_PATH
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(HOLDOUT_PATH, 'w', encoding='utf-8') as fw:
        for r in reservoir:
            fw.write(json.dumps(r, ensure_ascii=False) + '\n')

    meta = {'action':'create_holdout','path': os.path.relpath(HOLDOUT_PATH, ROOT), 'size': len(reservoir), 'date': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}
    with open(LOG_PATH, 'a', encoding='utf-8') as lf:
        lf.write(json.dumps(meta, ensure_ascii=False) + '\n')
    return len(reservoir)

def archive_active_raw():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    moved = []
    for fname in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, fname)
        if not fname.endswith('.jsonl'):
            continue
        if fname == os.path.basename(HOLDOUT_PATH):
            continue
        # skip files in low_quality dir
        if fname.startswith('raw_') and os.path.isfile(path):
            dst = os.path.join(ARCHIVE_DIR, fname)
            os.replace(path, dst)
            moved.append(dst)
    meta = {'action':'archive_phase1a','moved_count': len(moved), 'date': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}
    with open(LOG_PATH, 'a', encoding='utf-8') as lf:
        lf.write(json.dumps(meta, ensure_ascii=False) + '\n')
    return moved

def contains_roman_hindi(text):
    import re
    if not text or not text.strip():
        return False
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return any(t in ROMAN_HINDI_MARKERS for t in tokens)

def filtered_collect(api_key, target_code_mixed=5000, per_video_limit=500, per_video_cap=200, seed_videos=None, queries=None):
    saved_count = 0
    # seed videos: prioritize known high-yield IDs
    default_seed_videos = ['IJcaR0x0rmc']
    # default queries
    default_queries = ['Hinglish vlog','Hinglish','vlog hindi english mix','college life vlog India','daily vlog India','relationship advice Hindi','study with me Hindi','personal vlog hindi english']

    if seed_videos is None:
        seed_videos = default_seed_videos
    if queries is None:
        queries = default_queries

    # helper to save with per-video cap and category tag
    def save_limited(keep_list, vid, category='other'):
        nonlocal saved_count
        if not keep_list:
            return 0
        to_save = keep_list[:per_video_cap]
        path = save_raw_batch(to_save, f'filtered_{vid}_{category}', method='youtube_api_filtered')
        saved_count += len(to_save)
        return len(to_save)

    # first process seed videos
    for vid in seed_videos:
        if saved_count >= target_code_mixed:
            break
        try:
            recs = scrape_video_comments([vid], api_key=api_key, max_results=per_video_limit)
        except Exception as e:
            print('seed fetch failed', vid, e)
            continue
        keep = [r for r in recs if contains_roman_hindi(r.get('text',''))]
        if keep:
            n = save_limited(keep, vid, category='seed')
            print('Saved filtered', n, 'from seed', vid)
        time.sleep(0.5)

    # then search queries
    for q in queries:
        if saved_count >= target_code_mixed:
            break
        try:
            vids = search_videos(api_key, q, max_results=10)
        except Exception as e:
            print('search failed', q, e)
            continue
        for vid, title in vids:
            if saved_count >= target_code_mixed:
                break
            try:
                recs = scrape_video_comments([vid], api_key=api_key, max_results=per_video_limit)
            except Exception as e:
                print('fetch failed', vid, e)
                continue
            keep = [r for r in recs if contains_roman_hindi(r.get('text',''))]
            if keep:
                # determine category based on query text
                lq = q.lower()
                if any(x in lq for x in ['lecture','jee','neet','upsc','physics','study','class','khan','wallah','alakh']):
                    cat = 'educational'
                elif any(x in lq for x in ['vlog','vlog india','carryminati','technical guruji','bb ki vines','ashish','slayy']):
                    cat = 'big_creator'
                else:
                    cat = 'other'
                n = save_limited(keep, vid, category=cat)
                print('Saved filtered', n, 'from', vid, 'category', cat)
            time.sleep(0.5)

    # log methodology note
    meta = {'action':'filtered_collection','filter':'roman_hindi_marker','target_code_mixed': target_code_mixed, 'achieved': saved_count, 'date': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}
    with open(LOG_PATH, 'a', encoding='utf-8') as lf:
        lf.write(json.dumps(meta, ensure_ascii=False) + '\n')
    return saved_count

if __name__ == '__main__':
    api_key = read_env('YOUTUBE_API_KEY')
    if not api_key:
        print('YOUTUBE_API_KEY missing in .env')
        raise SystemExit(1)

    print('Creating unbiased holdout (N=400)')
    n = reservoir_sample_holdout(400)
    print('Holdout size', n)
    print('Archiving active raw files to', ARCHIVE_DIR)
    moved = archive_active_raw()
    print('Moved', len(moved), 'files to archive')
    print('Starting filtered collection to reach 5000 code-mixed comments')
    saved = filtered_collect(api_key, target_code_mixed=5000)
    print('Filtered collection saved code-mixed count:', saved)

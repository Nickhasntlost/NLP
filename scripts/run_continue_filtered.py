import os
import json
from datetime import datetime

from qa_spotcheck import classify_text
from filtered_collect import filtered_collect

ROOT = os.path.join(os.path.dirname(__file__), '..')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
HOLDOUT_PATH = os.path.join(RAW_DIR, 'holdout_unbiased_sample.jsonl')

def current_code_mixed_count():
    # scan existing filtered files to get current code-mixed count
    total = 0
    for fname in os.listdir(RAW_DIR):
        if not fname.startswith('raw_filtered_') and not fname.startswith('filtered_'):
            continue
        path = os.path.join(RAW_DIR, fname)
        if os.path.isdir(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                cat = classify_text(rec.get('text',''))
                if cat == 'code-mixed':
                    total += 1
    return total

def main():
    api_key = os.environ.get('YOUTUBE_API_KEY')
    if not api_key:
        # try .env
        env_path = os.path.join(ROOT, '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('YOUTUBE_API_KEY='):
                        api_key = line.split('=',1)[1].strip()
                        break
    if not api_key:
        print('Missing YOUTUBE_API_KEY; aborting')
        return

    current = current_code_mixed_count()
    target = 5000
    remaining = max(0, target - current)
    print('Current code-mixed:', current, 'remaining to target', remaining)
    if remaining == 0:
        print('Target already reached; nothing to do')
        return

    # New seeds and queries per user request
    educational_queries = ['Physics Wallah lecture','Khan Sir class','JEE NEET Hindi English','UPSC Hindi English mix','Alakh Pandey lecture']
    big_creator_queries = ['CarryMinati video','BB Ki Vines','Ashish Chanchlani','Technical Guruji review','Slayy Point vlog','vlog India Hinglish']

    # Run filtered collect with per-video cap 200 (honors ~150-200 guideline)
    achieved = filtered_collect(api_key, target_code_mixed=remaining, per_video_limit=500, per_video_cap=200, seed_videos=None, queries=educational_queries+big_creator_queries)

    print('Filtered collection run finished. New code-mixed saved:', achieved)

if __name__ == '__main__':
    main()

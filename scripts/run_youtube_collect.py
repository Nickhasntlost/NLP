"""Discover Hindi-medium videos and collect comments until target volume is reached.

Reads YOUTUBE_API_KEY from .env in repo root. Produces raw JSONL files per-video
in data/raw/ and logs metadata to logs/datasets_log.jsonl via save_raw_batch().
"""
import os
import sys
import time
from scraper.youtube_scraper import search_videos, scrape_video_comments, save_raw_batch

ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')

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

def main(target_min=2000, per_video_limit=500):
    api_key = read_env('YOUTUBE_API_KEY')
    if not api_key:
        print('YOUTUBE_API_KEY not found in .env or environment. Aborting.')
        sys.exit(1)

    queries = ['Hindi vlog', 'Hindi comedy', 'Hindi reaction', 'Hindi news', 'Bollywood', 'Hindi commentary', 'Hindi standup', 'Hindi vlog 2025']
    candidates = []
    # discover ~2-3 videos per query
    for q in queries:
        try:
            res = search_videos(api_key, q, max_results=3)
            for vid, title in res:
                candidates.append((vid, title, q))
            time.sleep(1)
        except Exception as e:
            print('Search failed for', q, e)

    # deduplicate preserving order
    seen = set()
    videos = []
    for vid, title, q in candidates:
        if vid not in seen:
            seen.add(vid)
            videos.append((vid, title, q))

    print('Discovered', len(videos), 'candidate videos. Will fetch comments per video up to', per_video_limit)

    total = 0
    chosen = []
    for vid, title, q in videos:
        print('Fetching comments for', vid, '-', title)
        try:
            records = scrape_video_comments([vid], api_key=api_key, max_results=per_video_limit)
        except Exception as e:
            print('Failed to fetch comments for', vid, 'error:', e)
            continue
        if not records:
            print('No comments returned or comments disabled for', vid)
            continue
        path = save_raw_batch(records, f'youtube_{vid}', method='youtube_api')
        print('Saved', len(records), 'comments to', path)
        chosen.append({'video_id': vid, 'title': title, 'query': q, 'collected': len(records), 'file': path})
        total += len(records)
        if total >= target_min:
            break
        # polite pause to avoid quota bursts
        time.sleep(1)

    print('Total comments collected:', total)
    print('Chosen videos:')
    for c in chosen:
        print('-', c['video_id'], c['collected'], c['title'])

if __name__ == '__main__':
    main()

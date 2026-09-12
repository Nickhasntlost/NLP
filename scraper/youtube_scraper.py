import os
import json
import hashlib
import re
from datetime import datetime
from dateutil import tz

LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'datasets_log.jsonl')

PII_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\+?\d[\d\-\s]{7,}\d"),
]

def _hash_pii(match: re.Match) -> str:
    token = match.group(0)
    return "__PII_" + hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]

def scrub_pii(text: str) -> str:
    for patt in PII_PATTERNS:
        text = patt.sub(_hash_pii, text)
    return text

def log_batch(meta: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def save_raw_batch(records, source_name, method='youtube_api'):
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(base, exist_ok=True)
    timestamp = datetime.now(tz.tzutc()).strftime('%Y%m%dT%H%M%SZ')
    filename = f"raw_{source_name}_{timestamp}.jsonl"
    path = os.path.join(base, filename)
    count = 0
    with open(path, 'w', encoding='utf-8') as fw:
        for r in records:
            rec = dict(r)
            if 'text' in rec and isinstance(rec['text'], str):
                rec['text'] = scrub_pii(rec['text'])
            if 'author' in rec and isinstance(rec['author'], str):
                rec['author_hashed'] = hashlib.sha256(rec['author'].encode('utf-8')).hexdigest()
                rec.pop('author', None)
            fw.write(json.dumps(rec, ensure_ascii=False) + '\n')
            count += 1

    meta = {
        'source': source_name,
        'method': method,
        'date_collected': timestamp,
        'file': os.path.relpath(path, os.path.join(os.path.dirname(__file__), '..')),
        'size': count,
        'license': 'platform_terms',
    }
    log_batch(meta)
    return path

def scrape_video_comments(video_ids, api_key=None, max_results=100):
    """Scrape comments for a list of YouTube video IDs using the official Data API.

    Returns list of records: {source, text, timestamp, source_id, author}
    """
    if not api_key:
        raise RuntimeError('YouTube API key missing; provide api_key')

    try:
        from googleapiclient.discovery import build
    except Exception:
        raise RuntimeError('google-api-python-client not installed')

    youtube = build('youtube', 'v3', developerKey=api_key)
    records = []
    for vid in video_ids:
        collected = 0
        next_page_token = None
        while True:
            to_fetch = min(100, max_results - collected)
            if to_fetch <= 0:
                break
            req = youtube.commentThreads().list(part='snippet', videoId=vid, textFormat='plainText', maxResults=to_fetch, pageToken=next_page_token)
            resp = req.execute()
            for item in resp.get('items', []):
                snip = item['snippet']['topLevelComment']['snippet']
                rec = {'source': f'youtube/{vid}', 'text': snip.get('textDisplay',''), 'timestamp': snip.get('publishedAt',''), 'source_id': item.get('id'), 'author': snip.get('authorDisplayName','')}
                records.append(rec)
                collected += 1
            next_page_token = resp.get('nextPageToken')
            if not next_page_token:
                break

    return records

def search_videos(api_key, query, max_results=5, regionCode='IN'):
    """Search YouTube for videos matching a query. Returns list of (videoId, title).

    Uses the Search.list endpoint and returns top results by relevance.
    """
    try:
        from googleapiclient.discovery import build
    except Exception:
        raise RuntimeError('google-api-python-client not installed')

    youtube = build('youtube', 'v3', developerKey=api_key)
    req = youtube.search().list(part='snippet', q=query, maxResults=max_results, type='video', regionCode=regionCode)
    resp = req.execute()
    out = []
    for item in resp.get('items', []):
        vid = item['id']['videoId']
        title = item['snippet'].get('title','')
        out.append((vid, title))
    return out

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--videos', nargs='+', required=True)
    p.add_argument('--api_key')
    args = p.parse_args()
    recs = scrape_video_comments(args.videos, api_key=args.api_key)
    path = save_raw_batch(recs, 'youtube')
    print('Saved', path)

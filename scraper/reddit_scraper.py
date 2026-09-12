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
    """Replace obvious PII (emails, phone-like) with hashed placeholders."""
    for patt in PII_PATTERNS:
        text = patt.sub(_hash_pii, text)
    return text

def log_batch(meta: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def save_raw_batch(records, source_name, method='praw'):
    """Save raw records to data/raw with PII hashed in-place and log metadata."""
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(base, exist_ok=True)
    timestamp = datetime.now(tz.tzutc()).strftime('%Y%m%dT%H%M%SZ')
    filename = f"raw_{source_name}_{timestamp}.jsonl"
    path = os.path.join(base, filename)
    count = 0
    with open(path, 'w', encoding='utf-8') as fw:
        for r in records:
            # Copy but hash PII fields in text and author
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

def scrape_subreddit(subreddits, limit=100, client_id=None, client_secret=None, user_agent='beyond-words-scraper/0.1'):
    """Scrape comments/posts from given subreddits using PRAW. Requires credentials.

    Returns list of records: {source, text, timestamp, source_id, author}
    """
    try:
        import praw
    except Exception:
        raise RuntimeError('praw is not installed; ensure requirements.txt is installed')

    if not client_id or not client_secret:
        raise RuntimeError('Reddit credentials missing: provide client_id and client_secret')

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    records = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.hot(limit=limit):
            # Collect post title and top-level comments (lightweight)
            ts = datetime.utcfromtimestamp(post.created_utc).isoformat() + 'Z'
            records.append({'source': f'reddit/{sub}', 'text': post.title, 'timestamp': ts, 'source_id': post.id, 'author': str(post.author) if post.author else ''})
            post.comments.replace_more(limit=0)
            for c in post.comments.list():
                records.append({'source': f'reddit/{sub}', 'text': c.body, 'timestamp': datetime.utcfromtimestamp(c.created_utc).isoformat() + 'Z', 'source_id': c.id, 'author': str(c.author) if c.author else ''})

    return records

if __name__ == '__main__':
    # quick local test scaffold (will error if creds missing)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--subs', nargs='+', default=['r/india'])
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--client_id')
    p.add_argument('--client_secret')
    args = p.parse_args()
    recs = scrape_subreddit(args.subs, limit=args.limit, client_id=args.client_id, client_secret=args.client_secret)
    path = save_raw_batch(recs, 'reddit')
    print('Saved', path)

import os
import json
from datetime import datetime

ROOT = os.path.join(os.path.dirname(__file__), '..')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
ARCHIVE_DIR = os.path.join(RAW_DIR, 'raw_archive_phase1a')
os.makedirs(ARCHIVE_DIR, exist_ok=True)

CAP = 200

def load_all_filtered():
    records_by_vid = {}
    files = [f for f in os.listdir(RAW_DIR) if f.startswith('raw_filtered_') and f.endswith('.jsonl')]
    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                src = rec.get('source','')
                vid = src.split('/')[-1] if src else fname
                records_by_vid.setdefault(vid, []).append((rec, fname))
    return records_by_vid

def trim_and_archive(target_videos):
    records_by_vid = load_all_filtered()
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    moved_total = 0
    for vid in target_videos:
        recs = records_by_vid.get(vid, [])
        total = len(recs)
        if total <= CAP:
            continue
        keep = recs[:CAP]
        excess = recs[CAP:]

        # write kept records into a new trimmed file
        trimmed_fname = f'raw_filtered_{vid}_trimmed_{timestamp}.jsonl'
        trimmed_path = os.path.join(RAW_DIR, trimmed_fname)
        with open(trimmed_path, 'w', encoding='utf-8') as out:
            for rec, srcfname in keep:
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')

        # write excess to archive file
        archive_fname = f'raw_excess_{vid}_{timestamp}.jsonl'
        archive_path = os.path.join(ARCHIVE_DIR, archive_fname)
        with open(archive_path, 'w', encoding='utf-8') as out:
            for rec, srcfname in excess:
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        moved_total += len(excess)

        # remove original files for this vid to avoid duplicates
        # find original files that contained this vid
        original_files = set(srcfname for rec, srcfname in recs)
        for of in original_files:
            try:
                os.remove(os.path.join(RAW_DIR, of))
            except Exception:
                pass

    return moved_total

if __name__ == '__main__':
    records_by_vid = load_all_filtered()
    targets = [vid for vid, recs in records_by_vid.items() if len(recs) > CAP]
    if not targets:
        print('No videos exceed cap')
    else:
        moved = trim_and_archive(targets)
        print('Moved excess records to archive:', moved)

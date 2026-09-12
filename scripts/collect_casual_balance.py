import os
import json
from filtered_collect import filtered_collect
from report_filtered_stats import scan

ROOT = os.path.join(os.path.dirname(__file__), '..')

def main():
    report = scan()
    edu = report['category_counts'].get('educational', 0)
    other = report['category_counts'].get('other', 0)
    seed = report['category_counts'].get('seed', 0)
    print('Current educational:', edu, 'casual/other:', other, 'seed:', seed)

    # target to make casual >= educational
    need = max(0, edu - other)
    # safety cap: don't collect more than 3000 additional
    max_collect = 3000
    target = min(need + 50, max_collect)
    if target <= 0:
        print('Already balanced')
        return

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

    big_creator_queries = ['CarryMinati video','BB Ki Vines','Ashish Chanchlani','Technical Guruji review','Slayy Point vlog','vlog India Hinglish','vlog India','Hinglish comedy','Indian vlog Hinglish']

    print('Collecting up to', target, 'casual code-mixed comments using big-creator queries')
    achieved = filtered_collect(api_key, target_code_mixed=target, per_video_limit=500, per_video_cap=200, seed_videos=None, queries=big_creator_queries)
    print('Achieved casual collect:', achieved)

if __name__ == '__main__':
    main()

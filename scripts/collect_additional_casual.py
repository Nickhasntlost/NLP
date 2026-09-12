import os
from filtered_collect import filtered_collect
from report_filtered_stats import scan

ROOT = os.path.join(os.path.dirname(__file__), '..')

def main():
    report = scan()
    edu = report['category_counts'].get('educational', 0)
    big = report['category_counts'].get('big_creator', 0)
    other = report['category_counts'].get('other', 0)
    casual = big + other
    need = max(0, edu - casual)
    print('edu', edu, 'casual', casual, 'need', need)
    if need <= 0:
        print('No additional casual needed')
        return

    target = need + 20
    api_key = os.environ.get('YOUTUBE_API_KEY')
    if not api_key:
        env_path = os.path.join(ROOT, '.env')
        if os.path.exists(env_path):
            with open(env_path,'r',encoding='utf-8') as f:
                for line in f:
                    if line.startswith('YOUTUBE_API_KEY='):
                        api_key = line.split('=',1)[1].strip()
                        break
    if not api_key:
        print('Missing API key')
        return

    big_creator_queries = ['CarryMinati video','BB Ki Vines','Ashish Chanchlani','Technical Guruji review','Slayy Point vlog','vlog India Hinglish','vlog India','Hinglish comedy','Indian vlog Hinglish']
    achieved = filtered_collect(api_key, target_code_mixed=target, per_video_limit=500, per_video_cap=200, seed_videos=[], queries=big_creator_queries)
    print('Achieved additional casual:', achieved)

if __name__ == '__main__':
    main()

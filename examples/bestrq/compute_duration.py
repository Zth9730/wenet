import json
from tqdm import tqdm
files = 'bilibili_video_2306.jsonl'

files2 = 'douyin_video_2306.jsonl'

ids_set = set()
times = 0
with open(files, 'r') as f1, open(files2, 'r') as f2:
    for l in tqdm(f1):
        l = l.strip()
        data = json.loads(l)
        ids = data['bvid']
        if ids not in ids_set:
            times += data['duration']
            ids_set.add(ids)
    # for l in tqdm(f2):
    #     l = l.strip()
    #     data = json.loads(l)
    #     ids = data['post_id']
    #     if ids not in ids_set:
    #         times += data['duration']
    #         ids_set.add(ids)
            
print(len(ids_set))
print(times/3600)
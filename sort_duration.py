import torchaudio
from Client import MyClient
from tqdm import tqdm
import io
from tqdm.contrib.concurrent import process_map


def process(l_list):
    client = MyClient()
    xx = []
    for l in tqdm(l_list):
        l = l.strip()
        audio_bytes = client.get(l)
        with io.BytesIO(audio_bytes) as fobj:
            y, sr = torchaudio.load(fobj)
        audio_length = y.shape[1]

        xx.append([l, audio_length])
    return xx

import os
import multiprocessing as mp


ids_dict = {}
data_list = []
with open('douyin_tts_shuf.txt', 'r') as f1:
    for l in tqdm(f1):
        data_list.append(l)

list_size = len(data_list)
part_size = list_size // 128
parts = [data_list[i:i+part_size] for i in range(0, list_size, part_size)]

pool = mp.Pool(processes=128)
results = pool.map(process, parts)
# results = process_map(process, data_list, max_workers=128)

for xx in results:
    for x in xx:
        ids_dict[x[0]] = x[1]

sorted_dict = dict(sorted(ids_dict.items(), key=lambda item: item[1]))
with open('sorted_douyin_tts.txt', 'w') as f:
    for key, value in sorted_dict.items():
        f.write(key + '\n')
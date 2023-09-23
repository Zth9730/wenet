from petrel_client.client import Client
import torch
import io
import os
from tqdm import tqdm
import subprocess
import sys
import shutil
import argparse
import logging
import librosa
import multiprocessing as mp

class MyClient(object):
	def __init__(self):
		self.client = Client("~/petreloss.conf")
	def get(self, key, enable_stream=False):
		index = key.find("/")
		bucket = key[:index]
		key = key[index+1:]
		if bucket == "asr":
			return self.client.get("asr:s3://{}/".format(bucket) + key, no_cache=True, enable_stream=enable_stream)
		elif bucket == "youtubeBucket":
			return self.client.get("youtube:s3://{}/".format(bucket) + key, no_cache=True, enable_stream=enable_stream)
	def get_file_iterator(self, key):
		index = key.find("/")
		bucket = key[:index]
		key = key[index+1:]
		if bucket == "asr":
				return self.client.get_file_iterator("s3://asr/" + key)
		elif bucket == "youtubeBucket":
				return self.client.get_file_iterator("youtube:s3://{}/".format(bucket) + key)
	def put(self, uri, content):
		return self.client.put(uri, content)



def process(wav_path):
    client = MyClient()
    wave_bytes = client.get(wav_path)
    with io.BytesIO(wave_bytes) as fobj:
        y, sr = librosa.load(fobj)
    lens = y.shape[0]/16000
    return lens

datalists = []
with open('for_dis.txt', 'r') as f:
    for l in f:
        datalists.append(l.strip())
job_num = 128
pool = mp.Pool(processes=job_num)
results = pool.map(process, datalists)
print(sum(results))


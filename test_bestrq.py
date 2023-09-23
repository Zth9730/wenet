import torch
import torchaudio
from wenet.ssl.bestrq.bestqr_model import BestRQModel
from wenet.transformer.encoder import ConformerEncoder
import torchaudio.compliance.kaldi as kaldi

import numpy as np
import random

import logging
import librosa
import soundfile as sf
import copy
import json
import random
import re
import gc
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse
import io
from petrel_client.client import Client

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

# lr = 0.01
# warmup_steps = 10000
# step_num=10000
# print(lr * warmup_steps ** 0.5 * min(step_num ** -0.5, step_num * warmup_steps ** -1.5))
# exit()


AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

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
        elif bucket == "exp":
            return self.client.get("asr:s3://{}/".format(bucket) + key, no_cache=True, enable_stream=enable_stream)
    def get_file_iterator(self, key):
        index = key.find("/")
        bucket = key[:index]
        key = key[index+1:]
        if bucket == "asr":
            return self.client.get_file_iterator("s3://asr/" + key)
        elif bucket == "youtubeBucket":
            return self.client.get_file_iterator("youtube:s3://{}/".format(bucket) + key)
        
client = MyClient()
        
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(777)

encoder = ConformerEncoder(input_size=80,
                        output_size=512,    # dimension of attention
                        attention_heads=8,
                        linear_units=2048,  # the number of units of position-wise feed forward
                        num_blocks=16,      # the number of encoder blocks
                        dropout_rate=0.1,
                        positional_dropout_rate=0.1,
                        attention_dropout_rate=0.1,
                        input_layer='conv2d', # encoder input type, you can chose conv2d, conv2d6 and conv2d8
                        normalize_before=True,
                        cnn_module_kernel=15,
                        use_cnn_module=True,
                        cnn_module_norm='layer_norm',
                        activation_type='swish',
                        pos_enc_layer_type='rel_pos',
                        selfattention_layer_type='rel_selfattn')

model = BestRQModel(encoder,
                    embedding_dim=16, 
                    num_mel_bins=80, 
                    num_embeddings=8192, 
                    mask_prob=0.01,
                    mask_length=15,
                    num_codebooks=1,
                    min_masks=10,
                    dynamic_mask_length=True)

model.eval()
model_path = '/mnt/petrelfs/zhoudinghao/work/thzhang/wenet/examples/bestrq/exp/success_20230809_lt80_dynamic_continue_3/ckpts/272000.pt'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint, strict=True)
# print(model)
with open('/mnt/petrelfs/zhoudinghao/work/thzhang/wenet/examples/bestrq/data/zh_en_20230720/test.list', 'r') as f, open('test.result', 'w') as f2:
    for line in f:
        line = line.strip()
        wav_bytes = client.get(line)
        with io.BytesIO(wav_bytes) as fobj:
            y, sr = torchaudio.load(fobj)
        inputs = y * (1 << 15)
        fbank =  kaldi.fbank(inputs,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            dither=0.0,
                            energy_floor=0.0,
                            sample_frequency=sr)
        inputs = fbank.unsqueeze(0)
        length = torch.tensor([fbank.shape[0]], dtype=torch.int32)
        loss_dict, _ = model(inputs, length)
        info = 'ids: {}, loss: {}, acc: {}'.format(line, loss_dict['loss'], loss_dict['codes_acc'])
        print(info)
        f2.write(info + '\n')


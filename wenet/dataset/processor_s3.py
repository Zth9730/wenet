# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
# from petrel_client.client import Client

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

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

def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    client = MyClient()
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            if "exp" in url or "asr" in url:
#                url = "s3://"+url
               # stream = io.BytesIO(url)
                # stream = open(url, "rb")
                tarcontent = client.get(url)
                if tarcontent is None:
                    continue
#                    raise ValueError('Failed to download {}'.format(url))
                tarbytes = copy.deepcopy(tarcontent)
                del tarcontent
                stream = io.BytesIO(tarbytes)
                # stream = io.BytesIO(client.get(url))
            else:
                pr = urlparse(url)
                # local file
                if pr.scheme == '' or pr.scheme == 'file':
                    stream = open(url, 'rb')
                # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
                else:
                    cmd = f'curl -s -L {url}'
                    process = Popen(cmd, shell=True, stdout=PIPE)
                    sample.update(process=process)
                    stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))
#    del client

def tar_file_and_group(data, client=None, vad=None):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    client = MyClient()
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True

        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix == 'txt':
                        example['txt'] = file_obj.read().decode('utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        if client is not None:
                            s3_key = file_obj.read().decode('utf8').strip()
                            s3b = client.get(s3_key)
                            if s3b is None:
                                valid = False 
                            else:
                                wavbytes = copy.deepcopy(s3b)
                                with io.BytesIO(wavbytes) as fobj: 
                                    if vad:
                                        wav, sr = librosa.load(fobj, sr=16000)
                                        wav, _ = librosa.effects.trim(wav)
                                        with io.BytesIO() as trimWav:
                                            sf.write(trimWav, wav, samplerate=16000, format="wav")
                                            trimWav.seek(0)
                                            waveform, sample_rate = torchaudio.load(trimWav)
                                    else:
                                        waveform, sample_rate = torchaudio.load(fobj)
                                    nbytes = random.randint(0,400) 
                                    n_zeros = torch.zeros(1,nbytes*16)
                                    
                                    nbytes = random.randint(0,400) 
                                    n_haed_zeros = torch.zeros(1,nbytes*16)
                                    waveform = torch.concat([n_haed_zeros, waveform, n_zeros], 1)
                                example['wav'] = waveform
                                example['sample_rate'] = sample_rate
                                valid = True

                                del s3b
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    if s3b is None:
                        wavbytes = b'download failed'
                    logging.warning('error to parse {} {}'.format(name, postfix))
                    continue
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            if valid: 
                yield example
            example = {}
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()

def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    client = MyClient()
    for sample in data:
        segs = sample['src'].split("\t")
        if len(segs) < 2:
            continue
        key, txt = segs[0], segs[1]
        wav_file = key
        try:
            tarcontent = client.get(key)
            tarbytes = copy.deepcopy(tarcontent)
            with io.BytesIO(tarbytes) as fobj: 
                waveform, sample_rate = torchaudio.load(fobj)
            # waveform = torch.concat([waveform, torch.zeros(1,300*16)], 1)
            example = dict(key=key,
                               txt=txt,
                               wav=waveform,
                               sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))
# zjm add
def to_one_hot(data,category_dict):
    # convert category label to one hot.
    for sample in data:
        assert 'label' in sample
        label= sample['label']
        sample['label'] = category_dict[label]
        yield sample
# zjm add
def parse_raw_classification(data, category_dict):
    """ Parse key/wav/category from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/category

        Returns:
            Iterable[{key, wav, category, sample_rate}]
    """
    # client = MyClient()
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        category = obj['txt']
        #wav_file = key
        try:
            # tarcontent = client.get(key)
            # tarbytes = copy.deepcopy(tarcontent)
            # with io.BytesIO(tarbytes) as fobj: 
            #     waveform, sample_rate = torchaudio.load(fobj)
            waveform, sample_rate = torchaudio.load(wav_file)
            # waveform = torch.concat([waveform, torch.zeros(1,300*16)], 1)
            label = category_dict[category] # eg. convert 'happy' to '2'
            one_hot = torch.nn.functional.one_hot(torch.tensor(label).long(), num_classes=len(category_dict))
            example = dict( key=key,
                            label=one_hot, 
                            wav=waveform,
                            sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))

def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second\
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            steps = int(max_length*sample['sample_rate'] / 100)
            for idx in range(0, len(sample['wav']), steps):
                cut_wav = sample.copy()
                cut_wav['wav'] = sample['wav'][:,idx:idx+steps]
                yield cut_wav
            continue
        # if len(sample['label']) < token_min_length:
        #     continue
        # if len(sample['label']) > token_max_length:
        #     continue
        # if num_frames != 0:
        #     if len(sample['label']) / num_frames < min_output_input_ratio:
        #         continue
        #     if len(sample['label']) / num_frames > max_output_input_ratio:
        #         continue
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        #speeds = [0.9, 1.0, 1.1]
        speeds = [0.9, 1.0, 1.1]
        # speeds = [0.8,0.9, 1.0, 1.1,1.2,1.3,1.4,1.5]
        #speeds = [1.3, 1.3, 1.3]
        #speeds = [1.0,1.0,1.0]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if "long_speed_audio_1.5" in sample['key'] or "short_speed_audio_1.3" in sample['key']: 
            speed = 1.0

        if "longaudio" in sample['key']: 
            speed = random.choice([0.9,1.0,1.1])
        if "ASR/manual/" in sample['key']:
            speed = random.choice([1.0,1.2,1.3,1.5])

        #speed_ways = ['tempo', 'tempo']
        speed_ways = ['speed', 'speed']
        speed_way = speed_ways[random.choice([0,1])]
        if speed != 1.0:
            if False : 
            #if waveform.size(1) < 16000 or len(sample["label"]) < 5:
                if random.choice([0,1]):
                    speed = 1.1 + 0.5 * (random.randrange(0,11,1)/10)
                    wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform, sample_rate,
                        [[speed_way, f'{speed:.5f}'], ['rate', str(sample_rate)]])
                    sample['wav'] = wav
                else:
                    wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform, sample_rate,
                        [[speed_way, str(speed)], ['rate', str(sample_rate)]])
                    sample['wav'] = wav
            
            else:
                wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, sample_rate,
                    [[speed_way, str(speed)], ['rate', str(sample_rate)]])
                sample['wav'] = wav

        yield sample


def compute_fbank(data,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0,
                  infer=False):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        infer = True
        if not infer:
            n_zeros_frams = random.randint(0,30)
            n_zeros_frams = 20
            if n_zeros_frams:
                apped_n_frames = torch.zeros(n_zeros_frams, num_mel_bins)
                mat = torch.concat([mat, apped_n_frames], 0)

        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def tokenize_space(data, symbol_table):
    '''
        eg : text: 你 好 中 国 _h e l l o _w o r l d
                   你 好 中 国 _he llo _wo rld 
    '''

    for sample in data:
        assert 'txt' in sample
        txt = sample['txt']
        label = []
        tokens = txt.split(" ")
        # if '<unk>' in txt:
        #     continue
        for ch in tokens:
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table['<unk>'])
        sample['tokens'] = tokens
        sample['label'] = label
        yield sample

def tokenize(data, symbol_table, bpe_model=None):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    # TODO(Binbin Zhang): Support BPE
    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    for sample in data:
        assert 'txt' in sample
        txt = sample['txt']
        label = []
        tokens = []
        if bpe_model is not None:
            txt = bpe_preprocess(txt)
            mix_chars = seg_char(txt)
            for j in mix_chars:
                for k in j.strip().split("▁"):
                    if not k.encode('UTF-8').isalpha():
                        tokens.append(k)
                    else:
                        for l in sp.encode_as_pieces(k):
                            tokens.append(l)
        else:
            for ch in txt:
                if ch == ' ':
                    ch = "▁"
                tokens.append(ch)

        for ch in tokens:
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table['<unk>'])

        sample['tokens'] = tokens
        sample['label'] = label
        yield sample


def bpe_preprocess(text):
    """ Use ▁ for blank among english words
        Warning: it is "▁" symbol, not "_" symbol
    """
    text = text.upper()
    text = re.sub(r'([A-Z])[ ]([A-Z])', r'\1▁\2', text)
    text = re.sub(r'([A-Z])[ ]([A-Z])', r'\1▁\2', text)
    text = text.replace(' ', '')
    text = text.replace('\xEF\xBB\xBF', '')
    return text


def seg_char(text):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(text)
    chars = [w for w in chars if len(w.strip()) > 0]
    return chars

def _get_wav_path(wav_path, raw_bytes=None):
    """
    Returns: file like
    """
    if raw_bytes:
        pad_zero = [0,320, 640]
        weights  = [1,1,1]
        # for 16k 16bit
        padding = 32 * random.choices(pad_zero, weights, k=1)[0]
        raw_bytes = bytes(bytearray(raw_bytes) +int(padding) * bytearray([0x00]))
        return io.BytesIO(raw_bytes)
    return wav_path

def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=100000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for i, sample in enumerate(data):
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(feats_length, descending=True)
        feats_lengths = torch.tensor(
            [sample[i]['feat'].size(0) for i in order], dtype=torch.int32)
        sorted_feats = [sample[i]['feat'] for i in order]
        sorted_keys = [sample[i]['key'] for i in order]
        sorted_labels = [
            torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
        ]
        label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                     dtype=torch.int32)

        padded_feats = pad_sequence(sorted_feats,
                                    batch_first=True,
                                    padding_value=0)
        padding_labels = pad_sequence(sorted_labels,
                                      batch_first=True,
                                      padding_value=-1)
        yield (sorted_keys, padded_feats, padding_labels, feats_lengths,
               label_lengths)

def text_read(data):
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            if "ASR" in url or "ASR2" in url or "zth" in url or "speech_annotations" in url or  "speech_manual_annotations" in url or  "speech_pre_annotations" in url:
                s3b = client.get(url)
                wavbytes = copy.deepcopy(s3b)
                with io.BytesIO(wavbytes) as fobj: 
                    fobj = fobj.read().decode('utf8')
                    tmp = ''
                    for i in fobj:
                        if i != '\n':
                            tmp += i
                        else:
                            print(tmp)
                            tmp = ''
            else:
                pr = urlparse(url)
                # local file
                if pr.scheme == '' or pr.scheme == 'file':
                    stream = open(url, 'rb')
                # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
                else:
                    cmd = f'curl -s -L {url}'
                    process = Popen(cmd, shell=True, stdout=PIPE)
                    sample.update(process=process)
                    stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def read_txt_file_list(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    client = MyClient()
    for sample in data:
        # sample['text'] = sample['src']
        # yield sample
        assert 'src' in sample
        path = sample['src']
        try:
            s3b = client.get(path, enable_stream=True)
            streams = s3b.iter_lines()
            # TODO(Tianhao Zhang): unique process function to filter
            for line in streams:
                line = line.decode('utf-8')
                example = dict(text=line)
                yield example
        except Exception as ex:
            logging.warning('Failed to open {}'.format(path))


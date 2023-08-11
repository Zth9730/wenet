import torchvision.transforms as transforms
from PIL import Image
import logging
import json
import random
import re
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse
import numpy
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence
import glob

import sys
import os
import subprocess
import numpy
import scipy.signal
import soundfile
import librosa
from scipy import signal

def shift(xs, n):
    e = numpy.empty_like(xs)
    if n > 0:
        e[:n] = 0.0
        e[n:] = xs[:-n]
    elif n == 0:
        e = xs
    else:
        e[n:] = 0.0
        e[:n] = xs[-n:]
    return e

def audioconv(sclean, IR):
    # sclean = sclean / np.max(np.abs(sclean))
    # sclean = norm01(sclean)

    # p_max = np.argmax(np.abs(IR.cpu().numpy())) if torch.is_tensor(IR) else np.argmax(np.abs(IR))
    IR = IR.cpu().numpy() if torch.is_tensor(IR) else IR
    p_max = numpy.argmax(numpy.abs(IR))
    signal_rev = signal.fftconvolve(sclean, IR, mode="full")

    signal_rev = shift(signal_rev, -p_max)   # IR delay compensation
    # signal_rev = readdata.shift(signal_rev, -p_max)[:len(sclean)]  # IR delay compensation
    # signal_rev = signal_rev / np.max(np.abs(signal_rev))  # normalization
    # signal_rev = norm01(signal_rev)
    return signal_rev



def apply_reverb(signal: numpy.ndarray, path: str) -> numpy.ndarray:
    reverb, sr = soundfile.read(path)
    signal = numpy.concatenate((signal, numpy.zeros(reverb.shape)))
    reverb /= numpy.max(numpy.abs(reverb))
    reverb[numpy.where(numpy.abs(reverb) < 0.1)] = 0
    mix = 0.01
    signal += scipy.signal.oaconvolve(signal, reverb, "full")[:len(signal)] * mix
    return signal


train_file = '/home/zth/work/new/wenet/examples/aishell/s0/data/train/wav.scp'
val_file = '/home/zth/work/new/wenet/examples/aishell/s0/data/dev/wav.scp'
test_file = '/home/zth/work/new/wenet/examples/aishell/s0/data/test/wav.scp'

files = [val_file, test_file]
for old_file in files:
    new_file = old_file.replace('wav.scp', 'new_reverse_wav.scp')
    with open(new_file, 'w') as f:
        with open(old_file, 'r') as f1:
            for line in f1.readlines():
                wav_file = line.strip().split(' ')[1]
                waveform, sample_rate = librosa.load(wav_file)
                XX = wav_file.split('/')[4]
                # if XX == 'dev':
                #     XX = 'val'
                pattern = "/home/zth/work/new/image2reverb/new_{}.txt".format(XX) # (or "*.*")
                reverse_wavs = open(pattern, 'r').readlines()
                reverb_file = random.choice(reverse_wavs).strip()
                image_file = reverb_file.replace('B', 'A').replace('img.wav', 'label.jpg')
                reverb, y_sr = soundfile.read(reverb_file)
                waveform = audioconv(waveform, reverb)
                # waveform /= numpy.max(numpy.abs(waveform))
                new_file = wav_file.replace('data_aishell', 'new_reverse_data_aishell')
                path = os.path.join('/',*new_file.split('/')[:-1])
                if not os.path.exists(path):
                    os.makedirs(path)
                soundfile.write(new_file, waveform, 16000)
                f.write(line.strip() + ' ' + image_file + '\n')

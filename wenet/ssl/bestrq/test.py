import torch
import torchaudio

from bestqr_model import BestRQModel

model = BestRQModel(embedding_dim=16, 
                    num_mel_bins=80, 
                    num_embeddings=8192, 
                    mask_prob=0.01,
                    mask_length=40,
                    num_codebooks=1,
                    min_masks=10)

inputs, sr = torchaudio.load('/mnt/petrelfs/zhoudinghao/work/thzhang/wenet/examples/bestrq/Y0000006402_GCkNpWodZTg_S00009.wav')
print(model(inputs))


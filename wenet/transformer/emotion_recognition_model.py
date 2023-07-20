# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch
from wenet.transformer.encoder import TransformerEncoder
import logging
logging.getLogger().setLevel(logging.INFO)

class EmotionRecognitionModel(torch.nn.Module):
    def __init__(
        self,
        category_size: int,
        encoder: TransformerEncoder,
        input_ln=False,
        **kwargs
    ):
        super().__init__()
        self.category_size = category_size
        self.encoder = encoder
        self.linear = torch.nn.Linear(self.encoder.output_size(), self.category_size)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.input_ln = input_ln
        if self.input_ln:
            self.input_ln = torch.nn.LayerNorm(
                80, eps=1e-5, elementwise_affine=False
            )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        label: torch.Tensor,
        args,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            label: (Batch, Length)
        """
        logging.info(label)
        if self.input_ln:
            assert self.encoder.global_cmvn is None
            speech = self.input_ln(speech)
            
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        output = self.linear(encoder_out)
        loss = self.criterion(output,label)
        return {"loss": loss}

    def inference(self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,):
        
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        output = self.linear(encoder_out)
        softmax_output = torch.nn.functional.softmax(output,dim=-1)
        output, confidence = torch.argmax(softmax_output,dim = -1)
        return output, confidence
    
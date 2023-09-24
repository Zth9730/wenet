# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
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

import torch
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import (ConvPredictor, EmbeddingPredictor,
                                        RNNPredictor)
from wenet.transducer.transducer import Transducer
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.multimodal_decoder import SemanticMultimodalDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.branchformer.encoder import BranchformerEncoder
from wenet.squeezeformer.encoder import SqueezeformerEncoder
from wenet.efficient_conformer.encoder import EfficientConformerEncoder
from wenet.paraformer.paraformer import Paraformer
from wenet.cif.predictor import Predictor
from wenet.utils.cmvn import load_cmvn
import torch.nn as nn
from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

def init_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    if configs.get('whisper', False):
        model = WhisperForConditionalGeneration.from_pretrained(confgs['whisper_conf']['base_model'],
                                                        load_in_8bit=confgs['whisper_conf']['use_8bit'],
                                                        device_map=confgs['whisper_conf']['device_map'],
                                                        local_files_only=confgs['whisper_conf']['local_files_only'])
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
                # 量化模型
        model = prepare_model_for_kbit_training(model)
        # 注册forward，否则多卡训练会失败
        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
        if confgs['whisper_conf'].get('use_lora', True):
            target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
            if confgs['whisper_conf']['use_adalora']:
                config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                                    lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules)
            else:
                config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none")
        model = get_peft_model(model, config)
        return model
    if encoder_type == 'conformer':
        print(configs['encoder_conf'])
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   vocab_size= vocab_size,
                                   **configs['encoder_conf'])
    elif encoder_type == 'squeezeformer':
        encoder = SqueezeformerEncoder(input_dim,
                                       global_cmvn=global_cmvn,
                                       vocab_size= vocab_size,
                                       **configs['encoder_conf'])
    elif encoder_type == 'efficientConformer':
        encoder = EfficientConformerEncoder(input_dim,
                                            global_cmvn=global_cmvn,
                                            **configs['encoder_conf'],
                                            **configs['encoder_conf']
                                            ['efficient_conf']
                                            if 'efficient_conf' in
                                               configs['encoder_conf'] else {})
    elif encoder_type == 'branchformer':
        encoder = BranchformerEncoder(input_dim,
                                      global_cmvn=global_cmvn,
                                      vocab_size= vocab_size,
                                      **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     vocab_size=vocab_size,
                                     **configs['encoder_conf'])
    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    elif decoder_type == 'semantic':
        decoder = SemanticMultimodalDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder.output_size())

    # Init joint CTC/Attention or Transducer model
    if 'predictor' in configs:
        predictor_type = configs.get('predictor', 'rnn')
        if predictor_type == 'rnn':
            predictor = RNNPredictor(vocab_size, **configs['predictor_conf'])
        elif predictor_type == 'embedding':
            predictor = EmbeddingPredictor(vocab_size,
                                           **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        elif predictor_type == 'conv':
            predictor = ConvPredictor(vocab_size, **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        else:
            raise NotImplementedError(
                "only rnn, embedding and conv type support now")
        configs['joint_conf']['enc_output_size'] = configs['encoder_conf'][
            'output_size']
        configs['joint_conf']['pred_output_size'] = configs['predictor_conf'][
            'output_size']
        joint = TransducerJoint(vocab_size, **configs['joint_conf'])
        model = Transducer(vocab_size=vocab_size,
                           blank=0,
                           predictor=predictor,
                           encoder=encoder,
                           attention_decoder=decoder,
                           joint=joint,
                           ctc=ctc,
                           **configs['model_conf'])
    elif 'paraformer' in configs:
        predictor = Predictor(**configs['cif_predictor_conf'])
        model = Paraformer(vocab_size=vocab_size,
                           encoder=encoder,
                           decoder=decoder,
                           ctc=ctc,
                           predictor=predictor,
                           **configs['model_conf'])
    else:
        model = ASRModel(vocab_size=vocab_size,
                         encoder=encoder,
                         decoder=decoder,
                         ctc=ctc,
                         lfmmi_dir=configs.get('lfmmi_dir', ''),
                         **configs['model_conf'])
    return model

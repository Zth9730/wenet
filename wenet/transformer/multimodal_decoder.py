from typing import Tuple, List, Optional

import torch
from typing import Any, List, Sequence, Tuple

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.embedding import NoPositionalEncoding
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.mask import (subsequent_mask, make_pad_mask)

import torch.nn as nn
import math


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def __init__(self, *args, layer_drop_rate=0.0):
        """Initialize MultiSequential with layer_drop.

        Args:
            layer_drop_rate (float): Probability of dropping out each fn (layer).

        """
        super(MultiSequential, self).__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        """Repeat."""
        _probs = torch.empty(len(self)).uniform_()
        for idx, m in enumerate(self):
            if not self.training or (_probs[idx] >= self.layer_drop_rate):
                args = m(*args)
        return args


def repeat(N, fn, layer_drop_rate=0.0):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.
        layer_drop_rate (float): Probability of dropping out each fn (layer).

    Returns:
        MultiSequential: Repeated model instance.

    """
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)

#### multimodal
class PositionalEncoding_Multi(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 2000):
        super(PositionalEncoding_Multi, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # -> max_len x 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
       
    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)

class SemanticEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        lnum=None):
        """Construct an EncoderLayer object."""
        super(SemanticEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = torch.nn.LayerNorm(size, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.lnum = lnum

    def forward(self, x, padding_mask, mask, l_a,cache=None, layer_num=0):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        acoustic_feature = x[:l_a,:,:]
        origin = x
        x = x[l_a:,:,:]
        residual = x

        if self.normalize_before:
            x = self.norm1(x)
        if cache is None:
            x_q = x
        else:
            cache = cache.permute(1,0,2)
            assert cache.shape == (origin.shape[0] - 1, origin.shape[1], self.size)
            x_q = x[-1:, :, :]
            residual = residual[-1:, :, :]
            mask = None if mask is None else mask[:, -1:, :]

        attn_out, attn_weight = self.self_attn(x_q, origin, origin, key_padding_mask=padding_mask, attn_mask=mask)

        # attn_weight = attn_weight.squeeze(0).permute(1,0).detach().numpy()
        # sns.heatmap(attn_weight, xticklabels=False, yticklabels=False,cbar=False)
        # plt.savefig('exp/attention_map/{}.jpg'.format(self.lnum))
        x = residual + self.dropout(
            attn_out
        )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=0)
        else:
            x = torch.concat([acoustic_feature, x], dim=0)
        return x, padding_mask, mask, l_a

class SemanticMultimodalDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        encoder_output_size=256,
        selfattention_layer_type="selfattn",
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length=11,
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        domain_proj=True,
        domain_attention=False,
        split_pe=False,
    ):
        """Construct an Decoder object."""
        nn.Module.__init__(self)
        attention_dim = encoder_output_size
        self.embed = nn.Embedding(vocab_size, attention_dim)
        self.normalize_before = normalize_before
        self.decoder_pe = PositionalEncoding_Multi(attention_dim, positional_dropout_rate)
     
        self.attention_heads = attention_heads
        self.semantic_proj = nn.Linear(attention_dim, attention_dim)   #模态映射
        self.audio_proj = nn.Linear(attention_dim, attention_dim)

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        self.decoders = repeat(
            num_blocks,
            lambda lnum: SemanticEncoderLayer(
                attention_dim,
                nn.MultiheadAttention(attention_dim,attention_heads,dropout=self_attention_dropout_rate),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
                lnum
            ),
        )
        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)

        # self.cls_op = nn.Sequential(nn.Linear(attention_dim, attention_dim*2),nn.ReLU(True),nn.Linear(attention_dim*2, vocab_size))
        self.cls_op = nn.Linear(attention_dim, vocab_size)

    @staticmethod
    def generate_multi_modal_mask_new(b, l_v, l_s, ilens, ylens):
        mask_s_s = (torch.triu(torch.ones(l_s, l_s)) == 1).transpose(0, 1).to(ilens.device)
        mask_s_s = mask_s_s.float().masked_fill(mask_s_s == 0, float('-inf')).masked_fill(mask_s_s == 1, float(0.0))    #只有下三角的semantic和semantic的mask矩阵,不mask的区域为0，其他区域为负无穷大
        mask_s_s = mask_s_s.expand(b, l_s, l_s)

        if ylens is not None:
            tgt_mask = (make_pad_mask(ylens)[:, None, :])
            tgt_mask = tgt_mask.expand(b,l_s,l_s)
            mask_s_s = mask_s_s.masked_fill(tgt_mask == 1, float('-inf'))

        mask_s_v = torch.zeros(l_s,l_v).to(ilens.device)
        mask_s_v = mask_s_v.expand(b,l_s,l_v)

        if ilens is not None:
            src_mask = (make_pad_mask(ilens)[:, None, :])
            src_mask_s_v = src_mask.expand(b,l_s,l_v)
            mask_s_v = mask_s_v.float().masked_fill(src_mask_s_v == 1, float('-inf'))

        mask = torch.cat([mask_s_v,mask_s_s],dim=2)
        return mask
    
    @staticmethod
    def wenet_make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
        """Make mask tensor containing indices of padded part.
        See description of make_non_pad_mask.
        Args:
            lengths (torch.Tensor): Batch of lengths (B,).
        Returns:
            torch.Tensor: Mask tensor containing indices of padded part.
        Examples:
            >>> lengths = [5, 3, 2]
            >>> make_pad_mask(lengths)
            masks = [[0, 0, 0, 0 ,0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 1, 1]]
        """
        batch_size = lengths.size(0)
        max_len = max_len if max_len > 0 else lengths.max().item()
        seq_range = torch.arange(0,
                                max_len,
                                dtype=torch.int64,
                                device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_length_expand = lengths.unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand
        return mask
    # @staticmethod
    # def generate_multi_modal_mask(l_v,l_s):
    #     mask_s_s = (torch.triu(torch.ones(l_s, l_s)) == 1).transpose(0, 1)
    #     mask_s_s = mask_s_s.float().masked_fill(mask_s_s == 0, float('-inf')).masked_fill(mask_s_s == 1, float(0.0))    #只有下三角的semantic和semantic的mask矩阵,不mask的区域为0，其他区域为负无穷大
    #     mask_v_v = torch.zeros(l_v,l_v)     #visual和visual的mask，全部为0
    #     mask_s_v = torch.zeros(l_s,l_v)     #semantic和visual的mask，全部为0
    #     mask_v_s = torch.zeros(l_v,l_s).fill_(float('-inf'))     #visual和semantic的mask，全部为0
    #     mask_t = torch.cat([mask_v_v,mask_v_s],dim=1)
    #     mask_b = torch.cat([mask_s_v,mask_s_s],dim=1)
    #     mask = torch.cat([mask_t,mask_b],dim=0)
    #     return mask

    def forward(self, hs_pad, hs_mask, ys_in_pad, ys_in_lens,
                r_ys_in_pad, reverse_weight):
        """Forward decoder.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out) if
                input_layer == "embed". In the other case, input tensor
                (#batch, maxlen_out, odim).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).

        Returns:
            torch.Tensor: Decoded token score before softmax (#batch, maxlen_out, odim)
                   if use_output_layer is True. In the other case,final block outputs
                   (#batch, maxlen_out, attention_dim).
            torch.Tensor: Score mask before softmax (#batch, maxlen_out).

        """
        
        # padding_mask = self.wenet_make_pad_mask(hlens)
        hlens = hs_mask.sum(dim=-1).squeeze(-1)
  
        encoder_out = hs_pad
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # padding_mask = torch.cat((padding_mask, ~tgt_mask.squeeze(1)), dim=1)
        device = encoder_out.device
        encoder_out = encoder_out.permute(1,0,2)
        l_a, B, C = encoder_out.size()
        dec_input = tgt.permute(1,0)  #L,B
        l_s,B = dec_input.size()
        # dec_input = torch.cat([self.STA_index*torch.ones(1,B).type_as(dec_input),dec_input[:-1,:]],dim=0)
        atten_mask = self.generate_multi_modal_mask_new(B, l_a,l_s, hlens, ys_in_lens).to(device)
        atten_mask = atten_mask.repeat(self.attention_heads, 1, 1)
        dec_input = self.embed(dec_input)
        audio_features = self.audio_proj(encoder_out)  #l_v,B,C
        semantic_features = self.semantic_proj(dec_input)     #l_s,B,C
        multi_modal_input = torch.cat([audio_features,semantic_features],dim=0)  #l_s+l_v,B,C
        multi_modal_input = self.decoder_pe(multi_modal_input)
        padding_mask = None
        multi_modal_input, padding_mask, atten_mask,l_a = self.decoders(
            multi_modal_input, padding_mask, atten_mask,l_a
        )
        if self.normalize_before:
            multi_modal_input = self.after_norm(multi_modal_input)
        x = multi_modal_input[l_a:,:,:]
        x = self.cls_op(x)
        x_ctc = multi_modal_input[:l_a,:,:]
        x_ctc = x_ctc.permute(1,0,2).contiguous()
        return x.permute(1,0,2).contiguous(), tgt_mask.sum(1), x_ctc

    def forward_one_step(self, tgt, tgt_mask, memory, cache=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (batch, maxlen_out, odim).
            List[torch.Tensor]: List of cache tensors of each decoder layer.

        """
        device = memory.device
        encoder_out = memory.permute(1,0,2)
        l_a, B, C = encoder_out.size()
        dec_input = tgt.permute(1,0)  #L,B
        l_s, B = dec_input.size()
        audio_features = self.audio_proj(encoder_out)  #l_v, B, C
        padding_mask = None
        atten_mask = self.generate_multi_modal_mask_new(B, l_a,l_s, None, None).to(device)
        atten_mask = atten_mask.repeat(self.attention_heads, 1, 1)
        dec_input = self.embed(dec_input)
        semantic_features = self.semantic_proj(dec_input)     #l_s,B,C
        multi_modal_input = torch.cat([audio_features,semantic_features],dim=0)  #l_s+l_v,B,C
        multi_modal_input = self.decoder_pe(multi_modal_input)

        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            multi_modal_input, padding_mask, atten_mask, l_a = decoder(
                multi_modal_input, padding_mask, atten_mask, l_a, cache=c
            )
            new_cache.append(multi_modal_input.permute(1,0,2))
        
        if self.normalize_before:
            multi_modal_input = self.after_norm(multi_modal_input)
        # multi_modal_input, atten_mask = self.decoders(
        #     multi_modal_input, atten_mask
        # )
        x = multi_modal_input[l_a:,:,:]
        x = x.permute(1,0,2).contiguous()
        x = x[:, -1]
        x = self.cls_op(x)
        x = torch.log_softmax(x, dim=-1)

        x_ctc = multi_modal_input[:l_a,:,:]
        x_ctc = x_ctc.permute(1,0,2).contiguous()
        return x, new_cache, x_ctc

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if self.selfattention_layer_type != "selfattn":
            # TODO(karita): implement cache
            logging.warning(
                f"{self.selfattention_layer_type} does not support cached decoding."
            )
            state = None
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]
        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states, _ = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

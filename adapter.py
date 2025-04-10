import torch
import torch.nn as nn
   
class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class Adapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        moe: bool
    ):
        super(Adapter, self).__init__()

        self.moe = moe
        if not moe:
            self.activation = nn.GELU()
            self.fc1 = nn.Linear(in_dim, mid_dim, bias=False)
            self.fc2 = nn.Linear(mid_dim, in_dim, bias=False)
        else:
            self.expert1 = nn.Linear(in_dim, mid_dim, bias=False)
            self.moe_layer1 = MoELayer(hidden_size=in_dim, num_experts=4, expert=self.expert1, 
                     route_method='gate-token', vocab_size=None, hash_list=None)
            self.expert2 = nn.Linear(mid_dim, in_dim, bias=False)
            self.moe_layer2 = MoELayer(hidden_size=mid_dim, num_experts=4, expert=self.expert2, 
                     route_method='gate-token', vocab_size=None, hash_list=None)
            self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        if not self.moe:
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            return residual + x, 0.0
        else:
            x, balance_loss1, gate_load1 = self.moe_layer1(x, None, None)
            x = self.activation(x)
            x, balance_loss2, gate_load2 = self.moe_layer2(x, None, None)
        return x, balance_loss1 + balance_loss2


model1 = Conv1dSubsampler(1280, 2*1280, 3584,kernel_sizes = (5, 5,5),)
nums = sum(p.numel() for p in model1.parameters() if p.requires_grad)
print(nums)
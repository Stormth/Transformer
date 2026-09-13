"""编码器：把源语言句子变成一组"带上下文的向量表示"（memory）。

    输入 id  [B, Tsrc]
      -> 词嵌入 + 位置编码        [B, Tsrc, d_model]
      -> N × EncoderLayer         [B, Tsrc, d_model]
      -> memory                   [B, Tsrc, d_model]

每个 EncoderLayer 包含两个子层：
    1. 多头自注意力（双向）：每个词都能看到整句的所有词（掩码只屏蔽 <pad>）
    2. 位置前馈网络

注意：编码器是**双向**的，这正是它能做机器翻译、BERT 式理解任务的关键；
解码器则是**单向**的（因果掩码），因为生成时必须避免偷看未来。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward, SublayerConnection


class EncoderLayer(nn.Module):
    """一层编码器。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.sublayer = nn.ModuleList(
            [SublayerConnection(d_model, dropout, norm_first) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # 子层 1：自注意力（query = key = value = x）
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, mask)[0])
        # 子层 2：前馈网络
        x = self.sublayer[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """N 层编码器堆叠。"""

    def __init__(self, layer: EncoderLayer, num_layers: int, norm_first: bool = False):
        super().__init__()
        # 用 deepcopy 复制 N 份：结构相同，但参数各自独立
        import copy

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        # pre-norm 结构需要在最后一层之后补一个 LayerNorm（否则输出未归一化）
        self.norm = nn.LayerNorm(layer.self_attn.d_model, eps=1e-6) if norm_first else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

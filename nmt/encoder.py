"""编码器：把英文句子读成一组上下文化向量。

一层编码器只有两个子层：

    1. 自注意力   —— 每个词看一整句，按相关度加权取信息
    2. 前馈网络   —— 对每个位置单独加工

外面各套一层"残差 + LayerNorm"。6 层堆起来，
每层都会在上层的表示基础上再做一次"全局沟通 + 局部加工"。

编码器的输出叫 memory，形状 [B, S, d_model]，
后面解码器的交叉注意力会反复查询它。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward, SublayerConnection, clone_layers


class EncoderLayer(nn.Module):
    """编码器单层。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = "relu",
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, attention_dropout, bias=bias)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.attn_sublayer = SublayerConnection(d_model, dropout, norm_first)
        self.ffn_sublayer = SublayerConnection(d_model, dropout, norm_first)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        captured = {}

        def self_attention(h: torch.Tensor) -> torch.Tensor:
            out, weights = self.self_attn(h, h, h, mask=mask, return_weights=return_weights)
            captured["self"] = weights
            return out

        x = self.attn_sublayer(x, self_attention)
        x = self.ffn_sublayer(x, self.ffn)
        return x, captured.get("self")


class Encoder(nn.Module):
    """N 层编码器堆叠。

    pre-norm 时每层的输出没有归一化，所以在整个堆叠的最后补一次 LayerNorm，
    否则送进解码器的数值尺度会随层数漂移。
    """

    def __init__(
        self,
        layer: EncoderLayer,
        num_layers: int,
        d_model: int,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.layers = clone_layers(layer, num_layers)
        self.norm = nn.LayerNorm(d_model, eps=1e-6) if norm_first else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """x: [B, S, d_model] -> memory: [B, S, d_model]"""

        all_weights = [] if return_weights else None
        for layer in self.layers:
            x, weights = layer(x, mask, return_weights=return_weights)
            if all_weights is not None:
                all_weights.append(weights)
        if self.norm is not None:
            x = self.norm(x)
        return x, all_weights

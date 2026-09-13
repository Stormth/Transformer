"""解码器：一边看英文原句，一边从左到右写字。

一层解码器有三个子层（比编码器多一个）：

    1. 掩码自注意力 —— 只能看已经写出来的部分（因果掩码）
    2. 交叉注意力   —— query 来自中文侧，key/value 来自编码器输出的英文 memory
    3. 前馈网络

"交叉注意力"就是翻译发生的地方：每次要写下一个中文字时，
它都在问"英文原句里哪几个词跟我现在写的这个字有关"。

增量解码（生成第 t 个字）时不必重算整段中文：
自注意力用 cache_append=True 把新的 K/V 接到历史后面，
交叉注意力用 cache 把英文的 K/V 存一次反复用。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward, SublayerConnection, clone_layers


class DecoderLayer(nn.Module):
    """解码器单层。"""

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
        self.cross_attn = MultiHeadAttention(d_model, n_heads, attention_dropout, bias=bias)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.self_sublayer = SublayerConnection(d_model, dropout, norm_first)
        self.cross_sublayer = SublayerConnection(d_model, dropout, norm_first)
        self.ffn_sublayer = SublayerConnection(d_model, dropout, norm_first)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        self_cache: Optional[dict] = None,
        cross_cache: Optional[dict] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        captured: Dict[str, torch.Tensor] = {}

        # 子层 1：中文侧的自注意力（因果 + padding 掩码）
        def self_attention(h: torch.Tensor) -> torch.Tensor:
            out, weights = self.self_attn(
                h, h, h, mask=self_mask, cache=self_cache,
                cache_append=True, return_weights=return_weights,
            )
            captured["self"] = weights
            return out

        # 子层 2：查询英文 memory（交叉注意力）
        def cross_attention(h: torch.Tensor) -> torch.Tensor:
            out, weights = self.cross_attn(
                h, memory, memory, mask=cross_mask, cache=cross_cache,
                cache_append=False, return_weights=return_weights,
            )
            captured["cross"] = weights
            return out

        x = self.self_sublayer(x, self_attention)
        x = self.cross_sublayer(x, cross_attention)
        x = self.ffn_sublayer(x, self.ffn)
        return x, captured


class Decoder(nn.Module):
    """N 层解码器堆叠。"""

    def __init__(
        self,
        layer: DecoderLayer,
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
        memory: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        caches: Optional[List[dict]] = None,
        cross_caches: Optional[List[dict]] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        all_weights = [] if return_weights else None
        for index, layer in enumerate(self.layers):
            x, weights = layer(
                x,
                memory,
                self_mask=self_mask,
                cross_mask=cross_mask,
                self_cache=None if caches is None else caches[index],
                cross_cache=None if cross_caches is None else cross_caches[index],
                return_weights=return_weights,
            )
            if all_weights is not None:
                all_weights.append(weights)
        if self.norm is not None:
            x = self.norm(x)
        return x, all_weights

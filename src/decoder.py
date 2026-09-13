"""解码器：自回归地生成目标语言句子。

    输入 id  [B, Ttgt]（右移一位，即以 <bos> 开头）
      -> 词嵌入 + 位置编码
      -> N × DecoderLayer(memory)
      -> 最后一层输出 [B, Ttgt, d_model]
      -> 线性层映射到词表 [B, Ttgt, vocab] -> softmax 得到下一个词的概率

每个 DecoderLayer 有三个子层：
    1. 掩码自注意力（causal）：第 t 个位置只能看到 0..t
    2. 交叉注意力（cross attention）：
          Q 来自解码器当前状态，K/V 来自编码器输出 memory
          这一步才是真正的"翻译"发生的地方——解码器在查阅源句
    3. 位置前馈网络

自回归（teacher forcing）训练时，整句一次性并行前向：
    输入  <bos> 我 喜欢 这本书
    预测  我    喜欢 这本书 <eos>
靠因果掩码保证每个位置看不到未来，所以训练可以并行；
而推理只能一个词一个词地生成。
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import KV, MultiHeadAttention
from .layers import PositionwiseFeedForward, SublayerConnection


class DecoderLayer(nn.Module):
    """一层解码器，同时支持「整句前向」和「单步增量解码」。"""

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
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.sublayer = nn.ModuleList(
            [SublayerConnection(d_model, dropout, norm_first) for _ in range(3)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor] = None,
        past: Optional[Tuple[Optional[KV], Optional[KV]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[KV, KV]]]:
        """前向传播。

        past=None      : 训练 / 全序列推理，x 形状 [B, Ttgt, d_model]
        past=(self_kv, cross_kv) : 增量解码，x 形状 [B, 1, d_model]

        返回 (输出, 新的缓存)；训练时缓存为 None。
        """
        if past is None:
            x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, tgt_mask)[0])
            x = self.sublayer[1](x, lambda t: self.cross_attn(t, memory, memory, src_mask)[0])
            x = self.sublayer[2](x, self.feed_forward)
            return x, None

        past_self, past_cross = past
        out: dict = {}

        def _self_attn(t: torch.Tensor) -> torch.Tensor:
            # 只输入 1 个新 token；历史 K/V 从缓存拼接（append=True）。
            # 不需要 causal mask：缓存里全是"过去"，当前 token 就是最新的那个。
            res, kv = self.self_attn(t, t, t, None, past_kv=past_self, append=True, use_cache=True)
            out["self"] = kv
            return res

        def _cross_attn(t: torch.Tensor) -> torch.Tensor:
            # 交叉注意力的 K/V 只与 memory 有关，整个生成过程中恒定，
            # 所以缓存一次、每步复用（append=False），这是很大的加速。
            res, kv = self.cross_attn(
                t, memory, memory, src_mask, past_kv=past_cross, append=False, use_cache=True
            )
            out["cross"] = kv
            return res

        x = self.sublayer[0](x, _self_attn)
        x = self.sublayer[1](x, _cross_attn)
        x = self.sublayer[2](x, self.feed_forward)
        return x, (out["self"], out["cross"])


class Decoder(nn.Module):
    """N 层解码器堆叠。"""

    def __init__(self, layer: DecoderLayer, num_layers: int, norm_first: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.d_model, eps=1e-6) if norm_first else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

    def forward_step(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        caches: list,
    ) -> Tuple[torch.Tensor, list]:
        """增量解码一步：x 为 [B, 1, d_model]，返回新的缓存列表。"""
        new_caches = []
        for layer, past in zip(self.layers, caches):
            x, new_past = layer(x, memory, src_mask, None, past)
            new_caches.append(new_past)
        return self.norm(x), new_caches

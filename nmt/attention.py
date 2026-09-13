"""注意力机制：《Attention Is All You Need》的核心，也是本项目的核心。

公式（论文 3.2.1）：

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

三个必须想清楚的点：

1. 为什么要除以 sqrt(d_k)？
   Q、K 每个分量独立、均值 0 方差 1 时，点积的方差是 d_k。
   d_k=64 时点积标准差约 8，量级一大，softmax 就会被推向"非 0 即 1"的
   饱和区，梯度几乎消失。除以 sqrt(d_k) 把方差拉回 1，训练才稳定。

2. 多头是怎么做的？
   把 d_model 切成 n_heads 份，每份独立做一次注意力，最后拼回去过一层线性。
   代码上就是一次 reshape + transpose，得到 [B, heads, T, d_k]，
   之后就是一次批量矩阵乘 —— "多头"并不需要用 for 循环写。

3. KV cache 是什么？
   自回归解码时，第 t 步只需要第 t 个位置的 query，
   但需要前 t 个位置的 key/value。把这些 K/V 存下来复用，
   避免每生成一个词都重算整段历史 —— 推理速度提升是数量级的。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import make_causal_mask


def mask_value(dtype: torch.dtype) -> float:
    """被屏蔽位置填的值。

    不能直接写 -1e9：混合精度下 float16 最大只能表示 65504，
    -1e9 会溢出成 -inf，万一整行被屏蔽，softmax(-inf) 就是 nan。
    用 finfo.min 既够小（softmax 后≈0），又不会溢出。
    """

    return torch.finfo(dtype).min


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_weights: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """缩放点积注意力。

    query: [B, h, Tq, d_k]
    key:   [B, h, Tk, d_k]
    value: [B, h, Tk, d_v]（这里 d_v == d_k）
    mask:  可广播到 [B, h, Tq, Tk] 的 bool 张量，True = 屏蔽

    返回 [B, h, Tq, d_v]；return_weights=True 时额外返回注意力权重 [B, h, Tq, Tk]。
    """

    d_k = query.size(-1)

    # [B,h,Tq,d_k] @ [B,h,d_k,Tk] -> [B,h,Tq,Tk]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask 里 True 的位置填极小值，softmax 后自然≈0
        scores = scores.masked_fill(mask, mask_value(scores.dtype))

    # 沿最后一维（所有 key 的位置）归一化
    weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=training)

    output = torch.matmul(weights, value)  # [B,h,Tq,Tk] @ [B,h,Tk,d_v] -> [B,h,Tq,d_v]
    return output, (weights if return_weights else None)


class MultiHeadAttention(nn.Module):
    """多头注意力：4 个线性层 + 一次缩放点积注意力。

    参数命名故意用 w_q / w_k / w_v / w_o，
    对照论文里的 W^Q W^K W^V W^O，读代码时不用猜。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] -> [B, h, T, d_k]"""

        batch, length, _ = x.shape
        x = x.view(batch, length, self.n_heads, self.d_k)  # 切分最后一维
        return x.transpose(1, 2)                           # 把"头"维度提到前面

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, h, T, d_k] -> [B, T, d_model]"""

        batch, heads, length, d_k = x.shape
        x = x.transpose(1, 2).contiguous()                 # 先换回来，否则 view 会报错
        return x.view(batch, length, heads * d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
        cache_append: bool = False,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """三种用法：

        1. 训练 / 全句前向：query/key/value 都是整段序列，cache=None；
        2. 解码器自注意力增量解码：query 只有 1 步，cache_append=True，
           本次的 K/V 会被接到 cache 里历史的后面；
        3. 交叉注意力：value 是编码器输出，cache_append=False，
           K/V 第一次算完就存进 cache，后面每步直接复用。
        """

        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        if cache is not None:
            if cache_append:
                # 自注意力：把历史 K/V 拼在本次前面 -> [B,h,t,d_k]
                if cache.get("k") is not None:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
            else:
                # 交叉注意力：编码器输出是固定的，算一次就够
                if cache.get("k") is None:
                    cache["k"], cache["v"] = k, v
                else:
                    k, v = cache["k"], cache["v"]

        out, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout, training=self.training,
            return_weights=return_weights,
        )
        return self.w_o(self._merge_heads(out)), weights


def attention_is_causal(weights: torch.Tensor, tol: float = 1e-4) -> bool:
    """单元测试用：检查注意力权重是否真的没有"偷看未来"。"""

    # 取 batch 0、head 0，看严格上三角部分是否都是 0
    matrix = weights[0, 0]
    upper = torch.triu(matrix, diagonal=1)
    return bool(upper.abs().max().item() < tol)


def causal_attention_scores(length: int, device: torch.device) -> torch.Tensor:
    """可视化辅助：返回一个 [length, length] 的因果掩码（True = 屏蔽）。"""

    return make_causal_mask(length, device)

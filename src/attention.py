"""缩放点积注意力与多头注意力。

这是整个 Transformer 的心脏，公式只有一行：

    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V

直觉理解（以翻译「我喜欢这本书」->「i like this book」为例）：
    Q（Query，查询）  : 当前位置"我在找什么信息"
    K（Key，键）      : 每个位置"我能提供什么信息"
    V（Value，值）    : 每个位置"我实际携带的信息"
    QKᵀ              : 打分，衡量"位置 i 需要的信息"与"位置 j 提供的信息"有多匹配
    softmax          : 把分数变成权重（和为 1）
    加权求和 V        : 按权重把信息聚合起来

除以 √d_k 的原因：若 q、k 各分量独立同分布且方差为 1，则点积方差为 d_k，
d_k 越大点积越极端，softmax 会退化成 one-hot，梯度趋近于 0。除以 √d_k 把方差拉回 1。

多头（Multi-Head）：
    把 d_model 维度切成 h 份，每份独立做一次 attention（关注不同的语言现象：
    有人关注主谓一致，有人关注词序，有人关注指代……），最后拼接再线性变换。
    注意 每个头的 d_k = d_model / h，所以多头并不会让计算量变大。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .masks import NEG_INF

# (输出, 注意力权重) 类型别名：注意力权重形状 [B, h, Tq, Tk]
AttentionOutput = Tuple[torch.Tensor, torch.Tensor]
# KV 缓存形状：[B, h, T, d_k]
KV = Tuple[torch.Tensor, torch.Tensor]


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> AttentionOutput:
    """缩放点积注意力（在"头"的维度上运算）。

    形状
    ----
    query : [B, h, Tq, d_k]
    key   : [B, h, Tk, d_k]
    value : [B, h, Tk, d_v]（通常 d_v = d_k）
    mask  : [B, 1, 1, Tk] 或 [B, 1, Tq, Tk]，True = 屏蔽

    返回
    ----
    out   : [B, h, Tq, d_v]
    p_attn: [B, h, Tq, Tk] 注意力权重（可视化时用它）
    """
    d_k = query.size(-1)

    # [B,h,Tq,d_k] @ [B,h,d_k,Tk] -> [B,h,Tq,Tk]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 广播规则：mask 的 Tk 维与 scores 最后一维对齐；
        # 若 mask 是 [B,1,Tq,Tk]，则连 query 位置也逐个对应（因果掩码就是这样用的）
        scores = scores.masked_fill(mask, NEG_INF)

    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    out = torch.matmul(p_attn, value)
    return out, p_attn


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """把同一层复制 n 份（浅拷贝副本，各自有独立参数）。

    nn.ModuleList 会正确注册子模块，不能用 Python 的 list，否则参数不会被优化器看到。
    """
    import copy

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadAttention(nn.Module):
    """多头注意力。

    参数
    ----
    d_model : 模型维度
    n_heads : 头数（必须整除 d_model）
    dropout : 作用在注意力权重上的 dropout

    额外支持的 past_kv（KV cache）：
        推理时每次只输入一个新词，如果每步都重算整个前缀的 K/V，
        复杂度是 O(T²)；缓存住历史 K/V 后每步只需 O(T)。
        这是自回归生成加速的关键（大模型推理必备）。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 四个线性层：W_q, W_k, W_v, W_o（对应论文中的四个投影矩阵）
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.last_attn: Optional[torch.Tensor] = None  # 保存最近一次注意力权重，便于可视化

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] -> [B, h, T, d_k]

        view 把最后一维拆成 (h, d_k)，transpose 把头维换到前面，
        这样后续 matmul 天然按每个头独立并行计算。
        """
        B, T, _ = x.size()
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def project_kv(self, memory: torch.Tensor) -> KV:
        """只对 memory 做 K/V 投影（交叉注意力可以一次算好、反复复用）。"""
        k = self._split_heads(self.linears[1](memory))
        v = self._split_heads(self.linears[2](memory))
        return k, v

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[KV] = None,
        append: bool = True,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KV]]:
        """前向传播。

        past_kv + append=True  : 自注意力增量解码 —— 把本次 K/V 拼到历史后面
        past_kv + append=False : 交叉注意力 —— 直接复用缓存，忽略传入的 key/value
        past_kv=None           : 训练 / 全序列推理 —— 一次性算完

        use_cache=True 时会返回 (k, v) 供下一步拼接。
        ⚠️ 注意第一步的 past_kv 是 None（还没有历史），如果只用
        "past_kv is not None" 来判断是否返回缓存，第一步的 K/V 就会丢掉，
        解码器将永远只能看到当前这一个 token。这个 bug 非常隐蔽
        （训练 loss 正常下降，只有推理输出乱码），所以用显式的 use_cache 控制。
        """
        B = query.size(0)

        if past_kv is not None and not append:
            q = self._split_heads(self.linears[0](query))
            k, v = past_kv
        else:
            q = self._split_heads(self.linears[0](query))
            k = self._split_heads(self.linears[1](key))
            v = self._split_heads(self.linears[2](value))
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=-2)
                v = torch.cat([past_kv[1], v], dim=-2)

        present: Optional[KV] = (k, v) if use_cache else None

        x, p_attn = attention(q, k, v, mask, self.dropout)
        self.last_attn = p_attn.detach()

        Tq = x.size(-2)
        x = x.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  # 多头拼接
        return self.linears[3](x), present

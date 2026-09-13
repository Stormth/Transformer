"""子层组件：残差连接、LayerNorm、位置前馈网络。

Transformer 的每一层都是「子层 + 残差 + LayerNorm」的堆叠。

    x = LayerNorm(x + Sublayer(x))        # post-norm（原论文，训练需要 warmup）
    x = x + Sublayer(LayerNorm(x))        # pre-norm（现代大模型默认，更易训练）

为什么需要残差？
    深层网络梯度容易消失，残差连接让梯度可以"抄近路"直接回传
    （∂(x+F(x))/∂x = 1 + ∂F/∂x，至少有 1 这一条通路）。

为什么 LayerNorm 而不是 BatchNorm？
    序列任务里 batch 内句子长度差异大、并常有 batch=1 的推理场景，
    按"单个样本的最后一维特征"做归一化更稳定。
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class SublayerConnection(nn.Module):
    """残差 + LayerNorm 包装器。

    forward(x, sublayer) 里的 sublayer 是一个"可调用对象"（通常是 lambda），
    接收张量并返回张量，这样同一套代码既能包装注意力也能包装前馈网络。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, norm_first: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络（FFN）：把每个位置独立地映射到高维再映射回来。

        FFN(x) = max(0, x W₁ + b₁) W₂ + b₂

    * 逐位置（position-wise）：同一个 MLP 独立作用于每个时间步，不做跨位置交互；
    * 先升维（d_model -> d_ff，通常 d_ff = 4 × d_model）再降维。

    有意思的事实：Transformer 的参数量和计算量大头都在这里（约占 2/3）。
    它有"键值记忆库"的作用——上层注意力从 FFN 里取回知识。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


def _get_activation(name: str) -> nn.Module:
    """激活函数工厂。

    relu   : 原论文，最经典的起点
    gelu   : BERT/GPT 系列，更平滑
    swish  : Swish/SiLU，注意对应的是 SwiGLU 这类现代变体（见 README 练习）
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("swish", "silu"):
        return nn.SiLU()
    raise ValueError(f"未知激活函数: {name}")

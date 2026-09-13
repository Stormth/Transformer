"""Transformer 的三种"零件"：残差连接、层归一化、前馈网络。

一个编码器/解码器层就是这几个零件的堆叠，读懂了它们，
剩下的只是"把数据接到哪一路"的问题。

残差连接解决什么？网络变深以后梯度传不回去。x = x + f(x) 让梯度
有一条可以直通的高速公路（恒等映射），6 层、12 层、24 层才训得动。

LayerNorm 和 BatchNorm 的区别？LayerNorm 在**单个样本的特征维度**上
做归一化，和 batch size 无关，所以推理时不需要维护滑动统计量，
也天然适合变长序列。

pre-norm 还是 post-norm？

    post-norm（原论文）：x = LayerNorm(x + Dropout(Sublayer(x)))
    pre-norm（现代模型）：x = x + Dropout(Sublayer(LayerNorm(x)))

post-norm 是论文的做法，效果上限高但需要 warmup 才能训稳；
pre-norm 更容易训，深层模型基本都用它。本项目两种都支持，
配置里的 norm_first 就是那个开关（True = pre-norm）。
"""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn


def clone_layers(module: nn.Module, n: int) -> nn.ModuleList:
    """深拷贝 n 份同样的层。

    注意：copy.deepcopy 会复制参数，这 n 层是独立参数的，
    不是共享权重（共享权重在 Transformer 里只出现在词嵌入和输出层之间）。
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SublayerConnection(nn.Module):
    """一个"残差 + 归一化"外壳，把任意子层包起来。"""

    def __init__(self, d_model: int, dropout: float, norm_first: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    """逐位置前馈网络：FFN(x) = W2 · act(W1 · x + b1) + b2。

    对每个位置**独立**作用（位置之间不交流，交流全靠注意力），
    中间维度 d_ff 通常是 4 倍 d_model。

    它在做什么？注意力负责"从别的位置取信息"，FFN 负责
    "在当前位置把信息加工一遍"。可以把 FFN 想成一个 key-value 记忆库，
    大部分参数量都在这里（base 模型每层约 2/3 参数在 FFN）。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


def _get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "relu":
        return nn.functional.relu
    if name == "gelu":
        return nn.functional.gelu
    if name == "swish":
        # 也叫 SiLU，Transformer 变体里很常见
        return nn.functional.silu
    raise ValueError(f"不支持的激活函数 {name!r}，可选：relu / gelu / swish")

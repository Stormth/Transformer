"""词嵌入与位置编码。

Transformer 完全依赖注意力，而注意力本身是"位置无关"的置换等变运算：
把输入词序打乱，输出只是跟着打乱。所以必须**显式**把位置信息注入模型。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """查表得到的词向量，并乘以 √d_model。

    乘 √d_model 的原因：随机初始化的 embedding 数值尺度约为 1，
    而位置编码（sin/cos）数值尺度也约为 1。为了让两者相加时量级相当，
    把 embedding 放大到 √d_model 的量级（这也是原论文的做法）。
    """

    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0, scale: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model) if scale else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] -> [B, T, d_model]"""
        return self.embed(x) * self.scale


class PositionalEncoding(nn.Module):
    """正弦位置编码（原论文）。

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    为什么用 sin/cos 而不是直接学一个位置向量？
        * 不用额外参数，且固定公式可以外推到训练时没见过的更长序列；
        * 对任意固定偏移 k，PE(pos+k) 都可以写成 PE(pos) 的线性变换，
          这给了模型"相对位置"的表达能力。

    现代模型更多用「可学习位置编码」(BERT/GPT) 或 RoPE (LLaMA)，
    见 README 的进阶练习。

    参数 offset 是为**增量解码**准备的：第 t 步只输入 1 个 token，
    需要取 pe[:, t:t+1] 而不是 pe[:, 0:1]。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        # 1 / 10000^(2i/d_model)，用 exp(log) 计算更稳定
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        # register_buffer：随模型一起搬到 GPU / 存进 checkpoint，但不是可训练参数
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: [B, T, d_model] -> [B, T, d_model]"""
        T = x.size(1)
        if offset + T > self.pe.size(1):
            raise ValueError(f"序列长度 {offset + T} 超出位置编码上限 {self.pe.size(1)}")
        return self.dropout(x + self.pe[:, offset : offset + T])

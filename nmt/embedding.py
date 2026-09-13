"""把 id 变成向量，再加上"位置信息"。

两件事：

1. 词嵌入：一个 [vocab, d_model] 的查表矩阵，乘 sqrt(d_model)。
   为什么要乘？位置编码的数值范围是 [-1, 1]，而词嵌入初始化后
   标准差约 1/sqrt(d_model)，两者量级不匹配。乘上 sqrt(d_model)
   让它们缩放到同一量级，相加才有意义（论文 3.4 节）。

2. 位置编码：注意力本身是"置换不变"的 —— 打乱输入顺序，
   输出只是跟着打乱，模型光看注意力根本不知道谁在前谁在后。
   所以必须显式注入位置信息。

   论文用的是正弦位置编码：

       PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
       PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

   为什么用 sin/cos 而不是直接用一个 [max_len, d_model] 的可学习矩阵？
   - 不用额外参数，也不受训练时见过的最大长度限制；
   - 相邻位置之间有平滑的相对关系（PE(pos+k) 可以表示成 PE(pos) 的线性变换），
     这让模型更容易学会"看前面第 k 个词"这种相对位置概念。

   实现上有一个小技巧：不必真去算 10000^(2i/d)，而是先在 log 空间
   算 1/10000^(2i/d) = exp(-(2i/d) * ln(10000))，再乘 pos，
   数值上更稳，也和官方实现完全一致。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int, device: torch.device | None = None) -> torch.Tensor:
    """返回 [max_len, d_model] 的正弦位置编码表。"""

    position = torch.arange(max_len, dtype=torch.float32, device=device).unsqueeze(1)  # [max_len, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )                                                                                 # [d_model/2]

    pe = torch.zeros(max_len, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维用 sin
    pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维用 cos
    return pe


class TokenEmbedding(nn.Module):
    """词嵌入 + 缩放。"""

    def __init__(self, vocab_size: int, d_model: int, scale: bool = True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = scale

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T] -> [B, T, d_model]
        x = self.embedding(tokens)
        return x * math.sqrt(self.embedding.embedding_dim) if self.scale else x


class PositionalEncoding(nn.Module):
    """把正弦位置编码加到词嵌入上。

    编码表是固定常量（不需要梯度），所以注册成 buffer：
    它会跟着 state_dict 一起保存，但不会被优化器更新。
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: [B, T, d_model]；offset 是这段序列的起始位置。

        增量解码时每次只喂 1 个位置，位置编码必须从 offset 处取，
        否则每生成一个字都会被当成位置 0，结果和全句前向对不上。
        """

        length = x.size(1)
        if offset + length > self.pe.size(0):
            raise ValueError(
                f"序列长度 {offset + length} 超过了位置编码支持的最大长度 {self.pe.size(0)}，"
                "请调大 ModelConfig.max_len 或把句子截短"
            )
        pe = self.pe[offset : offset + length].to(dtype=x.dtype)
        return self.dropout(x + pe)

"""掩码（mask）工具。

Transformer 里有两种掩码，**这是最容易搞混的地方**，记住一个约定：

    mask 为 True 的位置 = 不合法 = 在 attention 分数上填入 -inf
    mask 为 False 的位置 = 合法 = 参与 softmax

1. padding mask（填充掩码）
   一个 batch 里句子长度不同，短的用 <pad> 补齐。这些 <pad> 不是真实词，
   不能让它们参与注意力计算，否则模型会"看见"一堆无意义的占位符。

       src:  [我, 喜欢, 这本书, <pad>, <pad>]   lengths=[3]
       mask: [[F,F,F,T,T]]  形状 [B, 1, 1, T]

2. causal mask（因果掩码 / 下三角掩码）
   解码器是自回归的：预测第 t 个词时只能看到 0..t 个词，不能偷看未来。

       T=4:
         [ F  T  T  T ]
         [ F  F  T  T ]
         [ F  F  F  T ]
         [ F  F  F  F ]

三种 attention 各需要什么掩码：
    编码器自注意力  : padding mask(src)
    解码器自注意力  : causal mask  AND  padding mask(tgt)
    解码器交叉注意力: padding mask(src)
"""

from __future__ import annotations

import torch
from torch import Tensor

# 用有限大负数而不是 float('-inf')：
# 如果某一行被全部屏蔽，softmax(-inf) = nan，nan 会污染反向传播；
# 而 softmax 全为 -1e9 时得到均匀分布（数值安全，且该位置随后会被 loss 屏蔽）。
NEG_INF = -1e9


def make_pad_mask(lengths: Tensor, max_len: int | None = None) -> Tensor:
    """根据长度生成 padding mask。

    参数
    ----
    lengths : [B] 每句真实长度（不含 padding）
    max_len : 补齐后的序列长度

    返回
    ----
    mask : [B, 1, 1, max_len]，True 表示该位置是 <pad>
    """
    B = lengths.size(0)
    if max_len is None:
        max_len = int(lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device)  # [T]
    # [1, T] 与 [B, 1] 广播 -> [B, T]，再 reshape 成 [B, 1, 1, T] 以便和注意力分数广播
    mask = idx.unsqueeze(0) >= lengths.unsqueeze(1)
    return mask.view(B, 1, 1, max_len)


def make_causal_mask(size: int, device: torch.device | None = None) -> Tensor:
    """生成因果掩码。

    返回 [1, 1, size, size] 的上三角矩阵（对角线也是 True，因为 t 不能看自己
    之后的词；t 自己当然要看，所以严格来说是"严格上三角"作为屏蔽区）。

        mask[i][j] = (j > i)   -> 屏蔽未来
    """
    idx = torch.arange(size, device=device)
    # (j > i) 的位置是"未来"，需要屏蔽；结果是严格上三角矩阵
    mask = idx.unsqueeze(0) > idx.unsqueeze(1)  # [1, size] > [size, 1] -> [size, size]
    return mask.view(1, 1, size, size)


def combine_masks(*masks: Tensor | None) -> Tensor | None:
    """把多个 mask 做逻辑或合并（任何一个屏蔽生效即屏蔽）。"""
    result = None
    for m in masks:
        if m is None:
            continue
        result = m if result is None else (result | m)
    return result


def lengths_from_ids(ids: Tensor, pad_id: int = 0) -> Tensor:
    """从 id 矩阵反推每个序列的真实长度（用于统计，不参与计算）。"""
    return (ids != pad_id).sum(dim=1)

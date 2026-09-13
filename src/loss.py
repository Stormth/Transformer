"""损失函数：带标签平滑的交叉熵。

标准交叉熵把目标当成 one-hot：
    loss = -log P(正确词)
问题是：标签是"硬"的，模型被推向"正确词概率 -> 1，其余 -> 0"，
这会让模型过度自信、泛化变差（尤其是小语料上很快过拟合）。

标签平滑把目标分布改成：
    P_smooth(正确词) = 1 - ε
    P_smooth(其它词) = ε / (V - 1)
于是 loss = (1-ε) * NLL + ε * (所有词的均匀 NLL)，等价于给模型一个"别太自信"的正则项。

注意两个细节：
1. <pad> 必须从 loss 中排除（ignore_index）：它只是补齐用的占位符。
2. 分母用"有效 token 数"而不是 batch 元素数，这样不同长度的 batch 的 loss 才可比。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing 必须在 [0, 1) 之间")
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        logits : [B, T, V]（未过 softmax）
        target : [B, T]   （右移一位后的目标 id）
        """
        if logits.size(-2) != target.size(-1):
            raise ValueError(f"logits 长度 {logits.size(-2)} 与 target 长度 {target.size(-1)} 不一致")

        logprobs = F.log_softmax(logits, dim=-1)                              # [B, T, V]
        nll = -logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)          # 正确词的 NLL
        smooth = -logprobs.mean(dim=-1)                                       # 均匀分布的 NLL
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth         # [B, T]

        mask = (target != self.ignore_index).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss


def build_criterion(smoothing: float, pad_id: int = 0) -> nn.Module:
    """smoothing=0 时退化为普通交叉熵（依然忽略 <pad>），接口保持统一。"""
    return LabelSmoothedCrossEntropy(smoothing=smoothing, ignore_index=pad_id)


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 0) -> tuple[float, int]:
    """训练时观察用的"逐词准确率"（不含 <pad>）。返回 (准确率, 有效 token 数)。"""
    pred = logits.argmax(dim=-1)
    mask = target != ignore_index
    correct = ((pred == target) & mask).sum().item()
    total = int(mask.sum().item())
    return (correct / total if total else 0.0), total

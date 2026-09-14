"""训练目标：带标签平滑的交叉熵。

普通交叉熵只关心"正确答案那一项"，逼模型把正确词的概率推到 1，
在翻译这种"一句话有多种正确译法"的任务上容易过拟合。

标签平滑（论文 5.4 节，eps=0.1）把正确答案的置信度从 1 降到 0.9，
剩下的 0.1 平摊给其他词。它在说：
"这一个词是对的，但别的词也不是完全没道理。" 

副作用是 loss 不可能降到 0，训练初期你看到 loss 卡在 1.x 不要慌 ——
把 p=0.9 代进 -log(p) 就知道理论下界大约是 0.105*nats 再加平滑项。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """ignore_index（<pad>）位置不产生 loss。"""

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing 必须在 [0, 1) 区间")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: [B, T, V]（或 [N, V]）；target: [B, T]（或 [N]）。返回标量。

        实现上是"公式版"而不是"分布版"：不构造 [N, V] 的目标分布张量。

        为什么？词表 32000、一个 batch 5000 个 token 时，那个张量是 660 MB，
        再加上逐元素相乘的临时结果，光损失函数就要 1.3 GB —— 这是多卡训练里
        很典型的 OOM 来源（本项目的 8 卡跑就栽在这上面）。
        正确的做法是把标签平滑写成两项之和：

            loss = (1-eps)·(-log p(正确词)) + eps/(V-1)·(Σ_{v≠正确词} -log p(v))

        注意第二项要**排除正确词自己**（原来那版是把正确词那格改写成 1-eps，
        等于把它的 eps/(V-1) 那份抽掉了）。所以用"总和减去自己"来表示：

            Σ_{v≠t} (-log p(v)) = smooth - nll

        数学上和分布版逐位等价，但只需要 [N] 大小的中间量，不materialize [N, V]。
        """

        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        target = target.reshape(-1)

        log_probs = F.log_softmax(logits, dim=-1)

        valid = target != self.pad_id
        # gather 出正确词的对数概率（padding 位置取到的是无关项，后面会被 mask 掉）
        safe_target = target.masked_fill(~valid, 0)
        nll = -log_probs.gather(dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.sum(dim=-1)

        per_token = (1.0 - self.smoothing) * nll + self.smoothing / (vocab_size - 1) * (smooth - nll)
        n_valid = valid.sum().clamp(min=1)
        return (per_token * valid).sum() / n_valid


class PlainCrossEntropy(nn.Module):
    """普通交叉熵，但接受和 LabelSmoothingLoss 一样的输入形状。

    坑：nn.CrossEntropyLoss 对形状很敏感 —— 输入 [N, C, d1] 时它要求 target 是 [N, d1]，
    也就是把 C 当成"类别维"。而我们拿到的是 [B, T, V]，类别维在最后一维。
    所以这里统一拍平成 [B*T, V] / [B*T]，两个损失函数就能互换使用。
    """

    def __init__(self, pad_id: int) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


def build_criterion(pad_id: int, vocab_size: int, smoothing: float) -> nn.Module:
    """smoothing=0 时退化成普通交叉熵（更快、更省显存）。"""

    if smoothing <= 0.0:
        return PlainCrossEntropy(pad_id)
    return LabelSmoothingLoss(vocab_size, pad_id, smoothing)

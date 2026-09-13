"""学习率调度：Noam（原论文的 warmup + 逆平方根衰减）。

            lr = scale · d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))

曲线形状（先增后减）：
        lr
        │      ╭─╮
        │    ╱    ╲___
        │  ╱           ╲____
        └──────────────────────── step
        linear warmup ↑   ↑ 逆平方根衰减

为什么需要 warmup？
    训练初期模型参数随机、注意力分布接近均匀，梯度方向噪声很大。
    Adam 的二阶矩估计也还没稳定，此时用大学习率容易"训飞"。
    先线性升温、再缓慢衰减，是 Transformer 能被训练起来的关键技巧之一。

warmup 步数经验值：大规模语料用 4000（原论文），小语料几百即可。
本项目把 warmup 做成可配置项，正是为了让你亲手感受这个超参数的影响。
"""

from __future__ import annotations

import torch


class NoamLR(torch.optim.lr_scheduler.LambdaLR):
    """继承 LambdaLR，自动获得 state_dict / load_state_dict（断点续训很方便）。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        scale: float = 1.0,
        last_epoch: int = -1,
    ):
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)
        self.scale = scale
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, step: int) -> float:
        step = max(step, 1)  # 第 0 步不能取 0 的负幂
        return self.scale * (self.d_model ** -0.5) * min(
            step ** -0.5, step * (self.warmup_steps ** -1.5)
        )

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """按原论文设置构造 Adam（β2=0.98、eps=1e-9，而不是默认的 0.999/1e-8）。"""
    return torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # 真实学习率完全由调度器决定
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

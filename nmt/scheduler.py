"""Noam 学习率：warmup + 逆平方根衰减（论文 5.3 节）。

        lr = scale * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

分成两段看：

* step < warmup：lr 随 step **线性上升**。
  Transformer 是 post-norm + 一堆残差，训练最开始梯度方向很不可靠，
  先用小步长试探，再逐渐加大。

* step > warmup：lr 按 step^(-0.5) **衰减**。
  随着接近最优点，步长要越来越小，否则会在最优点附近来回跳。

两段的交点正好是 step == warmup，此时
min() 的两项相等，lr 达到峰值 scale * d_model^(-0.5) * warmup^(-0.5)。

论文取 warmup=4000、d_model=512，峰值约 7e-4，
base 模型上是个很稳的起点。
"""

from __future__ import annotations

from typing import Dict

import torch


class NoamScheduler:
    """手写而不是用 torch.optim.lr_scheduler，因为学习率公式就是论文的一行，
    自己写出来最清楚，也方便你改成别的衰减策略做对比实验。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        scale: float = 1.0,
        last_step: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)  # 防止除零
        self.scale = scale
        self._step = last_step
        self._set_lr(self.lr_at(last_step))

    def lr_at(self, step: int) -> float:
        """step 从 1 开始计数（第 0 步用 1 代入，避免 0^(-0.5) 除零）。"""

        step = max(1, step)
        return (
            self.scale
            * (self.d_model ** -0.5)
            * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        )

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> float:
        """走一步参数更新，返回本步用的学习率。"""

        self._step += 1
        lr = self.lr_at(self._step)
        self._set_lr(lr)
        return lr

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> Dict[str, float | int]:
        return {"step": self._step, "warmup_steps": self.warmup_steps, "scale": self.scale}

    def load_state_dict(self, state: Dict[str, float | int]) -> None:
        self._step = int(state.get("step", 0))
        self.warmup_steps = max(1, int(state.get("warmup_steps", self.warmup_steps)))
        self.scale = float(state.get("scale", self.scale))
        self._set_lr(self.lr_at(self._step))

    @property
    def step_count(self) -> int:
        return self._step

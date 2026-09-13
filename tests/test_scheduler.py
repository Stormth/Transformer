"""学习率调度：warmup 阶段线性上升，之后按 step^-0.5 衰减。"""

from __future__ import annotations

import torch

from nmt.scheduler import NoamScheduler


def _make(warmup: int = 100, d_model: int = 64) -> NoamScheduler:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    return NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup)


def test_lr_rises_during_warmup() -> None:
    scheduler = _make()
    rates = [scheduler.lr_at(step) for step in (1, 25, 50, 99)]
    assert rates == sorted(rates)
    assert rates[-1] < scheduler.lr_at(100)


def test_peak_is_at_warmup_then_decays() -> None:
    scheduler = _make(warmup=100)
    peak = scheduler.lr_at(100)
    assert scheduler.lr_at(200) < peak
    assert scheduler.lr_at(1000) < scheduler.lr_at(200)


def test_peak_matches_paper_formula() -> None:
    scheduler = _make(warmup=4000, d_model=512)
    expected = 512 ** -0.5 * 4000 ** -0.5
    assert abs(scheduler.lr_at(4000) - expected) < 1e-12


def test_step_updates_optimizer_lr() -> None:
    scheduler = _make()
    scheduler.step()
    assert scheduler.current_lr == scheduler.lr_at(1)
    for _ in range(9):
        scheduler.step()
    assert scheduler.current_lr == scheduler.lr_at(10)
    assert scheduler.step_count == 10


def test_state_dict_roundtrip() -> None:
    scheduler = _make()
    for _ in range(30):
        scheduler.step()
    state = scheduler.state_dict()

    restored = _make()
    restored.load_state_dict(state)
    assert restored.step_count == scheduler.step_count
    assert restored.current_lr == scheduler.current_lr

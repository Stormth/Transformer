"""checkpoint 的保存与恢复。

一个"能续训"的 checkpoint 里应该有什么？

    model     模型权重
    optimizer 优化器状态（Adam 的一阶/二阶矩，不存的话续训等于重新开始）
    scheduler 学习率走到第几步了
    scaler    混合精度缩放因子（float16 才有）
    progress  epoch / step / 历史指标 / 最优分数
    config    结构超参，保证"读的时候不会和写的时候对不上"
    rng       随机数状态（想让续训后的数据顺序也完全一致时才需要）

少一样，续训出来的曲线都会和你预期的不一样，而且往往是"看起来正常但悄悄变差"。
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .config import Config
from .utils import ensure_dir

CHECKPOINT_FORMAT = 1


@dataclass
class Checkpoint:
    config: Dict[str, Any]
    model: Dict[str, Any]
    optimizer: Optional[Dict[str, Any]] = None
    scheduler: Optional[Dict[str, Any]] = None
    scaler: Optional[Dict[str, Any]] = None
    epoch: int = 0
    step: int = 0
    best_score: float = float("-inf")
    best_metric: str = "bleu"
    history: List[Dict[str, float]] = field(default_factory=list)
    vocab_path: str = "data/ready/vocab.json"
    rng: Optional[Dict[str, Any]] = None


def _safe_load(path: Path, map_location: Any = "cpu") -> Dict[str, Any]:
    """torch 2.6 起 torch.load 默认 weights_only=True，会拒绝我们自己存的字典。
    这里的文件都是自己刚写出来的，显式声明 weights_only=False。"""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # 很老的 torch 没有这个参数
        return torch.load(path, map_location=map_location)


def capture_rng() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """把优化器状态里的张量搬到目标设备。

    为什么要单独做这件事？
    因为读取 checkpoint 时必须用 map_location="cpu"：
    `torch.save` 存下来的除了权重，还有 **随机数状态**（一个 CPU ByteTensor）。
    如果用 map_location="cuda" 去读，这个 ByteTensor 也会被搬到显卡上，
    后面 torch.set_rng_state() 就会抛 "RNG state must be a torch.ByteTensor"。

    正确做法是：权重和优化器状态按 CPU 读进来，再自己搬到设备上。
    模型侧用 load_state_dict 复制进已经就位的参数即可，
    但优化器状态不会被自动搬 —— 所以需要这个函数。
    """

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: Config,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    best_score: float = float("-inf"),
    best_metric: str = "bleu",
    history: Optional[List[Dict[str, float]]] = None,
    vocab_path: str = "data/ready/vocab.json",
    save_rng: bool = True,
    save_optimizer: bool = True,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": config.to_dict(),
        "model": model.state_dict(),
        # 快照（用于最后做 checkpoint 平均）不需要优化器状态：
        # Adam 的两个动量占了 2/3 的体积，存下来只是浪费磁盘。
        "optimizer": optimizer.state_dict() if (optimizer is not None and save_optimizer) else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_score": best_score,
        "best_metric": best_metric,
        "history": history or [],
        "vocab_path": vocab_path,
        "rng": capture_rng() if save_rng else None,
    }
    # 先写临时文件再原子替换：训练中途被杀时不会留下半个损坏的 checkpoint，
    # 也能避开 Windows 上"文件还被上一个句柄占着"导致的写入失败。
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> Checkpoint:
    payload = _safe_load(Path(path), map_location=map_location)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} 的格式版本是 {payload.get('format')}，本项目只认识 {CHECKPOINT_FORMAT}")
    return Checkpoint(
        config=payload["config"],
        model=payload["model"],
        optimizer=payload.get("optimizer"),
        scheduler=payload.get("scheduler"),
        scaler=payload.get("scaler"),
        epoch=payload.get("epoch", 0),
        step=payload.get("step", 0),
        best_score=payload.get("best_score", float("-inf")),
        best_metric=payload.get("best_metric", "bleu"),
        history=payload.get("history", []),
        vocab_path=payload.get("vocab_path", "data/ready/vocab.json"),
        rng=payload.get("rng"),
    )


def config_from_checkpoint(checkpoint: Checkpoint) -> Config:
    return Config.from_dict(checkpoint.config)

"""通用工具：随机种子、设备选择、日志、checkpoint 读写。"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


# ---------------------------------------------------------------------- #
# 复现性
# ---------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """固定所有随机源，让实验结果可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[warning] 请求了 {name} 但 CUDA 不可用，回退到 CPU")
        return torch.device("cpu")
    return device


# ---------------------------------------------------------------------- #
# 日志
# ---------------------------------------------------------------------- #
def setup_logger(log_file: Optional[str | Path] = None, name: str = "transformer") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


class Timer:
    """最简单的耗时统计。"""

    def __init__(self) -> None:
        self.t0 = time.time()

    def reset(self) -> None:
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def format(self) -> str:
        s = self.elapsed
        if s < 60:
            return f"{s:.1f}s"
        if s < 3600:
            return f"{s / 60:.1f}m"
        return f"{s / 3600:.1f}h"


def human_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------- #
# Checkpoint
# ---------------------------------------------------------------------- #
def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    **meta: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"model": model.state_dict(), **meta}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def describe_environment() -> str:
    parts = [f"python {sys.version.split()[0]}", f"torch {torch.__version__}"]
    if torch.cuda.is_available():
        parts.append(f"gpu {torch.cuda.get_device_name(0)}")
    else:
        parts.append("cpu only")
    return " | ".join(parts)

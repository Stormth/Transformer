"""通用小工具：控制台编码、随机种子、设备选择、日志、混合精度上下文。

这里刻意不放任何"魔法"，每个函数都能一眼看完。
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


# --------------------------------------------------------------------------
# 控制台 / 日志
# --------------------------------------------------------------------------
class _LiveStderrHandler(logging.Handler):
    """每条日志都现取 sys.stderr、用 print 写出来并立刻 flush。

    为什么不直接用 logging.StreamHandler(sys.stderr)？
    因为 torchrun 这类启动器会替换/包装 sys.stderr，而 StreamHandler 在构造时
    就把 stream 对象抓死了 —— 结果是多卡训练时日志静默消失：
    屏幕上只有 warning，看不到 loss / tok/s / 验证分数，
    人会以为"是不是没在跑"。（这个坑在本项目的 8 卡首跑里真实发生过。）

    每次 emit 时重新取 sys.stderr，行为就和普通的 print(..., file=sys.stderr) 完全一致。
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            print(self.format(record), file=sys.stderr, flush=True)
        except Exception:  # 日志本身不该把训练搞崩
            self.handleError(record)


def setup_console() -> None:
    """把标准输出切成 UTF-8，并让已经建好的 logger 重新绑定到当前的 stderr。

    Windows 控制台默认是 GBK，打印中文/德语等非 ASCII 日志时可能乱码甚至直接抛
    UnicodeEncodeError。所有入口脚本第一件事就调用它。

    为什么要重新绑定 handler？因为 torchrun 这类启动器会在进程启动后**替换**
    sys.stdout / sys.stderr。如果 handler 还攥着旧对象，日志就会静默消失 ——
    表现是"8 卡跑起来了，但屏幕上什么都没有，不知道在不在跑"。这个坑踩过一次。
    """

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    for name in list(logging.root.manager.loggerDict):
        if name != "nmt" and not name.startswith("nmt."):
            continue
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            # 注意也要匹配我们自己的 _LiveStderrHandler（它不是 StreamHandler 的子类），
            # 否则旧 handler 不会被移除、新 handler 又加一个，日志就会打印两遍。
            if isinstance(handler, (logging.StreamHandler, _LiveStderrHandler)):
                logger.removeHandler(handler)
        handler = _LiveStderrHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)


def get_logger(name: str = "nmt", level: int = logging.INFO) -> logging.Logger:
    """返回一个只打一行、带时间戳的 logger。

    注意日志走的是 **stderr** 而不是 stdout：torchrun 只转发子进程的 stderr，
    写 stdout 的日志在多卡运行时会被吞掉 —— 表现就是"8 卡跑起来了，但屏幕上
    只有一个 warning，完全不知道训练进度"。这个坑踩过一次。
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _LiveStderrHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# 复现性 / 设备
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定所有随机源，让实验可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """把 `auto` / `cpu` / `cuda` / `cuda:1` 解析成 torch.device。"""

    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"指定了 {spec}，但这台机器上 torch.cuda.is_available() 是 False")
    return device


def describe_device(device: torch.device) -> str:
    """给日志用的一句话设备描述。"""

    if device.type != "cuda":
        return "CPU"
    props = torch.cuda.get_device_properties(device)
    return f"{props.name}（{props.total_memory / 1024 ** 3:.1f} GB）"


# --------------------------------------------------------------------------
# 混合精度：兼容 torch 2.0 ~ 2.6 的写法差异
# --------------------------------------------------------------------------
def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    """返回 autocast 上下文；CPU 或未开启时返回一个什么都不做的上下文。"""

    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, enabled: bool) -> Optional[Any]:
    """梯度缩放器。

    float16 动态范围窄，梯度容易下溢，必须配 GradScaler；
    bfloat16 动态范围和 float32 一样宽，不需要缩放，直接返回 None。
    torch 2.3 之后推荐 torch.amp.GradScaler("cuda")，老版本用 torch.cuda.amp.GradScaler()。
    """

    if not enabled or device.type != "cuda":
        return None
    try:
        from torch.amp import GradScaler  # torch >= 2.3

        return GradScaler(device.type)
    except ImportError:
        return torch.cuda.amp.GradScaler()


def need_grad_scaler(amp_dtype: torch.dtype) -> bool:
    """只有 float16 需要梯度缩放。"""

    return amp_dtype == torch.float16


# --------------------------------------------------------------------------
# 文件 / 格式化
# --------------------------------------------------------------------------
def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Union[str, Path], obj: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    # utf-8-sig：Windows 上用记事本另存过的 JSON 会带 BOM
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_lines(path: Union[str, Path]) -> List[str]:
    """读一个纯文本文件，每行一句（评测参考译文就是这种格式）。"""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在")
    return path.read_text(encoding="utf-8-sig").splitlines()


def write_lines(path: Union[str, Path], lines: List[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def format_big(n: float) -> str:
    """1234567 -> 1.23M"""

    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return f"{n:.0f}"


def count_parameters(model: torch.nn.Module, only_trainable: bool = True) -> int:
    params = model.parameters()
    if only_trainable:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


class Timer:
    """with Timer() as t: ...  然后 t.elapsed 是秒数。"""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start


class ProgressBar:
    """极简进度条：不依赖 tqdm，只往 stderr 写一行、用 \\r 原地刷新。

    为什么自己写？本项目的卖点之一就是"只依赖 PyTorch"，为了一条进度条引入
    tqdm 不值当；而且自己写才能控制两个关键行为：

    * **只在 TTY 上画**。输出被重定向到文件时（`> log` 或 `| tee`），
      带 \\r 的进度条会在日志里留下一堆互相覆盖的乱码 —— 这种情况自动退化成
      "每隔 N 步打一行"，也就是原来那种日志。
    * **只在真的要画的时候做 GPU 同步**。loss 要显示就得 `.item()`，
      而每步同步会把流水线排空；这里把它限制在最多每 `min_interval` 秒一次。

    多卡时只让 rank 0 构造它，否则 8 条进度条会互相覆盖。
    """

    def __init__(
        self,
        total: int,
        label: str = "",
        enabled: bool = True,
        force: bool = False,
        min_interval: float = 1.0,
        width: int = 20,
    ) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.width = width
        self.min_interval = min_interval
        # 不是 TTY 就不画：避免日志文件被 \r 搞乱
        self.enabled = bool(enabled) and (force or sys.stderr.isatty())
        self.start = time.perf_counter()
        self._last_draw = 0.0
        self._drawn = False

    def _should_draw(self, step: int) -> bool:
        if not self.enabled:
            return False
        if step >= self.total:      # 最后一步一定要画出来
            return True
        now = time.perf_counter()
        if now - self._last_draw < self.min_interval:
            return False
        self._last_draw = now
        return True

    def update(self, step: int, loss: Optional[float] = None, lr: Optional[float] = None, tok_s: Optional[float] = None) -> None:
        """调一次刷新一次（内部自己节流）。loss/lr/tok_s 传已经算好的数字。"""

        if not self._should_draw(step):
            return
        fraction = min(1.0, step / self.total)
        filled = int(self.width * fraction)
        bar = "█" * filled + "·" * (self.width - filled)
        elapsed = time.perf_counter() - self.start
        rate = step / max(1e-9, elapsed)
        eta = (self.total - step) / max(1e-9, rate)

        parts = [f"{self.label}{bar}", f"{step}/{self.total}", f"{fraction:5.1%}",
                 f"{rate:.2f} step/s", f"ETA {human_time(eta)}"]
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
        if lr:
            parts.append(f"lr {lr:.2e}")
        if tok_s:
            parts.append(f"{format_big(tok_s)} tok/s")
        print("\r" + " | ".join(parts), end="", file=sys.stderr, flush=True)
        self._drawn = True

    def finish(self) -> None:
        """换行收尾，免得后面的日志接在进度条同一行上。"""

        if self.enabled and self._drawn:
            print("", file=sys.stderr, flush=True)
            self._drawn = False

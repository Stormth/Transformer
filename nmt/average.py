"""把最后几个 checkpoint 的权重平均起来 —— 论文报的分数就是这么来的。

《Attention Is All You Need》6.1 节：

    "We averaged the last 5 checkpoints, which were written at 10-minute intervals."

为什么平均一下就能涨点？训练后期模型在损失面上来回晃（学习率还没降到 0），
不同的 checkpoint 落在不同的局部坑里，取平均相当于在参数空间里做了一次集成，
通常能白捡 0.5~1 BLEU，而且不需要任何额外的训练。

用法：

    # 训练时开 --override train.keep_last_checkpoints=5，会自动存 epoch_*.pt
    python -m nmt.average --checkpoint-dir checkpoints --num 5 --output checkpoints/averaged.pt

    # 然后照常评测（checkpoint 里带着完整配置，Translator 直接能读）
    python -m nmt.evaluate --checkpoint checkpoints/averaged.pt --split test2014 --beam-size 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .checkpoint import _safe_load
from .utils import get_logger, setup_console

logger = get_logger()


def find_snapshots(checkpoint_dir: Path, prefix: str = "epoch_") -> List[Path]:
    """按 epoch 数字排序，找出训练时存的快照。字符串排序会把 epoch_10 排在 epoch_2 前面。"""

    def sort_key(path: Path) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else -1

    return sorted(
        (p for p in Path(checkpoint_dir).glob(f"{prefix}*.pt")),
        key=sort_key,
    )


def average_checkpoints(paths: List[Path], output: Optional[Path] = None, weights: Optional[List[float]] = None) -> Dict:
    """把多个 checkpoint 的模型权重按元素平均。返回平均后的 payload。

    只在 CPU 上算，而且**逐个累加**而不是全部读进内存再平均：
    60M 参数的 base 模型一个 checkpoint 240 MB，读 5 个就是 1.2 GB，
    再翻倍做平均很容易把内存吃满。
    """

    if not paths:
        raise ValueError("没有找到 checkpoint 快照")
    if weights is None:
        weights = [1.0] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("weights 和 paths 数量不一致")
    total_weight = sum(weights)

    logger.info(f"平均 {len(paths)} 个 checkpoint：{[p.name for p in paths]}")
    first = _safe_load(paths[0], map_location="cpu")
    averaged: Dict[str, torch.Tensor] = {}

    for index, path in enumerate(paths):
        payload = first if index == 0 else _safe_load(path, map_location="cpu")
        state = payload["model"]
        weight = weights[index] / total_weight
        for key, value in state.items():
            if not torch.is_floating_point(value):
                # 整数 buffer（比如位置编码表如果存成整型）直接照抄第一个
                if key not in averaged:
                    averaged[key] = value.clone()
                continue
            contribution = value.to(torch.float32) * weight
            if key in averaged:
                averaged[key] += contribution
            else:
                averaged[key] = contribution
        if payload is not first:
            del payload

    # 转回原来的 dtype（模型权重通常是 float32，混合精度训练的 master 权重也是 float32）
    reference_state = first["model"]
    for key, value in averaged.items():
        averaged[key] = value.to(reference_state[key].dtype)

    result = dict(first)
    result["model"] = averaged
    result["averaged_from"] = [str(p) for p in paths]
    logger.info(f"平均完成（{len(averaged)} 个张量），来自 {len(paths)} 个 checkpoint")

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output)
        logger.info(f"已写入 {output}")
    return result


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description="平均最后 N 个 checkpoint（论文的做法）")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="存放 epoch_*.pt 快照的目录")
    parser.add_argument("--checkpoints", nargs="*", default=None, help="也可以直接指定文件列表")
    parser.add_argument("--num", type=int, default=5, help="取最后几个（默认 5，和论文一致）")
    parser.add_argument("--output", default="checkpoints/averaged.pt")
    args = parser.parse_args()

    if args.checkpoints:
        paths = [Path(p) for p in args.checkpoints]
    else:
        paths = find_snapshots(Path(args.checkpoint_dir))
        if len(paths) < args.num:
            raise SystemExit(
                f"{args.checkpoint_dir} 里只有 {len(paths)} 个快照，不够平均 {args.num} 个。\n"
                "训练时加上 --override train.keep_last_checkpoints=5 才会存快照。"
            )
        paths = paths[-args.num :]

    average_checkpoints(paths, Path(args.output))


if __name__ == "__main__":
    main()

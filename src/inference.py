"""统一的推理接口：文本进、文本出。

训练脚本、评估脚本、命令行翻译工具都复用这里的函数，避免三份重复代码
（也避免"训练时和上线时的预处理不一致"这类经典事故）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from .bpe import BPE
from .config import Config
from .model import Transformer
from .utils import resolve_device


@dataclass
class TranslationResult:
    source: str
    hypothesis: str
    reference: str | None = None
    score: float | None = None  # 束搜索分数（仅束搜索时有值）


def load_model(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path | None = None,
    device: str | torch.device = "auto",
) -> Tuple[Transformer, BPE, Config, torch.device]:
    """从 checkpoint 目录/文件恢复模型。

    checkpoint 目录约定（由 train.py 生成）：
        best.pt / last.pt : 权重 + 优化器状态
        config.json       : 模型与训练超参数
        tokenizer.json    : BPE 分词器
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    if checkpoint_path.is_dir():
        for name in ("best.pt", "last.pt"):
            if (ckpt_dir / name).exists():
                checkpoint_path = ckpt_dir / name
                break
        else:
            raise FileNotFoundError(f"{ckpt_dir} 下找不到 best.pt / last.pt")

    config = Config.load(ckpt_dir / "config.json")
    tokenizer = BPE.load(tokenizer_path or (ckpt_dir / "tokenizer.json"))

    dev = resolve_device(str(device) if device != "auto" else config.train.device)
    model = Transformer(config.model, pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.to(dev).eval()
    return model, tokenizer, config, dev


@torch.no_grad()
def translate_texts(
    model: Transformer,
    tokenizer: BPE,
    texts: Sequence[str],
    device: torch.device | str = "cpu",
    beam_size: int = 4,
    max_len: int = 64,
    batch_size: int = 64,
) -> List[Tuple[str, float | None]]:
    """批量翻译。返回 [(译文, 束搜索分数或 None)]。"""
    device = torch.device(device)
    model.eval()
    results: List[Tuple[str, float | None]] = []

    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        encoded = [tokenizer.encode(t) for t in chunk]
        max_src = max(len(e) for e in encoded) if encoded else 1

        src = torch.full((len(chunk), max_src), tokenizer.pad_id, dtype=torch.long, device=device)
        for i, ids in enumerate(encoded):
            src[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

        if beam_size and beam_size > 1:
            ranked = model.beam_search(src, beam_size=beam_size, max_len=max_len)
            for hyps in ranked:
                ids, score = hyps[0]
                results.append((tokenizer.decode(ids), score))
        else:
            generations = model.greedy_decode(src, max_len=max_len)
            for ids in generations:
                results.append((tokenizer.decode(ids), None))

    return results


def translate_batch_with_references(
    model: Transformer,
    tokenizer: BPE,
    pairs: Sequence[Tuple[str, str]],
    device: torch.device | str = "cpu",
    beam_size: int = 1,
    max_len: int = 64,
    batch_size: int = 64,
) -> List[TranslationResult]:
    """翻译带参考译文的句对，便于算 BLEU / 打印样例。"""
    sources = [p[0] for p in pairs]
    outputs = translate_texts(
        model, tokenizer, sources, device=device, beam_size=beam_size, max_len=max_len, batch_size=batch_size
    )
    return [
        TranslationResult(source=src, hypothesis=hyp, reference=ref, score=score)
        for (src, ref), (hyp, score) in zip(pairs, outputs)
    ]

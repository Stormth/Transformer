"""评测：在 WMT 官方测试集上算 BLEU / chrF，并对比不同解码策略。

常用命令：

    # 在 dev（newsdev2017）上评测
    python -m nmt.evaluate --checkpoint checkpoints/best.pt --split dev

    # 在 2019 年官方测试集上评测，并把译文存下来
    python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --save-pred preds/test2019.txt

    # 对比贪心 vs 不同宽度的束搜索
    python -m nmt.evaluate --checkpoint checkpoints/best.pt --split test2019 --compare-decoders

BLEU 只是"跟参考译文重合了多少 n-gram"，它不理解语义，也不惩罚"读起来不像人话"。
所以评测时建议同时看 chrF 和长度比，并且一定要把译文打印出来读几段。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from .bleu import evaluate_all
from .dataset import ParallelTextDataset
from .inference import Translator, select_eval_indices, translate_dataset
from .utils import Timer, ensure_dir, get_logger, human_time, read_lines, setup_console, write_lines

logger = get_logger()


def load_split(data_dir: Path, split: str) -> ParallelTextDataset:
    path = data_dir / f"{split}.pt"
    if not path.exists():
        available = sorted(p.stem for p in data_dir.glob("*.pt"))
        raise SystemExit(f"没有 {path}，可用的 split：{', '.join(available) or '（还没准备数据）'}")
    return ParallelTextDataset(path)


def run_once(
    translator: Translator,
    dataset: ParallelTextDataset,
    split: str,
    data_dir: Path,
    beam_size: int,
    max_sentences: int,
    batch_tokens: int = 4096,
) -> Tuple[Dict[str, float], List[str], List[int], List[str]]:
    indices = select_eval_indices(len(dataset), max_sentences)
    with Timer() as timer:
        hypotheses = translate_dataset(
            translator.model, dataset, translator.tokenizer, translator.device,
            indices=indices, beam_size=beam_size, max_tokens=batch_tokens,
            length_penalty=translator.length_penalty, max_len=translator.max_len,
        )
    references = read_lines(data_dir / f"{split}.ref.zh")
    references = [references[index] for index in indices]
    metrics = evaluate_all(hypotheses, references, lang="zh")
    metrics["seconds"] = timer.elapsed
    metrics["sentences"] = float(len(indices))
    metrics["beam_size"] = float(beam_size)
    return metrics, hypotheses, indices, references


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description="评测英译中模型")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-dir", default="data/ready")
    parser.add_argument("--split", default="dev", help="dev / test2017 / test2018 / test2019")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--max-sentences", type=int, default=0, help="只评这么多句（0=全部）")
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-pred", default=None, help="把译文写到文件")
    parser.add_argument("--show", type=int, default=5, help="打印前几句对照（0=不打印）")
    parser.add_argument(
        "--compare-decoders", action="store_true",
        help="对比贪心与 2/4/8 宽度的束搜索",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset = load_split(data_dir, args.split)
    translator = Translator(
        args.checkpoint, device=args.device,
        beam_size=max(1, args.beam_size), length_penalty=args.length_penalty,
    )
    logger.info(f"数据 {args.split}（{len(dataset)} 句）| 模型来自 {args.checkpoint}")

    if args.compare_decoders:
        logger.info("=" * 68)
        logger.info("解码策略对比（同一批句子，只换解码方式）")
        results: List[Dict[str, float]] = []
        for beam in [1, 2, 4, 8]:
            metrics, _, _, _ = run_once(
                translator, dataset, args.split, data_dir,
                beam_size=beam, max_sentences=args.max_sentences, batch_tokens=args.batch_tokens,
            )
            results.append(metrics)
            label = "贪心" if beam == 1 else f"束搜索 K={beam}"
            logger.info(
                f"{label:<12} BLEU {metrics['bleu']:6.2f} | chrF {metrics['chrf']:6.2f} | "
                f"长度比 {metrics['length_ratio']:.3f} | {metrics['seconds']:.1f}s"
            )
        best = max(results, key=lambda m: m["bleu"])
        best_label = "贪心" if best["beam_size"] == 1 else f"束搜索 K={int(best['beam_size'])}"
        logger.info(f"这批数据上最优：{best_label}（BLEU {best['bleu']:.2f}）")
        return

    metrics, hypotheses, indices, references = run_once(
        translator, dataset, args.split, data_dir,
        beam_size=max(1, args.beam_size), max_sentences=args.max_sentences,
        batch_tokens=args.batch_tokens,
    )

    logger.info("=" * 68)
    logger.info(f"split {args.split} | {int(metrics['sentences'])} 句 | 束宽 {int(metrics['beam_size'])}")
    logger.info(f"BLEU-4   {metrics['bleu']:.2f}")
    logger.info(f"chrF     {metrics['chrf']:.2f}")
    logger.info(f"长度比   {metrics['length_ratio']:.3f}（译文 token 数 / 参考 token 数）")
    logger.info(f"耗时     {human_time(metrics['seconds'])}")

    if args.save_pred:
        ensure_dir(Path(args.save_pred).parent)
        write_lines(args.save_pred, hypotheses)
        logger.info(f"译文已写入 {args.save_pred}")

    if args.show > 0:
        logger.info("=" * 68)
        logger.info(f"抽 {args.show} 句对照（英文原文 / 模型译文 / 参考译文）")
        step = max(1, len(hypotheses) // args.show)
        for i in range(0, len(hypotheses), step):
            source = dataset.decode_pair(indices[i], translator.tokenizer)[0]
            logger.info("-" * 68)
            logger.info(f"[{i}] 英文  {source}")
            logger.info(f"    模型  {hypotheses[i]}")
            logger.info(f"    参考  {references[i]}")


if __name__ == "__main__":
    main()

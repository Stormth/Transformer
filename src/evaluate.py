"""第 4 步：在测试集上评估，并把不同解码策略放在一起对比。

    python -m src.evaluate --checkpoint checkpoints/best.pt --split test
    python -m src.evaluate --checkpoint checkpoints/best.pt --compare-decoders

输出包括：
    * 语料级 BLEU-4；
    * 贪心 vs 束搜索（1/3/5）的质量与耗时对比；
    * 若干"好/坏"样例，方便你肉眼定位模型弱点（语序？形态？未登录词？）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence

from .bleu import corpus_bleu
from .data import read_tsv
from .inference import load_model, translate_batch_with_references


def evaluate(
    checkpoint: str,
    split: str = "test",
    data_dir: str = "data/processed",
    beam_size: int = 1,
    max_len: int = 64,
    limit: int = 0,
    show: int = 10,
    device: str = "auto",
    output: str | None = None,
):
    model, tokenizer, cfg, dev = load_model(checkpoint, None, device)
    pairs = read_tsv(Path(data_dir) / f"{split}.tsv")
    if limit:
        pairs = pairs[:limit]

    t0 = time.time()
    results = translate_batch_with_references(
        model, tokenizer, pairs, device=dev, beam_size=beam_size, max_len=max_len
    )
    elapsed = time.time() - t0

    hyps = [r.hypothesis for r in results]
    refs = [r.reference or "" for r in results]
    bleu = corpus_bleu(hyps, refs)

    print(f"文件: {Path(data_dir) / f'{split}.tsv'}（{len(pairs)} 句）  设备: {dev}")
    print(f"解码: {'贪心' if beam_size <= 1 else f'束搜索 beam={beam_size}'}  "
          f"耗时: {elapsed:.1f}s（{len(pairs) / max(elapsed, 1e-9):.1f} 句/秒）")
    print(f"BLEU-4: {bleu:.2f}")

    if show:
        pairs_hyps = list(zip(results, refs))
        print("\n---- 随机样例 ----")
        step = max(1, len(pairs_hyps) // show)
        for r, ref in pairs_hyps[::step][:show]:
            print(f"  源: {r.source}")
            print(f"  译: {r.hypothesis}")
            print(f"  参: {ref}")
            print()

    if output:
        lines = [f"{r.source}\t{r.hypothesis}\t{r.reference}" for r in results]
        Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"详细结果已写入 {output}")

    return bleu, results


def compare_decoders(
    checkpoint: str,
    data_dir: str = "data/processed",
    split: str = "test",
    limit: int = 300,
    beam_sizes: Sequence[int] = (1, 2, 4, 8),
    device: str = "auto",
):
    """同一条数据上比较不同解码策略，直观看到"质量 / 速度"的权衡。"""
    rows: List[tuple] = []
    for beam in beam_sizes:
        model, tokenizer, cfg, dev = load_model(checkpoint, None, device)
        pairs = read_tsv(Path(data_dir) / f"{split}.tsv")[:limit]
        t0 = time.time()
        results = translate_batch_with_references(
            model, tokenizer, pairs, device=dev, beam_size=beam, max_len=64
        )
        elapsed = time.time() - t0
        bleu = corpus_bleu([r.hypothesis for r in results], [r.reference or "" for r in results])
        rows.append((beam, bleu, elapsed, len(pairs) / max(elapsed, 1e-9)))
        print(f"beam={beam:<2} BLEU={bleu:6.2f}  耗时 {elapsed:6.1f}s  {rows[-1][3]:5.1f} 句/秒")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="评估翻译质量")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", default="test", choices=["test", "dev", "probe"])
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 句（0 = 全部）")
    parser.add_argument("--show", type=int, default=10, help="打印多少条样例")
    parser.add_argument("--output", default=None, help="把逐句结果写到文件")
    parser.add_argument("--compare-decoders", action="store_true", help="对比贪心与不同 beam 大小")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if args.compare_decoders:
        compare_decoders(args.checkpoint, args.data_dir, args.split, device=args.device)
        return

    evaluate(
        args.checkpoint,
        split=args.split,
        data_dir=args.data_dir,
        beam_size=args.beam_size,
        max_len=args.max_len,
        limit=args.limit,
        show=args.show,
        device=args.device,
        output=args.output,
    )


if __name__ == "__main__":
    main()

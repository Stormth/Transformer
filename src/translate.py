"""第 3 步：使用模型翻译（命令行工具）。

    # 单句
    python -m src.translate --checkpoint checkpoints/best.pt --text "我明天想去图书馆看书。"

    # 交互模式（输入空行退出）
    python -m src.translate --checkpoint checkpoints/best.pt --interactive

    # 批量翻译文件（每行一句中文），结果写到 out.txt
    python -m src.translate --checkpoint checkpoints/best.pt --file input.txt --output out.txt

    # 用束搜索（质量更好，速度更慢）
    python -m src.translate --checkpoint checkpoints/best.pt --beam-size 5 --text "他很忙。"

对比实验建议：
    --beam-size 1  = 贪心解码（每步只选最优）
    --beam-size 5  = 束搜索（保留 5 条候选，用整句分数挑选）
    你会看到束搜索在"语序/形态"这类长距离决策上明显更稳。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from .data import read_tsv
from .inference import load_model, translate_batch_with_references, translate_texts


def translate_lines(
    checkpoint: str | Path,
    texts: Sequence[str],
    beam_size: int = 4,
    max_len: int = 64,
    batch_size: int = 64,
    device: str = "auto",
    tokenizer_path: str | Path | None = None,
) -> List[str]:
    """对外暴露的最简 API：给一批中文，返回一批英文。"""
    model, tokenizer, cfg, dev = load_model(checkpoint, tokenizer_path, device)
    outputs = translate_texts(
        model, tokenizer, list(texts), device=dev, beam_size=beam_size,
        max_len=max_len, batch_size=batch_size,
    )
    return [text for text, _ in outputs]


def _print_result(src: str, hyp: str, score, ref: str | None = None) -> None:
    print(f"  源句: {src}")
    print(f"  译文: {hyp}")
    if ref is not None:
        print(f"  参考: {ref}")
    if score is not None:
        print(f"  束分数: {score:.3f}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="用训练好的模型翻译")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="best.pt / last.pt 或所在目录")
    parser.add_argument("--tokenizer", default=None, help="默认用 checkpoint 同目录的 tokenizer.json")
    parser.add_argument("--text", default=None, help="直接翻译一句话")
    parser.add_argument("--file", default=None, help="每行一句的输入文件")
    parser.add_argument("--output", default=None, help="结果输出文件")
    parser.add_argument("--probe", default=None, help="翻译带参考译文的 TSV（用于对照）")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--beam-size", type=int, default=4, help="1 = 贪心解码")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    model, tokenizer, cfg, device = load_model(args.checkpoint, args.tokenizer, args.device)
    print(f"已加载模型：{args.checkpoint} | 设备 {device} | 词表 {len(tokenizer)} | beam={args.beam_size}")

    if args.text:
        hyp, score = translate_texts(
            model, tokenizer, [args.text], device=device, beam_size=args.beam_size,
            max_len=args.max_len, batch_size=1,
        )[0]
        _print_result(args.text, hyp, score)
        return

    if args.file:
        texts = [
            line.strip()
            for line in Path(args.file).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        outputs = translate_texts(
            model, tokenizer, texts, device=device, beam_size=args.beam_size,
            max_len=args.max_len, batch_size=args.batch_size,
        )
        lines = [f"{src}\t{hyp}" for src, (hyp, _) in zip(texts, outputs)]
        if args.output:
            Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"已写入 {args.output}（{len(lines)} 行）")
        else:
            print("\n".join(lines))
        return

    if args.probe:
        pairs = read_tsv(args.probe)
        results = translate_batch_with_references(
            model, tokenizer, pairs, device=device, beam_size=args.beam_size,
            max_len=args.max_len, batch_size=args.batch_size,
        )
        for r in results:
            _print_result(r.source, r.hypothesis, r.score, r.reference)
        return

    if args.interactive:
        print("输入中文句子，回车翻译；直接回车退出。")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            hyp, score = translate_texts(
                model, tokenizer, [line], device=device, beam_size=args.beam_size,
                max_len=args.max_len, batch_size=1,
            )[0]
            _print_result(line, hyp, score)
        return

    parser.error("请指定 --text / --file / --probe / --interactive 之一")


if __name__ == "__main__":
    main()

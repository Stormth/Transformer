"""命令行翻译：训练好的模型怎么"用"。

    # 翻一句
    python -m nmt.translate --checkpoint checkpoints/best.pt --text "The meeting was postponed."

    # 翻多句 / 翻一个文件
    python -m nmt.translate --checkpoint checkpoints/best.pt --text "..." --text "..."
    python -m nmt.translate --checkpoint checkpoints/best.pt --file news.en --out news.de

    # 交互模式：不带 --text/--file，直接从键盘输入，一行一句
    python -m nmt.translate --checkpoint checkpoints/best.pt

几个常用开关：
    --beam-size 1    贪心，最快，适合看模型"第一反应"
    --beam-size 4    束搜索，通常能涨 1~2 BLEU
    --show-tokens    把分词结果打出来，观察 BPE 到底把句子切成了什么
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .inference import Translator
from .utils import ensure_dir, human_time, read_lines, setup_console, write_lines, Timer, get_logger

logger = get_logger()


def translate_file(
    translator: Translator,
    inputs: List[str],
    batch_size: int,
    beam_size: int,
    show_tokens: bool,
) -> List[str]:
    if show_tokens:
        for text in inputs[:3]:
            logger.info(f"分词示例：{translator.tokenizer.tokenize(text)[:24]}")
    with Timer() as timer:
        outputs = translator.translate(inputs, batch_size=batch_size, beam_size=beam_size)
    logger.info(
        f"译完 {len(inputs)} 句，耗时 {human_time(timer.elapsed)}"
        f"（{len(inputs) / max(1e-6, timer.elapsed):.1f} 句/秒）"
    )
    return outputs


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description="英译中命令行翻译")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--text", action="append", default=[], help="要翻译的句子，可重复")
    parser.add_argument("--file", default=None, help="每行一句的输入文件")
    parser.add_argument("--out", default=None, help="译文输出文件")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show-tokens", action="store_true", help="打印 BPE 分词结果")
    parser.add_argument("--quiet", action="store_true", help="只输出译文")
    args = parser.parse_args()

    translator = Translator(
        args.checkpoint, device=args.device, beam_size=args.beam_size,
        length_penalty=args.length_penalty, max_len=args.max_len,
    )
    if not args.quiet:
        logger.info(translator.describe())

    # --- 交互模式 ---
    if not args.text and not args.file:
        if not args.quiet:
            logger.info("交互模式：输入英文后回车，空行退出")
        for line in sys.stdin:
            text = line.strip()
            if not text:
                break
            print(translator.translate_one(text, beam_size=args.beam_size), flush=True)
        return

    inputs: List[str] = list(args.text)
    if args.file:
        inputs.extend(read_lines(args.file))
    if not inputs:
        raise SystemExit("没有输入句子")

    outputs = translate_file(translator, inputs, args.batch_size, args.beam_size, args.show_tokens)

    if args.out:
        ensure_dir(Path(args.out).parent)
        write_lines(args.out, outputs)
        if not args.quiet:
            logger.info(f"译文已写入 {args.out}")
    for source, target in zip(inputs, outputs):
        if args.quiet:
            print(target)
        else:
            print(f"  英文  {source}")
            print(f"  德文  {target}")
            print()


if __name__ == "__main__":
    main()

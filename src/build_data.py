"""第 1 步：准备数据 —— 生成平行语料、训练 BPE、切分数据集。

用法
----
    python -m src.build_data                    # 生成默认规模语料
    python -m src.build_data --num-pairs 40000  # 换规模
    python -m src.build_data --include-natural  # 把人工语料也混进训练集

产出（默认写到 data/processed/）
    train.tsv / dev.tsv / test.tsv : 平行语料（中文 TAB 英文）
    probe.tsv                      : 人工口语测试句（不参与训练，只用来肉眼观察）
    tokenizer.json                 : 训练好的 BPE（merges + 词表）
    meta.json                      : 数据统计信息，训练脚本会读取

重要原则：**只在训练集上训练分词器**。如果在全量数据上训练 BPE，
测试集的词汇信息会通过词表"泄漏"进模型，你看到的指标就是虚高的。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Sequence

from . import synth
from .bpe import BPE, PAD_ID, SPECIAL_TOKENS, UNK_ID
from .data import Pair, read_tsv, write_tsv


def _length_stats(pairs: Sequence[Pair], tokenizer: BPE) -> dict:
    src_lens, tgt_lens = [], []
    unk = Counter()
    for src_text, tgt_text in pairs:
        src_ids = tokenizer.encode(src_text)
        tgt_ids = tokenizer.encode(tgt_text)
        src_lens.append(len(src_ids))
        tgt_lens.append(len(tgt_ids))
        unk["src_unk"] += sum(1 for i in src_ids if i == UNK_ID)
        unk["tgt_unk"] += sum(1 for i in tgt_ids if i == UNK_ID)
        unk["src_tokens"] += len(src_ids)
        unk["tgt_tokens"] += len(tgt_ids)

    def q(xs: List[int], p: float) -> float:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0

    return {
        "src_len_mean": round(sum(src_lens) / len(src_lens), 2),
        "tgt_len_mean": round(sum(tgt_lens) / len(tgt_lens), 2),
        "src_len_p50": q(src_lens, 0.5),
        "src_len_p95": q(src_lens, 0.95),
        "src_len_max": max(src_lens),
        "tgt_len_p50": q(tgt_lens, 0.5),
        "tgt_len_p95": q(tgt_lens, 0.95),
        "tgt_len_max": max(tgt_lens),
        "src_unk_rate": round(unk["src_unk"] / max(1, unk["src_tokens"]), 6),
        "tgt_unk_rate": round(unk["tgt_unk"] / max(1, unk["tgt_tokens"]), 6),
    }


def build(
    out_dir: str | Path = "data/processed",
    num_pairs: int = 24000,
    dev_size: int = 1000,
    test_size: int = 1000,
    vocab_size: int = 4000,
    seed: int = 2024,
    include_natural: bool = False,
    natural_path: str | Path = "data/natural/probe.tsv",
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 生成平行语料 ------------------------------------------- #
    pairs = synth.generate_pairs(num_pairs, seed=seed)
    train, dev, test = synth.split_pairs(pairs, dev_size=dev_size, test_size=test_size, seed=seed)

    natural: List[Pair] = []
    natural_path = Path(natural_path)
    if natural_path.exists():
        natural = read_tsv(natural_path)

    train_with_natural = list(train)
    if include_natural and natural:
        seen = set(train)
        train_with_natural += [p for p in natural if p not in seen]

    # 人工语料永远单独留一份作为"肉眼观察集"，不参与训练
    probe = natural

    # ---- 2. 训练 BPE（只在训练集上！）------------------------------ #
    train_texts: List[str] = []
    for src, tgt in train_with_natural:
        train_texts.append(src)
        train_texts.append(tgt)
    if verbose:
        print(f"[data] 训练句对 {len(train_with_natural)}，验证 {len(dev)}，测试 {len(test)}"
              f"（含人工语料 {len(natural)} 条的观察集单独保存）")
        print(f"[data] 开始训练 BPE（目标词表 {vocab_size}）...")
    tokenizer = BPE.train(train_texts, vocab_size=vocab_size, min_pair_freq=2, verbose=verbose)

    # ---- 3. 落盘 --------------------------------------------------- #
    write_tsv(train_with_natural, out_dir / "train.tsv")
    write_tsv(dev, out_dir / "dev.tsv")
    write_tsv(test, out_dir / "test.tsv")
    if probe:
        write_tsv(probe, out_dir / "probe.tsv")
    tokenizer.save(out_dir / "tokenizer.json")

    meta = {
        "vocab_size": len(tokenizer),
        "pad_id": PAD_ID,
        "special_tokens": list(SPECIAL_TOKENS),
        "train_size": len(train_with_natural),
        "dev_size": len(dev),
        "test_size": len(test),
        "probe_size": len(probe),
        "synthetic_pairs": num_pairs,
        "include_natural": include_natural,
        "seed": seed,
        "train_stats": _length_stats(train_with_natural, tokenizer),
        "test_stats": _length_stats(test, tokenizer),
        "probe_stats": _length_stats(probe, tokenizer) if probe else {},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print("\n================= 数据准备完成 =================")
        print(f"词表大小        : {len(tokenizer)}")
        print(f"训练/验证/测试  : {meta['train_size']} / {meta['dev_size']} / {meta['test_size']}")
        ts = meta["train_stats"]
        print(f"源句平均长度    : {ts['src_len_mean']} tokens（p95={ts['src_len_p95']}，max={ts['src_len_max']}）")
        print(f"目标句平均长度  : {ts['tgt_len_mean']} tokens（p95={ts['tgt_len_p95']}，max={ts['tgt_len_max']}）")
        print(f"测试集 <unk> 率 : 源 {meta['test_stats']['src_unk_rate']:.2%} / 目标 {meta['test_stats']['tgt_unk_rate']:.2%}")
        if probe:
            ps = meta["probe_stats"]
            print(f"人工句集 <unk> 率: 目标 {ps['tgt_unk_rate']:.2%}"
                  f"（这些是模板外的真实口语，可用来观察子词切分与未登录词）")
        print(f"产物目录        : {out_dir.resolve()}")
        print("下一步：python -m src.train")
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成平行语料并训练 BPE 分词器")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--num-pairs", type=int, default=24000, help="合成语料句对数")
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--vocab-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--include-natural", action="store_true", help="把人工语料也加入训练集")
    parser.add_argument("--natural-path", default="data/natural/probe.tsv")
    args = parser.parse_args(argv)

    build(
        out_dir=args.out_dir,
        num_pairs=args.num_pairs,
        dev_size=args.dev_size,
        test_size=args.test_size,
        vocab_size=args.vocab_size,
        seed=args.seed,
        include_natural=args.include_natural,
        natural_path=args.natural_path,
    )


if __name__ == "__main__":
    main()

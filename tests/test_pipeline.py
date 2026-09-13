"""端到端冒烟测试：造数据 -> 训练几步 -> 翻译，确认整条链路能跑通。

它不联网、不用真实语料，只验证"代码没写错"。
真实翻译质量和 BLEU 要用 WMT 数据单独评测。
"""

from __future__ import annotations

import torch

from nmt.bpe import BPE
from nmt.config import tiny_config
from nmt.corpus import encode_split, save_split
from nmt.inference import Translator
from nmt.synth import toy_pairs
from nmt.train import train


def _prepare(data_dir, vocab_size: int = 400) -> None:
    pairs = toy_pairs(240, seed=11)
    train_pairs, dev_pairs = pairs[:200], pairs[200:]

    tokenizer = BPE.train(
        [text for pair in train_pairs for text in pair], vocab_size=vocab_size, min_frequency=1
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(data_dir / "vocab.json")

    for name, split in (("train", train_pairs), ("dev", dev_pairs)):
        result = encode_split(split, tokenizer, 64, 64)
        save_split(data_dir / f"{name}.pt", *result[:4], result[5])


def test_training_and_translation_end_to_end(tmp_path) -> None:
    data_dir = tmp_path / "ready"
    save_dir = tmp_path / "ckpt"
    _prepare(data_dir)

    config = tiny_config(vocab_size=400)
    config.train.epochs = 40
    config.train.warmup_steps = 30
    config.train.lr_scale = 0.05     # warmup 调短会让 Noam 峰值变大，这里压回来
    config.train.log_every_steps = 200
    config.train.eval_max_sentences = 20
    config.train.amp = False
    config.train.device = "cpu"
    config.train.seed = 0

    summary = train(config, data_dir, save_dir, max_steps=300)
    assert summary["steps"] > 0
    assert (save_dir / "best.pt").exists()
    assert (save_dir / "last.pt").exists()
    assert (save_dir / "train_log.csv").exists()

    # 词表被复制到了 checkpoint 目录，方便整个目录搬走
    assert (save_dir / "vocab.json").exists()

    translator = Translator(save_dir / "best.pt", device="cpu", beam_size=1)
    output = translator.translate_one("he will finish the project tomorrow at school.")
    assert isinstance(output, str)
    assert len(output) > 0
    # 结果应该是中文，而不是把英文照抄回来
    assert any("\u4e00" <= ch <= "\u9fff" for ch in output)


def test_resume_continues_from_checkpoint(tmp_path) -> None:
    data_dir = tmp_path / "ready"
    save_dir = tmp_path / "ckpt"
    _prepare(data_dir, vocab_size=300)

    config = tiny_config(vocab_size=300)
    config.train.epochs = 4
    config.train.warmup_steps = 20
    config.train.eval_max_sentences = 10
    config.train.log_every_steps = 100
    config.train.amp = False
    config.train.device = "cpu"

    first = train(config, data_dir, save_dir, max_steps=20)
    second = train(config, data_dir, save_dir, resume="auto", max_steps=40)

    assert second["steps"] >= first["steps"]
    from nmt.checkpoint import load_checkpoint

    checkpoint = load_checkpoint(save_dir / "last.pt")
    assert checkpoint.optimizer is not None     # 续训必须带上优化器状态
    assert checkpoint.scheduler is not None
    assert checkpoint.epoch >= 1

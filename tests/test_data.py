"""数据集与批处理测试：padding、动态 batch、样本不重不漏。"""

from __future__ import annotations

import pytest
import torch

from nmt.corpus import align_parallel_lines, encode_split, read_moses_pairs, save_split
from nmt.dataset import LengthBucketSampler, ParallelTextDataset, collate_batch
from nmt.synth import toy_pairs


def _write_split(path, tokenizer, pairs, max_len: int = 64) -> None:
    result = encode_split(pairs, tokenizer, max_len, max_len)
    save_split(path, *result[:4], result[5])


def test_align_is_identity_when_lengths_match() -> None:
    en = ["Hello world.", "How are you today?"]
    de = ["Hallo Welt.", "Wie geht es dir heute?"]
    assert align_parallel_lines(en, de) == list(zip(en, de))


def test_align_repairs_sentence_split_across_lines() -> None:
    """真实语料里偶尔有句子被换行拆开（本项目用的 de-en 包里有 5 处）。

    如果不修，从错位点开始后面**所有**句对都会错位 —— 这是最隐蔽的数据 bug 之一。
    """

    en = ["Hello world.", "How are you today?", "Fine."]
    de = ["Hallo Welt.", "Wie geht es dir", "heute?", "Gut."]
    pairs = align_parallel_lines(en, de)

    assert len(pairs) == len(en)
    assert pairs[1] == ("How are you today?", "Wie geht es dir heute?")
    assert pairs[2] == ("Fine.", "Gut.")


def test_align_truncates_on_large_mismatch() -> None:
    """行数差得太多（超过修复上限）时按较短一侧截断，而不是硬对齐或报错。

    WMT 的官方语料就是这样：News Commentary v9 两侧差 141 行，
    而且错位量在文件里来回摆动（不是单调累积），没有便宜的修法。
    工业上的常规做法就是截断：损失 0.1% 的数据，省掉一个真正的句子对齐器。
    """

    en = ["one", "two"]
    de = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    pairs = align_parallel_lines(en, de)
    assert len(pairs) == len(en)
    assert pairs[0] == ("one", "a")


def test_read_pairs_does_not_split_on_unicode_line_separator(tmp_path) -> None:
    """U+2028 不是换行符。

    str.splitlines() 会把 U+2028、\\x0c 这些也当换行，于是句子中间被切开、
    两侧行数凭空对不上。WMT 的 News Commentary v9 里就有 7 处 U+2028，
    这个坑会直接导致几万句错位。
    """

    (tmp_path / "pair.en").write_text("Hello\u2028world\nSecond line\n", encoding="utf-8")
    (tmp_path / "pair.de").write_text("Hallo\u2028Welt\nZweite Zeile\n", encoding="utf-8")
    pairs = read_moses_pairs(tmp_path)
    assert len(pairs) == 2
    assert pairs[0][0] == "Hello\u2028world"
    assert pairs[1] == ("Second line", "Zweite Zeile")


def test_collate_pads_to_batch_max() -> None:
    samples = [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 9, 2])),
        (torch.tensor([1, 2]), torch.tensor([1, 2])),
    ]
    batch = collate_batch(samples, pad_id=0)
    assert batch["src"].shape == (2, 3)
    assert batch["tgt"].shape == (2, 3)
    assert batch["src_lengths"].tolist() == [3, 2]
    # 短的那句后面补的是 <pad>
    assert batch["src"][1].tolist() == [1, 2, 0]


def test_dataset_slices_match_original(tokenizer, tmp_path) -> None:
    pairs = toy_pairs(20, seed=1)
    path = tmp_path / "toy.pt"
    _write_split(path, tokenizer, pairs)
    dataset = ParallelTextDataset(path)

    assert len(dataset) == len(pairs)
    src_ids, tgt_ids = dataset[3]
    assert tokenizer.decode(src_ids.tolist()) == pairs[3][0]
    assert tokenizer.decode(tgt_ids.tolist()) == pairs[3][1]
    # 目标句带 <bos> 和 <eos>
    assert int(tgt_ids[0]) == tokenizer.bos_id
    assert int(tgt_ids[-1]) == tokenizer.eos_id


def test_sampler_covers_every_sample_once(tokenizer, tmp_path) -> None:
    pairs = toy_pairs(200, seed=2)
    path = tmp_path / "toy.pt"
    _write_split(path, tokenizer, pairs)
    dataset = ParallelTextDataset(path)
    sampler = LengthBucketSampler(
        dataset, max_tokens=400, max_sentences=16, bucket_size=8, shuffle=True, seed=0
    )

    seen = [index for batch in sampler for index in batch]
    assert sorted(seen) == list(range(len(dataset)))
    assert len(sampler) == len(list(iter(sampler)))


def test_sampler_respects_token_budget(tokenizer, tmp_path) -> None:
    """一个 batch 的 padding 之后总 token 数不应该超过 max_tokens 太多。"""

    pairs = toy_pairs(120, seed=3)
    path = tmp_path / "toy.pt"
    _write_split(path, tokenizer, pairs)
    dataset = ParallelTextDataset(path)
    sampler = LengthBucketSampler(dataset, max_tokens=256, max_sentences=64, bucket_size=16)

    for batch in sampler:
        lengths = [sum(dataset.length(i)) for i in batch]
        padded = max(lengths) * len(batch)
        assert padded <= 256 or len(batch) == 1


def test_sampler_orders_by_length_inside_batch(tokenizer, tmp_path) -> None:
    """动态 batch 的意义就是同批长度接近；这里确认最长的两句不会差太多。"""

    pairs = toy_pairs(300, seed=4)
    path = tmp_path / "toy.pt"
    _write_split(path, tokenizer, pairs)
    dataset = ParallelTextDataset(path)
    sampler = LengthBucketSampler(dataset, max_tokens=1024, max_sentences=32, bucket_size=32)

    for batch in sampler:
        lengths = [sum(dataset.length(i)) for i in batch]
        assert max(lengths) <= 4 * max(1, min(lengths)) + 8

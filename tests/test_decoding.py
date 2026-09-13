"""解码测试：贪心与束搜索要产出自洽的结果。"""

from __future__ import annotations

import torch

from nmt.decoding import beam_search_decode, greedy_decode
from nmt.masks import make_encoder_attn_mask


def test_greedy_returns_valid_sequences(model, tokenizer) -> None:
    src = torch.tensor([[1, 5, 6, 2], [1, 7, 2, 0]])
    mask = make_encoder_attn_mask(src, 0)
    sequences = greedy_decode(
        model, src, mask,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id, max_len=12,
    )
    assert len(sequences) == 2
    for sequence in sequences:
        assert all(0 <= token < len(tokenizer) for token in sequence)
        assert tokenizer.eos_id not in sequence   # 已经在 <eos> 处截断


def test_greedy_respects_max_len(model, tokenizer) -> None:
    src = torch.tensor([[1, 5, 6, 2]])
    mask = make_encoder_attn_mask(src, 0)
    sequences = greedy_decode(
        model, src, mask,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id, max_len=4,
    )
    assert len(sequences[0]) <= 4


def test_beam_size_one_equals_greedy(model, tokenizer) -> None:
    src = torch.tensor([[1, 5, 6, 2]])
    mask = make_encoder_attn_mask(src, 0)
    greedy = greedy_decode(
        model, src, mask,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id, max_len=10,
    )
    beam, _ = beam_search_decode(
        model, src, mask,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id, beam_size=1, max_len=10,
    )
    assert greedy == beam


def test_beam_search_returns_one_sequence_per_sentence(model, tokenizer) -> None:
    src = torch.tensor([[1, 5, 6, 2], [1, 7, 2, 0]])
    mask = make_encoder_attn_mask(src, 0)
    sequences, scores = beam_search_decode(
        model, src, mask,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id, beam_size=3, max_len=10,
    )
    assert len(sequences) == 2
    assert len(scores) == 2
    for sequence in sequences:
        assert all(0 <= token < len(tokenizer) for token in sequence)


def test_kv_cache_is_used(model, tokenizer) -> None:
    """beam_size>1 时每步都要重排 cache；这里确认它不会崩、形状也对。"""

    src = torch.tensor([[1, 5, 6, 2]])
    mask = make_encoder_attn_mask(src, 0)
    sequences, _ = beam_search_decode(
        model, src, mask,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id, beam_size=4, max_len=15,
    )
    assert len(sequences) == 1

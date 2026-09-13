"""掩码测试：这是 Transformer 最容易写错的地方，所以测得细一点。"""

from __future__ import annotations

import torch

from nmt.masks import (
    make_causal_mask,
    make_cross_attn_mask,
    make_decoder_self_attn_mask,
    make_encoder_attn_mask,
    make_incremental_self_attn_mask,
    make_pad_mask,
)


def test_causal_mask_is_strict_upper_triangle() -> None:
    mask = make_causal_mask(4, torch.device("cpu"))
    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected)
    # 每个位置至少能看到自己，否则 softmax 会出现整行被屏蔽
    assert not mask.diagonal().any()


def test_pad_mask() -> None:
    tokens = torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]])
    mask = make_pad_mask(tokens, pad_id=0)
    assert torch.equal(
        mask, torch.tensor([[False, False, True, True], [False, True, True, True]])
    )


def test_encoder_mask_shape_and_content() -> None:
    tokens = torch.tensor([[5, 6, 0]])
    mask = make_encoder_attn_mask(tokens, pad_id=0)
    assert mask.shape == (1, 1, 1, 3)
    assert torch.equal(mask.flatten(), torch.tensor([False, False, True]))


def test_decoder_mask_combines_causal_and_padding() -> None:
    tokens = torch.tensor([[5, 6, 0]])
    mask = make_decoder_self_attn_mask(tokens, pad_id=0)
    assert mask.shape == (1, 1, 3, 3)
    # 第 0 个位置只能看自己，且第 2 个位置是 padding
    assert mask[0, 0, 0].tolist() == [False, True, True]
    # 最后一行是 padding 位置（作为 query 没人关心），但它仍然不许看 padding key
    assert mask[0, 0, 2].tolist() == [False, False, True]


def test_cross_mask_equals_encoder_mask() -> None:
    tokens = torch.tensor([[5, 6, 0, 0]])
    assert torch.equal(
        make_cross_attn_mask(tokens, 0), make_encoder_attn_mask(tokens, 0)
    )


def test_incremental_mask_has_no_causal_part() -> None:
    """增量解码时 query 只有一个位置，因果性由"只有一步"天然保证。"""

    prefix = torch.tensor([[1, 5, 6, 0]])
    mask = make_incremental_self_attn_mask(prefix, pad_id=0)
    assert mask.shape == (1, 1, 1, 4)
    assert torch.equal(mask.flatten(), torch.tensor([False, False, False, True]))

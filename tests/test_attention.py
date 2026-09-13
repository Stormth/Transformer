"""注意力测试：形状、归一化、掩码、KV cache 一致性。"""

from __future__ import annotations

import torch

from nmt.attention import MultiHeadAttention, mask_value, scaled_dot_product_attention


def test_scaled_dot_product_attention_shapes() -> None:
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)
    out, weights = scaled_dot_product_attention(q, k, v, return_weights=True)
    assert out.shape == (2, 4, 5, 8)
    assert weights.shape == (2, 4, 5, 6)


def test_attention_weights_are_probabilities() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    _, weights = scaled_dot_product_attention(q, k, v, return_weights=True)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 2, 4), atol=1e-5)
    assert (weights >= 0).all()


def test_mask_blocks_positions() -> None:
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    mask[:, :, :, 2] = True  # 谁都不许看第 2 个位置
    _, weights = scaled_dot_product_attention(q, k, v, mask=mask, return_weights=True)
    assert torch.allclose(weights[:, :, :, 2], torch.zeros_like(weights[:, :, :, 2]))


def test_mask_value_is_safe_for_float16() -> None:
    """-1e9 在 float16 里会溢出成 -inf，整行被屏蔽时就会变成 NaN。"""

    assert mask_value(torch.float16) >= torch.finfo(torch.float16).min
    assert mask_value(torch.float32) >= torch.finfo(torch.float32).min


def test_multi_head_attention_roundtrip_shape() -> None:
    attention = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0).eval()
    x = torch.randn(2, 7, 32)
    out, _ = attention(x, x, x)
    assert out.shape == x.shape


def test_multi_head_attention_causal_property() -> None:
    attention = MultiHeadAttention(d_model=16, n_heads=2, dropout=0.0).eval()
    x = torch.randn(1, 6, 16)
    causal = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)[None, None]
    _, weights = attention(x, x, x, mask=causal, return_weights=True)
    upper = torch.triu(weights[0, 0], diagonal=1)
    assert upper.abs().max().item() < 1e-6


def test_kv_cache_matches_full_attention() -> None:
    """增量解码 + cache 的结果必须和一次算完整句完全一致。

    注意全句前向要带因果掩码，否则第 0 个位置会去看后面的 key，
    和增量解码（只能看自己）自然对不上。
    """

    torch.manual_seed(0)
    attention = MultiHeadAttention(d_model=16, n_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 5, 16)
    causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)[None, None]

    with torch.no_grad():
        full, _ = attention(x, x, x, mask=causal)
        cache = {"k": None, "v": None}
        steps = []
        for position in range(x.size(1)):
            out, _ = attention(
                x[:, position : position + 1], x[:, position : position + 1],
                x[:, position : position + 1], cache=cache, cache_append=True,
            )
            steps.append(out)
        incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-5)

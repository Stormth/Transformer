"""模型整体测试：形状、因果性、增量解码一致性、权重共享、梯度能不能传。"""

from __future__ import annotations

import pytest
import torch

from nmt.config import tiny_config
from nmt.model import Transformer


def test_forward_shapes(model: Transformer) -> None:
    src = torch.tensor([[1, 5, 6, 7, 2, 0]])
    tgt = torch.tensor([[1, 8, 9, 2, 0]])
    src_mask, tgt_mask = model.make_masks(src, tgt, 0)
    logits = model(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (1, 5, model.config.tgt_vocab_size)


def test_encoder_decoder_shapes(model: Transformer) -> None:
    src = torch.tensor([[1, 5, 6, 2]])
    tgt = torch.tensor([[1, 7, 2]])
    src_mask, tgt_mask = model.make_masks(src, tgt, 0)
    memory, _ = model.encode(src, src_mask)
    hidden, _ = model.decode(tgt, memory, tgt_mask, src_mask)
    assert memory.shape == (1, 4, model.config.d_model)
    assert hidden.shape == (1, 3, model.config.d_model)


def test_incremental_decoding_matches_full_forward(model: Transformer) -> None:
    """带 KV cache 的逐步解码，必须和整句一次前向数值一致。"""

    from nmt.masks import make_incremental_self_attn_mask

    src = torch.tensor([[1, 5, 6, 7, 2]])
    tgt = torch.tensor([[1, 8, 9, 10, 2]])
    src_mask, tgt_mask = model.make_masks(src, tgt, 0)

    with torch.no_grad():
        memory, _ = model.encode(src, src_mask)
        full, _ = model.decode(tgt, memory, tgt_mask, src_mask)

        self_caches, cross_caches = model.make_decoder_caches()
        steps = []
        for position in range(tgt.size(1)):
            prefix = tgt[:, : position + 1]
            out, _ = model.decode(
                tgt[:, position : position + 1],
                memory,
                make_incremental_self_attn_mask(prefix, 0),
                src_mask,
                caches=self_caches,
                cross_caches=cross_caches,
                position_offset=position,
            )
            steps.append(out)
        incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-5)


def test_decoder_is_causal(model: Transformer) -> None:
    """改后面的 token 不应该影响前面位置的输出。"""

    src = torch.tensor([[1, 5, 6, 2]])
    src_mask, _ = model.make_masks(src, src, 0)
    tgt = torch.tensor([[1, 8, 9, 10, 2]])
    _, tgt_mask = model.make_masks(src, tgt, 0)

    model.eval()
    with torch.no_grad():
        memory, _ = model.encode(src, src_mask)
        first, _ = model.decode(tgt, memory, tgt_mask, src_mask)

        changed = tgt.clone()
        changed[0, 3] = 11  # 改第 4 个位置
        _, changed_mask = model.make_masks(src, changed, 0)
        second, _ = model.decode(changed, memory, changed_mask, src_mask)

    # 前 3 个位置必须一模一样
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-5)
    # 第 4 个位置应该变了（否则说明掩码把信息全挡住了）
    assert not torch.allclose(first[:, 3], second[:, 3], atol=1e-6)


def test_tie_embeddings_share_storage(model: Transformer) -> None:
    assert model.config.tie_embeddings
    assert model.generator.weight.data_ptr() == model.tgt_embedding.embedding.weight.data_ptr()


def test_gradients_flow(model: Transformer) -> None:
    src = torch.tensor([[1, 5, 6, 2]])
    tgt = torch.tensor([[1, 8, 9, 2]])
    src_mask, tgt_mask = model.make_masks(src, tgt, 0)
    logits = model(src, tgt, src_mask, tgt_mask)
    loss = logits.sum()
    loss.backward()
    # 编码器和解码器的第一层权重都该拿到梯度
    assert model.encoder.layers[0].self_attn.w_q.weight.grad is not None
    assert model.decoder.layers[0].cross_attn.w_o.weight.grad is not None
    assert model.encoder.layers[0].self_attn.w_q.weight.grad.abs().sum() > 0


def test_pre_norm_variant_runs(tokenizer) -> None:
    config = tiny_config(vocab_size=len(tokenizer))
    config.model.norm_first = True
    model = Transformer(config.model).eval()
    src = torch.tensor([[1, 5, 2]])
    tgt = torch.tensor([[1, 8, 2]])
    src_mask, tgt_mask = model.make_masks(src, tgt, 0)
    logits = model(src, tgt, src_mask, tgt_mask)
    assert torch.isfinite(logits).all()


def test_describe_mentions_key_numbers(model: Transformer) -> None:
    text = model.describe()
    assert "d_model" in text
    assert "编码器" in text and "解码器" in text

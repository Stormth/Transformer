"""和 PyTorch 官方 nn.Transformer 逐层对齐：验证"我到底写对没有"。

手写实现最容易出的问题是"能跑、loss 也降，但细节偷偷错了"：
掩码方向反了、残差接错位置、pre/post-norm 弄混、交叉注意力用错了张量。
这些错误不会崩，只会让效果差一点，肉眼很难发现。

最硬的验证方式：

    1. 造一个官方 nn.Transformer，结构和我们完全一致；
    2. 把我们的权重一项一项拷过去（Q/K/V 要拼成官方的 in_proj_weight）；
    3. 喂同样的输入，逐层比较编码器和解码器的输出；
    4. 差异应该只有浮点误差量级（float64 下 1e-15，float32 下 1e-6）。

两个必须知道的细节：

* 官方 nn.Transformer **不含词嵌入和位置编码**，那部分要自己写。
  所以这里把我们算好的"词嵌入 + 位置编码"喂给官方模型，比的是编码器/解码器本身。

* 别期望"完全零差异"。官方在推理模式下有几条融合快速路径：带 key_padding_mask 时
  会走 nested tensor（把 padding 位置的输出直接填 0），部分组合还会走 C++ 融合算子
  （内部按 float32 算）。结果是**官方自己两条路径在 float64 下就能差 7e-7**，
  带掩码时甚至差 3.0。所以：

      - 我们和官方逐层调用之间的差异，相对误差在 1e-6 量级 → 这是融合算子的精度地板；
      - 如果你真的写错了（掩码方向反、残差接错、交叉注意力喂错张量），
        差异会是 0.1 甚至 1 的量级，高下立判。

  这条"知道精度地板在哪"的经验，比记住"应该完全相等"有用得多。
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .config import PRESETS, Config, tiny_config
from .masks import make_decoder_self_attn_mask, make_encoder_attn_mask
from .model import Transformer
from .utils import get_logger, set_seed, setup_console

logger = get_logger()


# --------------------------------------------------------------------------
# 造参照模型 + 搬权重
# --------------------------------------------------------------------------
def build_reference(config: Config) -> nn.Transformer:
    """官方实现。结构超参必须和我们逐项对上。"""

    model_cfg = config.model
    return nn.Transformer(
        d_model=model_cfg.d_model,
        nhead=model_cfg.n_heads,
        num_encoder_layers=model_cfg.num_encoder_layers,
        num_decoder_layers=model_cfg.num_decoder_layers,
        dim_feedforward=model_cfg.d_ff,
        dropout=0.0,                 # 验证时不能有任何随机性
        activation=model_cfg.activation,
        layer_norm_eps=1e-6,         # PyTorch 默认 1e-5，我们用的是 1e-6
        batch_first=True,            # 输入布局 [B, T, d]，和我们一致
        norm_first=model_cfg.norm_first,
        bias=model_cfg.bias,
    )


def copy_attention(ours, reference) -> None:
    """搬注意力的 4 组权重。

    官方把 Q/K/V 拼成一个 [3*d_model, d_model] 的 in_proj_weight，顺序 [Wq; Wk; Wv]。
    """

    reference.in_proj_weight.data.copy_(
        torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight], dim=0)
    )
    if ours.w_q.bias is not None and reference.in_proj_bias is not None:
        reference.in_proj_bias.data.copy_(
            torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias], dim=0)
        )
    reference.out_proj.weight.data.copy_(ours.w_o.weight)
    if ours.w_o.bias is not None and reference.out_proj.bias is not None:
        reference.out_proj.bias.data.copy_(ours.w_o.bias)


def copy_weights(ours: Transformer, reference: nn.Transformer, config: Config) -> None:
    model_cfg = config.model

    for i in range(model_cfg.num_encoder_layers):
        layer, ref = ours.encoder.layers[i], reference.encoder.layers[i]
        copy_attention(layer.self_attn, ref.self_attn)
        ref.linear1.weight.data.copy_(layer.ffn.linear1.weight)
        ref.linear1.bias.data.copy_(layer.ffn.linear1.bias)
        ref.linear2.weight.data.copy_(layer.ffn.linear2.weight)
        ref.linear2.bias.data.copy_(layer.ffn.linear2.bias)
        ref.norm1.weight.data.copy_(layer.attn_sublayer.norm.weight)
        ref.norm1.bias.data.copy_(layer.attn_sublayer.norm.bias)
        ref.norm2.weight.data.copy_(layer.ffn_sublayer.norm.weight)
        ref.norm2.bias.data.copy_(layer.ffn_sublayer.norm.bias)

    for i in range(model_cfg.num_decoder_layers):
        layer, ref = ours.decoder.layers[i], reference.decoder.layers[i]
        copy_attention(layer.self_attn, ref.self_attn)
        copy_attention(layer.cross_attn, ref.multihead_attn)
        ref.linear1.weight.data.copy_(layer.ffn.linear1.weight)
        ref.linear1.bias.data.copy_(layer.ffn.linear1.bias)
        ref.linear2.weight.data.copy_(layer.ffn.linear2.weight)
        ref.linear2.bias.data.copy_(layer.ffn.linear2.bias)
        ref.norm1.weight.data.copy_(layer.self_sublayer.norm.weight)
        ref.norm1.bias.data.copy_(layer.self_sublayer.norm.bias)
        ref.norm2.weight.data.copy_(layer.cross_sublayer.norm.weight)
        ref.norm2.bias.data.copy_(layer.cross_sublayer.norm.bias)
        ref.norm3.weight.data.copy_(layer.ffn_sublayer.norm.weight)
        ref.norm3.bias.data.copy_(layer.ffn_sublayer.norm.bias)

    if model_cfg.norm_first:
        # pre-norm 时整个堆叠末尾还有一个 LayerNorm，两边都要搬
        reference.encoder.norm.weight.data.copy_(ours.encoder.norm.weight)
        reference.encoder.norm.bias.data.copy_(ours.encoder.norm.bias)
        reference.decoder.norm.weight.data.copy_(ours.decoder.norm.weight)
        reference.decoder.norm.bias.data.copy_(ours.decoder.norm.bias)


# --------------------------------------------------------------------------
# 两条参照路径
# --------------------------------------------------------------------------
def run_ours(
    ours: Transformer, src_tokens, tgt_tokens, pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """我们的实现：编码 + 解码（一次算完整句）。"""

    src_mask = make_encoder_attn_mask(src_tokens, pad_id)
    tgt_mask = make_decoder_self_attn_mask(tgt_tokens, pad_id)

    x = ours.src_pos(ours.src_embedding(src_tokens))
    encoder_steps = []
    for layer in ours.encoder.layers:
        x, _ = layer(x, src_mask)
        encoder_steps.append(x)
    memory = ours.encoder.norm(x) if ours.encoder.norm is not None else x

    y = ours.tgt_pos(ours.tgt_embedding(tgt_tokens))
    decoder_steps = []
    for layer in ours.decoder.layers:
        y, _ = layer(y, memory, self_mask=tgt_mask, cross_mask=src_mask)
        decoder_steps.append(y)
    hidden = ours.decoder.norm(y) if ours.decoder.norm is not None else y
    return memory, hidden, encoder_steps, decoder_steps


def run_reference_layerwise(
    reference: nn.Transformer,
    src_input: torch.Tensor,
    tgt_input: torch.Tensor,
    src_pad: torch.Tensor,
    tgt_pad: torch.Tensor,
    causal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """官方实现，逐层显式调用（论文的计算顺序）。"""

    x = src_input
    encoder_steps = []
    for layer in reference.encoder.layers:
        x = layer(x, src_key_padding_mask=src_pad)
        encoder_steps.append(x)
    if reference.encoder.norm is not None:
        x = reference.encoder.norm(x)
    memory = x

    y = tgt_input
    decoder_steps = []
    for layer in reference.decoder.layers:
        y = layer(
            y, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        decoder_steps.append(y)
    if reference.decoder.norm is not None:
        y = reference.decoder.norm(y)
    return memory, y, encoder_steps, decoder_steps


def run_reference_forward(
    reference: nn.Transformer,
    src_input: torch.Tensor,
    tgt_input: torch.Tensor,
    src_pad: torch.Tensor,
    tgt_pad: torch.Tensor,
    causal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """官方实现，整体调用（会走内部融合快速路径）。"""

    memory = reference.encoder(src_input, src_key_padding_mask=src_pad)
    hidden = reference.decoder(
        tgt_input, memory,
        tgt_mask=causal,
        tgt_key_padding_mask=tgt_pad,
        memory_key_padding_mask=src_pad,
    )
    return memory, hidden


# --------------------------------------------------------------------------
# 主验证流程
# --------------------------------------------------------------------------
def verify(
    config: Config,
    batch: int = 2,
    src_len: int = 7,
    tgt_len: int = 6,
    dtype: torch.dtype = torch.float32,
    show_internals: bool = False,
) -> Dict[str, float]:
    set_seed(0)
    model_cfg = config.model
    pad_id = 0

    ours = Transformer(model_cfg).eval().to(dtype)
    reference = build_reference(config).eval().to(dtype)
    copy_weights(ours, reference, config)

    # 造输入：故意在末尾塞 <pad>，顺便检验掩码对不对
    src_tokens = torch.randint(1, model_cfg.src_vocab_size, (batch, src_len))
    tgt_tokens = torch.randint(1, model_cfg.tgt_vocab_size, (batch, tgt_len))
    src_tokens[0, 5:] = pad_id
    tgt_tokens[1, 4:] = pad_id

    src_pad = src_tokens.eq(pad_id)
    tgt_pad = tgt_tokens.eq(pad_id)
    causal = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool), diagonal=1)

    with torch.no_grad():
        src_input = ours.src_pos(ours.src_embedding(src_tokens)).to(dtype)
        tgt_input = ours.tgt_pos(ours.tgt_embedding(tgt_tokens)).to(dtype)

        our_memory, our_hidden, our_enc_steps, our_dec_steps = run_ours(
            ours, src_tokens, tgt_tokens, pad_id
        )
        ref_memory, ref_hidden, ref_enc_steps, ref_dec_steps = run_reference_layerwise(
            reference, src_input, tgt_input, src_pad, tgt_pad, causal
        )
        fwd_memory, fwd_hidden = run_reference_forward(
            reference, src_input, tgt_input, src_pad, tgt_pad, causal
        )

    def diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.double() - b.double()).abs().max())

    # 有效位置的差异（padding 位置官方可能填 0，本来就不该一样）
    def valid_diff(a: torch.Tensor, b: torch.Tensor, keep: torch.Tensor) -> float:
        delta = (a.double() - b.double()).abs() * keep.unsqueeze(-1)
        return float(delta.max())

    results = {
        "encoder": valid_diff(our_memory, ref_memory, ~src_pad),
        "decoder": valid_diff(our_hidden, ref_hidden, ~tgt_pad),
        "encoder_pad": diff(our_memory, ref_memory),
        "decoder_pad": diff(our_hidden, ref_hidden),
        # 输出的数值量级，用来算相对误差（绝对差异跟 d_model、层数、初始化都有关）
        "encoder_scale": float(our_memory.double().abs().max()),
        "decoder_scale": float(our_hidden.double().abs().max()),
        "reference_fastpath_encoder": diff(fwd_memory, ref_memory),
        "reference_fastpath_decoder": diff(fwd_hidden, ref_hidden),
    }

    logger.info("=" * 74)
    logger.info(
        f"结构：{'pre-norm' if model_cfg.norm_first else 'post-norm'} | "
        f"d_model={model_cfg.d_model} heads={model_cfg.n_heads} "
        f"layers={model_cfg.num_encoder_layers}+{model_cfg.num_decoder_layers} | 精度 {str(dtype).replace('torch.', '')}"
    )
    logger.info(
        f"输入：英文 [B={batch}, S={src_len}]（其中 {int(src_pad.sum())} 个 <pad>）"
        f"，中文 [B={batch}, T={tgt_len}]（{int(tgt_pad.sum())} 个 <pad>）"
    )
    logger.info("-" * 74)
    logger.info(f"编码器输出最大差异（只看有效位置）  {results['encoder']:.3e}")
    logger.info(f"解码器输出最大差异（只看有效位置）  {results['decoder']:.3e}")

    if show_internals:
        logger.info("-" * 74)
        logger.info("逐层最大差异（只在有效位置上比）")
        for index, (our_step, ref_step) in enumerate(zip(our_enc_steps, ref_enc_steps), start=1):
            logger.info(
                f"  编码器第 {index} 层：有效位置 "
                f"{valid_diff(our_step, ref_step, ~src_pad):.3e} | 全部位置 {diff(our_step, ref_step):.3e}"
            )
        for index, (our_step, ref_step) in enumerate(zip(our_dec_steps, ref_dec_steps), start=1):
            logger.info(
                f"  解码器第 {index} 层：有效位置 "
                f"{valid_diff(our_step, ref_step, ~tgt_pad):.3e} | 全部位置 {diff(our_step, ref_step):.3e}"
            )

    if show_internals:
        logger.info("-" * 74)
        logger.info(
            f"参考信息：官方 encoder.forward / decoder.forward 与官方逐层调用之间差 "
            f"{results['reference_fastpath_encoder']:.3e} / {results['reference_fastpath_decoder']:.3e}"
        )
        logger.info(
            f"          官方在 padding 位置上的输出和逐层调用差 "
            f"{max(diff(fwd_memory, ref_memory), diff(fwd_hidden, ref_hidden)):.3e}"
            "（nested tensor 会把 padding 位置直接填 0）"
        )

    worst = max(results["encoder"], results["decoder"])
    scale = max(results["encoder_scale"], results["decoder_scale"], 1.0)
    relative = worst / scale
    # 判据用**相对误差**：绝对值大小跟 d_model、层数、初始化都有关，不好定阈值。
    # 1e-5 是留给官方融合算子的余量；真写错了会是 1e-1 量级。
    tolerance = 1e-5
    ok = relative < tolerance
    logger.info("-" * 74)
    logger.info(
        f"判定：{'一致 ✅' if ok else '不一致 ❌'} —— "
        f"最大差异 {worst:.3e} / 数值量级 {scale:.2f} = 相对误差 {relative:.3e}"
        f"（阈值 {tolerance:g}）"
    )
    if not ok:
        logger.info(
            "排查顺序建议：1) norm_first 是否一致；2) 残差接在 norm 前还是后；"
            "3) 因果掩码方向；4) 交叉注意力的 K/V 是否用了编码器输出；"
            "5) 权重是不是搬错了（in_proj_weight 的拼接顺序是 Wq;Wk;Wv）"
        )
    return results


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description="和 PyTorch 官方 nn.Transformer 对齐验证")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None, help="同时设置编码器和解码器层数")
    parser.add_argument("--norm_first", type=int, default=None, help="1=pre-norm，0=post-norm")
    parser.add_argument("--activation", default=None, choices=["relu", "gelu", "swish"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--show-internals", action="store_true", help="额外展示官方两条路径之间的差异")
    args = parser.parse_args()

    config = PRESETS[args.preset](vocab_size=200)
    config.model.src_vocab_size = 200
    config.model.tgt_vocab_size = 200
    if args.d_model:
        config.model.d_model = args.d_model
    if args.n_heads:
        config.model.n_heads = args.n_heads
    if args.layers:
        config.model.num_encoder_layers = args.layers
        config.model.num_decoder_layers = args.layers
    if args.norm_first is not None:
        config.model.norm_first = bool(args.norm_first)
    if args.activation:
        config.model.activation = args.activation

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    verify(config, batch=args.batch, dtype=dtype, show_internals=args.show_internals)


if __name__ == "__main__":
    main()

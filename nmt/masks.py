"""掩码（mask）：Transformer 里最容易写错、错了又最难发现的地方。

本项目统一约定：

    mask 是 bool 张量，**True 表示"这个位置不允许被看到"**，
    在注意力里会被填成一个极小值（见 attention.mask_value），
    softmax 之后权重≈0。

三个用得到的地方：

    编码器自注意力   需要 mask 源句的 padding       形状 [B, 1, 1, S]
    解码器自注意力   需要 mask 目标句 padding + 因果 形状 [B, 1, T, T]
    解码器交叉注意力 需要 mask 源句 padding          形状 [B, 1, 1, S]

为什么中间的维度是 1？因为要和注意力分数 [B, heads, Tq, Tk] 做广播。
"""

from __future__ import annotations

import torch


def make_pad_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """padding 的位置为 True。

    tokens: [B, T] -> mask: [B, T]
    """

    return tokens.eq(pad_id)


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """因果掩码（也叫 look-ahead mask、上三角掩码）：[size, size]。

    mask[i, j] = True 表示第 i 个位置不许看第 j 个位置。
    解码器要"根据前文猜下一个字"，所以 j > i 的位置全部屏蔽。

        位置   0  1  2  3
          0   .  X  X  X
          1   .  .  X  X
          2   .  .  .  X
          3   .  .  .  .

    （. 可看，X 屏蔽）
    """

    # torch.triu(..., diagonal=1) 取严格上三角：正好是"未来"的位置
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def make_encoder_attn_mask(src_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """编码器自注意力掩码：[B, 1, 1, S]。

    源句里 padding 的位置不该被读到 —— 否则 [PAD] 会参与加权，
    句子的表示会随 padding 数量变化，短句和长句就不公平了。
    """

    return make_pad_mask(src_tokens, pad_id)[:, None, None, :]


def make_decoder_self_attn_mask(tgt_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """解码器自注意力掩码：[B, 1, T, T]。两件事一起做：

    1. 因果：不能看到未来的字（否则训练时等于抄答案）；
    2. padding：不能看到 <pad>（训练时 <pad> 是凑长度补出来的）。
    """

    batch_size, tgt_len = tgt_tokens.shape
    causal = make_causal_mask(tgt_len, tgt_tokens.device)          # [T, T]
    pad = make_pad_mask(tgt_tokens, pad_id)                        # [B, T]
    return causal[None, None, :, :] | pad[:, None, None, :]        # [B,1,T,T]，广播相加时用"或"


def make_cross_attn_mask(src_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """交叉注意力掩码：[B, 1, 1, S]。

    解码器查询英文（key/value 是编码器输出），所以只需要屏蔽英文的 padding。
    """

    return make_encoder_attn_mask(src_tokens, pad_id)


def make_incremental_self_attn_mask(tgt_prefix: torch.Tensor, pad_id: int) -> torch.Tensor:
    """增量解码时的自注意力掩码：[B, 1, 1, t]。

    全句训练时用 [B,1,T,T] 的因果掩码；
    但增量解码每步只送 **1 个位置**的 query，key 是全部历史，
    "只能看前面"这件事由"query 只有当前这一步"天然保证了，
    所以这里只需要屏蔽 padding。
    """

    return make_pad_mask(tgt_prefix, pad_id)[:, None, None, :]


def mask_summary(mask: torch.Tensor) -> str:
    """给日志/调试用：打印掩码形状与被屏蔽比例。"""

    ratio = mask.float().mean().item()
    return f"shape={tuple(mask.shape)} 屏蔽比例={ratio:.1%}"

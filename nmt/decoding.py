"""解码：怎么把训练好的模型"用"起来，一个词一个词地生成译文。

两种策略：

**贪心（greedy）** —— 每步都选概率最大的词。
    快，但一步选错就再也回不来了。适合看"模型当前最想说什么"。

**束搜索（beam search）** —— 同时保留 K 条候选路径，
    每步把所有路径的下一步扩展出来，只留累计概率最高的 K 条。
    更贵（约 K 倍），但能找回"眼前看着一般、整体更好"的译法。

束搜索里的两个细节，直接决定译文长短和分数：

* **长度惩罚**：对数概率是越乘越小的负数，直接按总分排序会偏向短句
  （因为短句少乘几个负数）。GNMT 的做法是除以 ((5+len)/6)^alpha，
  alpha=0.6 时短句略微受罚、长句略微占优。
* **EOS 的处理**：某条路径生成了结束符，就把它放进"完成池"，
  不再参与后续扩展，但仍然是这个句子的候选答案。

两者都用 KV cache 做增量解码：第 t 步只算第 t 个位置的 query，
历史 K/V 从 cache 里取，避免每步都重算整段 —— 这是能实际用起来的前提。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .masks import make_incremental_self_attn_mask


def _max_len_limit(model, max_len: int) -> int:
    """不能超过位置编码支持的长度。"""

    return min(max_len, model.config.max_len - 1)


def _trim(sequence: Sequence[int], eos_id: int) -> List[int]:
    """去掉 <bos>，并在第一个 <eos> 处截断。"""

    result: List[int] = []
    for token in sequence:
        if token == eos_id:
            break
        result.append(int(token))
    return result


@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = 192,
    min_len: int = 0,
) -> List[List[int]]:
    """贪心解码。src: [B, S] -> 每个样本一个 id 列表（不含 <bos>）。"""

    model.eval()
    device = src.device
    batch_size = src.size(0)
    max_len = _max_len_limit(model, max_len)

    # 英文只编码一次，之后解码器每步都复用它
    memory, _ = model.encode(src, src_mask)
    self_caches, cross_caches = model.make_decoder_caches()

    tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(max_len):
        mask = make_incremental_self_attn_mask(tokens, pad_id)
        hidden, _ = model.decode(
            tokens[:, -1:],            # 只喂最新生成的那一个 token
            memory,
            mask,
            src_mask,
            caches=self_caches,
            cross_caches=cross_caches,
            position_offset=step,      # 位置编码要接着往下排
        )
        logits = model.generator(hidden[:, -1])          # [B, V]
        if step < min_len:
            logits[:, eos_id] = float("-inf")            # 前 min_len 步不许结束

        next_token = logits.argmax(dim=-1)               # [B]
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        finished |= next_token.eq(eos_id)
        if bool(finished.all()):
            break

    return [_trim(row, eos_id) for row in tokens[:, 1:].tolist()]


def _length_penalty(length: int, alpha: float) -> float:
    """GNMT 长度惩罚：((5 + len) / 6) ^ alpha。"""

    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search_decode(
    model,
    src: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam_size: int = 4,
    max_len: int = 192,
    length_penalty: float = 0.6,
    min_len: int = 0,
) -> Tuple[List[List[int]], List[float]]:
    """束搜索解码。返回 (每个样本的 id 列表, 对应的归一化分数)。

    实现上把 [B, K] 条活跃路径拍平成 [B*K] 一批来算，
    每步只对"选中的父路径"重排 KV cache —— 这是束搜索 + cache 的标准写法。
    """

    model.eval()
    device = src.device
    batch_size = src.size(0)
    beam = max(1, beam_size)
    max_len = _max_len_limit(model, max_len)

    memory, _ = model.encode(src, src_mask)                      # [B, S, d]
    memory = memory.repeat_interleave(beam, dim=0)               # [B*K, S, d]
    beam_src_mask = src_mask.repeat_interleave(beam, dim=0) if src_mask is not None else None

    self_caches, cross_caches = model.make_decoder_caches()
    tokens = torch.full((batch_size * beam, 1), bos_id, dtype=torch.long, device=device)

    scores = torch.full((batch_size * beam,), float("-inf"), device=device)
    scores[::beam] = 0.0                                          # 每个样本只有第一条路径是活的
    dead = torch.zeros(batch_size * beam, dtype=torch.bool, device=device)
    completed: List[List[Tuple[float, List[int]]]] = [[] for _ in range(batch_size)]

    for step in range(max_len):
        mask = make_incremental_self_attn_mask(tokens, pad_id)
        hidden, _ = model.decode(
            tokens[:, -1:], memory, mask, beam_src_mask,
            caches=self_caches, cross_caches=cross_caches, position_offset=step,
        )
        log_probs = F.log_softmax(model.generator(hidden[:, -1]).float(), dim=-1)  # [B*K, V]
        if step < min_len:
            log_probs[:, eos_id] = float("-inf")
        log_probs[dead] = float("-inf")     # 已经结束的路径不再扩展

        vocab_size = log_probs.size(-1)
        candidates = (scores.unsqueeze(1) + log_probs).view(batch_size, beam * vocab_size)
        top_scores, top_indices = candidates.topk(min(2 * beam, candidates.size(-1)), dim=-1)

        # 新的 K 条路径（正在进行的）与它们各自的父路径、新 token
        new_scores = torch.full((batch_size, beam), float("-inf"), device=device)
        new_parents = torch.zeros(batch_size, beam, dtype=torch.long, device=device)
        new_tokens = torch.zeros(batch_size, beam, dtype=torch.long, device=device)
        new_dead = torch.ones(batch_size, beam, dtype=torch.bool, device=device)

        for b in range(batch_size):
            slot = 0
            for score, index in zip(top_scores[b].tolist(), top_indices[b].tolist()):
                if slot >= beam:
                    break
                if score == float("-inf"):
                    break
                parent, token = divmod(index, vocab_size)
                sequence = _trim(tokens[b * beam + parent, 1:].tolist(), eos_id)
                if token == eos_id:
                    # 走完了：记进完成池，长度包含刚生成的 <eos>
                    length = len(sequence) + 1
                    normalized = score / _length_penalty(length, length_penalty)
                    completed[b].append((normalized, sequence))
                else:
                    new_scores[b, slot] = score
                    new_parents[b, slot] = parent
                    new_tokens[b, slot] = token
                    new_dead[b, slot] = False
                    slot += 1

        # 重排 KV cache：把被选中的父路径的缓存搬到新的位置
        flat_parents = (torch.arange(batch_size, device=device).unsqueeze(1) * beam + new_parents).view(-1)
        tokens = tokens.index_select(0, flat_parents)
        tokens = torch.cat([tokens, new_tokens.view(-1, 1)], dim=1)
        for cache in self_caches + cross_caches:
            if cache.get("k") is not None:
                cache["k"] = cache["k"].index_select(0, flat_parents)
                cache["v"] = cache["v"].index_select(0, flat_parents)

        scores = new_scores.view(-1)
        dead = new_dead.view(-1)
        if bool(dead.all()) or bool(torch.isinf(scores).all()):
            break

    # 收尾：优先取完成池里分数最高的；一条都没走完就退回"当前最好的一条"
    results: List[List[int]] = []
    result_scores: List[float] = []
    for b in range(batch_size):
        if completed[b]:
            best_score, best_sequence = max(completed[b], key=lambda item: item[0])
        else:
            best_slot = int(torch.argmax(scores[b * beam : (b + 1) * beam]))
            best_sequence = _trim(tokens[b * beam + best_slot, 1:].tolist(), eos_id)
            best_score = float(scores[b * beam + best_slot]) / _length_penalty(
                max(1, len(best_sequence)), length_penalty
            )
        results.append(best_sequence)
        result_scores.append(float(best_score))
    return results, result_scores

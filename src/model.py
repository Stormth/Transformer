"""把各个零件组装成完整的 Transformer，并实现推理（贪心 / 束搜索）。

整体数据流（以 batch_size=B、源句长 Ts、目标句长 Tt 为例）：

    src [B, Ts] ──embed+PE──> [B,Ts,d] ──Encoder──> memory [B,Ts,d]
                                                        │
    tgt [B, Tt] ──embed+PE──> [B,Tt,d] ──Decoder(memory)──> [B,Tt,d]
                                                        │
                                              Linear(d_model -> vocab)
                                                        │
                                               logits [B, Tt, V]

训练：一次前向算出所有位置的 logits，与"右移一位"的标签算交叉熵。
推理：从 <bos> 开始一个词一个词地生成，用 KV cache 加速、用束搜索提升质量。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .attention import KV
from .config import ModelConfig
from .decoder import Decoder, DecoderLayer
from .embedding import PositionalEncoding, TokenEmbedding
from .encoder import Encoder, EncoderLayer
from .masks import combine_masks, make_causal_mask, make_pad_mask

# 每个解码器层的缓存： (自注意力 KV, 交叉注意力 KV)
LayerCache = Tuple[Optional[KV], KV]


class Transformer(nn.Module):
    """标准的 Encoder-Decoder Transformer（原论文结构）。"""

    def __init__(self, cfg: ModelConfig, pad_id: int = 0, bos_id: int = 1, eos_id: int = 2):
        super().__init__()
        self.cfg = cfg
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id

        # ---- 嵌入层（源/目标语言共享同一个词表时，可以用一个 Embedding，这里为清晰起见分开）----
        self.src_embed = TokenEmbedding(cfg.src_vocab_size, cfg.d_model, pad_id)
        self.tgt_embed = TokenEmbedding(cfg.tgt_vocab_size, cfg.d_model, pad_id)
        self.positional = PositionalEncoding(cfg.d_model, cfg.dropout, cfg.max_len)

        # ---- 编码器 / 解码器 ----
        enc_layer = EncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.norm_first, cfg.activation
        )
        dec_layer = DecoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.norm_first, cfg.activation
        )
        self.encoder = Encoder(enc_layer, cfg.num_encoder_layers, cfg.norm_first)
        self.decoder = Decoder(dec_layer, cfg.num_decoder_layers, cfg.norm_first)

        # ---- 输出层：d_model -> 词表大小 ----
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

        if cfg.tie_embeddings and cfg.src_vocab_size == cfg.tgt_vocab_size:
            # 权重共享（weight tying）：输入端和输出端用同一个矩阵。
            # 优点：省参数、缓解小语料过拟合；Transformer 原论文用的是类似思路。
            self.generator.weight = self.tgt_embed.embed.weight

        self._reset_parameters()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def _reset_parameters(self) -> None:
        """Xavier 初始化 + 偏置置零 + padding 行置零。

        ⚠️ 一个很容易踩的坑：不能简单地"把一维参数全部置零"。
        LayerNorm 的 weight 默认是 1，若被置零，每个子层的输出都会被清零，
        整个模型会退化成"常数预测"，loss 卡在 log(词表大小) 附近不动。
        所以这里只把名字里带 bias 的一维参数置零。
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            else:
                # LayerNorm 的 weight：保持默认的 1.0
                with torch.no_grad():
                    p.fill_(1.0)
        # padding_idx 对应的向量必须保持为 0：它不携带语义，也不应产生梯度
        with torch.no_grad():
            self.src_embed.embed.weight[self.pad_id].zero_()
            self.tgt_embed.embed.weight[self.pad_id].zero_()

    def num_parameters(self, only_trainable: bool = True) -> int:
        params = self.parameters()
        return sum(p.numel() for p in params if (p.requires_grad or not only_trainable))

    # ------------------------------------------------------------------ #
    # 掩码工具
    # ------------------------------------------------------------------ #
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """[B, Ts] -> [B, 1, 1, Ts]，屏蔽 <pad>。"""
        lengths = (src != self.pad_id).sum(dim=1)
        return make_pad_mask(lengths, src.size(1))

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """[B, Tt] -> [B, 1, Tt, Tt]，既屏蔽 <pad> 又屏蔽未来。"""
        lengths = (tgt != self.pad_id).sum(dim=1)
        pad_mask = make_pad_mask(lengths, tgt.size(1))
        causal = make_causal_mask(tgt.size(1), tgt.device)
        return combine_masks(pad_mask, causal)

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """[B, Ts] -> memory [B, Ts, d_model]"""
        x = self.positional(self.src_embed(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B, Tt] -> logits [B, Tt, V]"""
        x = self.positional(self.tgt_embed(tgt))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.generator(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ------------------------------------------------------------------ #
    # 增量解码（KV cache）
    # ------------------------------------------------------------------ #
    def _init_caches(self, memory: torch.Tensor) -> List[LayerCache]:
        """初始化缓存：交叉注意力的 K/V 一次算好（它与已生成内容无关）。"""
        return [(None, layer.cross_attn.project_kv(memory)) for layer in self.decoder.layers]

    def _step(
        self, token: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor,
        caches: List[LayerCache], offset: int,
    ) -> Tuple[torch.Tensor, List[LayerCache]]:
        """输入 [B, 1] 的新 token，返回 [B, V] 的 logits 与新缓存。"""
        x = self.positional(self.tgt_embed(token), offset=offset)  # 位置编码要用真实位置
        hidden, caches = self.decoder.forward_step(x, memory, src_mask, caches)
        return self.generator(hidden[:, -1]), caches

    @staticmethod
    def _reorder_caches(caches: List[LayerCache], index: torch.Tensor) -> List[LayerCache]:
        """按 beam 索引重排缓存（束搜索中每一行"血脉"会变）。"""
        out: List[LayerCache] = []
        for self_kv, cross_kv in caches:
            new_self = None
            if self_kv is not None:
                new_self = (self_kv[0].index_select(0, index), self_kv[1].index_select(0, index))
            new_cross = (cross_kv[0].index_select(0, index), cross_kv[1].index_select(0, index))
            out.append((new_self, new_cross))
        return out

    # ------------------------------------------------------------------ #
    # 推理 1：贪心解码（每步选概率最大的词）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int = 64,
        src_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """返回每个样本生成的 id 列表（不含 <bos>，含 <eos> 为止）。"""
        was_training = self.training
        self.eval()
        device = src.device
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        memory = self.encode(src, src_mask)
        caches = self._init_caches(memory)

        B = src.size(0)
        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results: List[List[int]] = [[] for _ in range(B)]

        for step in range(max_len):
            logits, caches = self._step(ys[:, -1:], memory, src_mask, caches, step)
            next_ids = logits.argmax(dim=-1)  # [B]

            for b in range(B):
                if finished[b]:
                    continue
                token = int(next_ids[b])
                results[b].append(token)
                if token == self.eos_id:
                    finished[b] = True
            ys = next_ids.unsqueeze(1)
            if bool(finished.all()):
                break

        if was_training:
            self.train()
        return results

    # ------------------------------------------------------------------ #
    # 推理 2：束搜索（每步保留 beam 个最优假设）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        beam_size: int = 4,
        max_len: int = 64,
        length_penalty: float = 0.6,
        src_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Tuple[List[int], float]]]:
        """束搜索解码。

        返回：外层按 batch，内层是按分数从高到低排列的 (token_ids, 归一化分数)。

        为什么需要束搜索？
            贪心每步只保留 1 个最优词，一旦早期选错就无法回头；
            束搜索同时保留 beam 条候选，用"整句分数"而非"逐步贪心"来选最终结果。

        长度惩罚：
            生成概率是连乘（取对数后是累加），句子越长分数越低，
            会让模型偏爱短句。用 lp = ((5 + len) / 6)^alpha 归一化即可修正。
        """
        was_training = self.training
        self.eval()
        device = src.device
        B = src.size(0)
        beam = beam_size
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        # 1) 编码一次，然后把 batch 复制 beam 份（batch 维 = B*beam）
        memory = self.encode(src, src_mask)
        memory_b = memory.repeat_interleave(beam, dim=0)
        src_mask_b = src_mask.repeat_interleave(beam, dim=0)
        caches = self._init_caches(memory_b)

        ys = torch.full((B * beam, 1), self.bos_id, dtype=torch.long, device=device)
        # 2) 每行的 beam 分数：第 0 条为 0，其余为 -inf（第一步只有一条有效路径）
        beam_scores = torch.full((B, beam), float("-inf"), device=device)
        beam_scores[:, 0] = 0.0

        finished: List[List[Tuple[List[int], float]]] = [[] for _ in range(B)]
        V = self.cfg.tgt_vocab_size

        for step in range(max_len):
            logits, caches = self._step(ys[:, -1:], memory_b, src_mask_b, caches, step)
            logp = logits.log_softmax(dim=-1).view(B, beam, V)          # [B, beam, V]
            total = logp + beam_scores.unsqueeze(-1)                    # [B, beam, V]
            if step == 0:
                total[:, 1:, :] = float("-inf")                         # 第一步其余 beam 无效

            flat = total.view(B, beam * V)
            topk_scores, topk_index = flat.topk(beam, dim=-1)           # [B, beam]
            beam_index = topk_index // V                                # 来自哪条 beam
            token_index = topk_index % V                                # 选了哪个词

            # 3) 重排历史 token 与缓存
            flat_beam = (torch.arange(B, device=device).unsqueeze(1) * beam + beam_index).view(-1)
            ys = torch.cat([ys.index_select(0, flat_beam), token_index.view(-1, 1)], dim=1)
            caches = self._reorder_caches(caches, flat_beam)

            # 4) 收集已经生成 <eos> 的假设，并把这些 beam 标记为"已结束"
            new_scores = topk_scores.clone()
            for b in range(B):
                for j in range(beam):
                    if int(token_index[b, j]) == self.eos_id and topk_scores[b, j] != float("-inf"):
                        ids = ys[b * beam + j].tolist()[1:-1]  # 去掉 <bos> 与 <eos>
                        finished[b].append((ids, float(topk_scores[b, j])))
                        new_scores[b, j] = float("-inf")        # 不再参与后续扩展
            beam_scores = new_scores

            # 5) 所有 beam 都结束则提前退出
            if bool((beam_scores == float("-inf")).all()) or step == max_len - 1:
                for b in range(B):
                    for j in range(beam):
                        score = float(beam_scores[b, j])
                        if score != float("-inf"):
                            ids = ys[b * beam + j].tolist()[1:]
                            if ids and ids[-1] == self.eos_id:
                                ids = ids[:-1]
                            finished[b].append((ids, score))
                break

        # 6) 用长度惩罚归一化后排序
        def _norm(item: Tuple[List[int], float]) -> float:
            ids, score = item
            lp = ((5 + len(ids)) / 6) ** length_penalty
            return score / lp

        # 只返回前 beam 个候选（搜索过程中可能收集到超过 beam 个以 <eos> 结尾的假设）
        ranked = [sorted(hyps, key=_norm, reverse=True)[:beam] for hyps in finished]
        if was_training:
            self.train()
        return ranked

    def beam_search_best_ids(self, src: torch.Tensor, **kwargs) -> List[List[int]]:
        """只取束搜索的第一名。"""
        return [hyps[0][0] for hyps in self.beam_search(src, **kwargs)]

    # ------------------------------------------------------------------ #
    # 其它
    # ------------------------------------------------------------------ #
    @staticmethod
    def shift_right(tgt: torch.Tensor, bos_id: int) -> torch.Tensor:
        """把目标句右移一位作为解码器输入：[t1..tn] -> [<bos>, t1..t_{n-1}]"""
        bos = tgt.new_full((tgt.size(0), 1), bos_id)
        return torch.cat([bos, tgt[:, :-1]], dim=1)

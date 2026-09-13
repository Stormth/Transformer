"""把零件组装成完整的 Transformer，并提供掩码工具。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .decoder import Decoder, DecoderLayer
from .embedding import PositionalEncoding, TokenEmbedding
from .encoder import Encoder, EncoderLayer
from .masks import (
    make_cross_attn_mask,
    make_decoder_self_attn_mask,
    make_encoder_attn_mask,
)
from .utils import count_parameters, format_big


class Transformer(nn.Module):
    """标准的编码器-解码器 Transformer。

    前向流程（训练时一次算完整句）：

        英文 id [B,S] --词嵌入+位置--> [B,S,d] --编码器--> memory [B,S,d]
        德文 id [B,T] --词嵌入+位置--> [B,T,d] --解码器(看 memory)--> [B,T,d]
                                                         --输出层--> logits [B,T,V]

    之后用 logits 和"右移一位的德文"算交叉熵 —— 每个位置都在预测下一个词。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # --- 词嵌入 ---
        self.src_embedding = TokenEmbedding(
            config.src_vocab_size, config.d_model, config.scale_embedding
        )
        if config.share_vocab and config.src_vocab_size == config.tgt_vocab_size:
            # 英德共用一个词表时，直接复用同一个模块 -> 参数也共享
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TokenEmbedding(
                config.tgt_vocab_size, config.d_model, config.scale_embedding
            )

        self.src_pos = PositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.tgt_pos = PositionalEncoding(config.d_model, config.max_len, config.dropout)

        # --- 编码器 ---
        encoder_layer = EncoderLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
            norm_first=config.norm_first,
            bias=config.bias,
        )
        self.encoder = Encoder(
            encoder_layer, config.num_encoder_layers, config.d_model, config.norm_first
        )

        # --- 解码器 ---
        decoder_layer = DecoderLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
            norm_first=config.norm_first,
            bias=config.bias,
        )
        self.decoder = Decoder(
            decoder_layer, config.num_decoder_layers, config.d_model, config.norm_first
        )

        # --- 输出层：把 d_model 投到词表大小 ---
        self.generator = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)
        if config.tie_embeddings:
            # 共享权重：输出的每个词向量，就是目标词嵌入矩阵的一行。
            # 少一份参数，小数据上通常也更稳。（注意 generator 没有 bias 才能共享）
            if isinstance(self.tgt_embedding, TokenEmbedding):
                self.generator.weight = self.tgt_embedding.embedding.weight

        self._reset_parameters()

    # ---------------------------------------------------------------- 初始化
    def _reset_parameters(self) -> None:
        """Xavier 初始化 + 词嵌入正态初始化。

        词嵌入用 N(0, 0.02) 而不是默认的 N(0, 1)：
        默认初始化方差太大，配合 sqrt(d_model) 缩放会让第一层的激活值爆炸。
        """

        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # 共享权重时上面那个循环会把同一个张量初始化两次，无伤大雅；
        # 但 generator.weight 被当成"dim>1 的参数"做过 xavier，这里重新按词嵌入初始化。
        if self.config.tie_embeddings and isinstance(self.tgt_embedding, TokenEmbedding):
            nn.init.normal_(self.tgt_embedding.embedding.weight, mean=0.0, std=0.02)

    # ---------------------------------------------------------------- 掩码
    @staticmethod
    def make_masks(
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        pad_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """一次性造好两个掩码：编码器侧与解码器侧。"""

        return (
            make_encoder_attn_mask(src_tokens, pad_id),
            make_decoder_self_attn_mask(tgt_tokens, pad_id),
        )

    # ---------------------------------------------------------------- 前向
    def encode(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """英文 id [B,S] -> memory [B,S,d_model]"""

        x = self.src_pos(self.src_embedding(src_tokens))
        return self.encoder(x, src_mask, return_weights=return_weights)

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        caches: Optional[list] = None,
        cross_caches: Optional[list] = None,
        return_weights: bool = False,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """德文 id [B,T] + memory -> 隐状态 [B,T,d_model]

        position_offset：增量解码时告诉位置编码"这一小段从第几个位置开始"。
        """

        x = self.tgt_pos(self.tgt_embedding(tgt_tokens), offset=position_offset)
        return self.decoder(
            x,
            memory,
            self_mask=tgt_mask,
            cross_mask=src_mask,
            caches=caches,
            cross_caches=cross_caches,
            return_weights=return_weights,
        )

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """训练用：一次算出所有位置的 logits，形状 [B, T, tgt_vocab_size]。"""

        memory, _ = self.encode(src_tokens, src_mask)
        hidden, _ = self.decode(tgt_tokens, memory, tgt_mask, src_mask)
        return self.generator(hidden)

    def make_decoder_caches(self) -> Tuple[list, list]:
        """给增量解码准备每层的空 cache。"""

        self_caches = [{"k": None, "v": None} for _ in range(self.config.num_decoder_layers)]
        cross_caches = [{"k": None, "v": None} for _ in range(self.config.num_decoder_layers)]
        return self_caches, cross_caches

    # ---------------------------------------------------------------- 信息
    def describe(self) -> str:
        """给学习者看的结构摘要：每部分多少参数。"""

        def group_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        total = count_parameters(self)
        shared_note = "（英德共享同一张词嵌入表）" if self.config.share_vocab else ""
        lines = [
            "模型结构摘要",
            f"  d_model={self.config.d_model}  n_heads={self.config.n_heads}  "
            f"d_k={self.config.d_k}  d_ff={self.config.d_ff}",
            f"  编码器 / 解码器层数：{self.config.num_encoder_layers} / "
            f"{self.config.num_decoder_layers}",
            f"  {'pre-norm' if self.config.norm_first else 'post-norm'}，"
            f"激活函数 {self.config.activation}",
            f"  词嵌入        {format_big(group_params(self.src_embedding))} {shared_note}",
            f"  编码器        {format_big(group_params(self.encoder))}",
            f"  解码器        {format_big(group_params(self.decoder))}",
            f"  输出层        {format_big(group_params(self.generator))}",
            f"  合计          {format_big(total)} 个可训练参数",
        ]
        return "\n".join(lines)

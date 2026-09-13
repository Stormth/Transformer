"""集中管理超参数。

命名沿用《Attention Is All You Need》论文与后续文献的常用叫法，
方便你对照论文/开源代码阅读：

    base 配置 : d_model=512, n_heads=8, num_layers=6, d_ff=2048（论文里的 base model）
    tiny 配置 : 本项目默认值，笔记本 GPU 几分钟即可训练出可用的玩具翻译模型
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelConfig:
    """模型结构超参数。"""

    src_vocab_size: int = 3000          # 源语言（中文）词表大小
    tgt_vocab_size: int = 3000          # 目标语言（英文）词表大小
    d_model: int = 192                  # 隐藏层维度
    n_heads: int = 4                    # 注意力头数，d_k = d_model / n_heads = 48
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 384                     # 前馈网络中间层维度
    dropout: float = 0.1
    norm_first: bool = False            # False=post-norm(原论文)，True=pre-norm(现代大模型)
    activation: str = "relu"
    tie_embeddings: bool = False        # 输出层与目标词嵌入共享权重（少参数，且常更稳）
    max_len: int = 512                  # 位置编码支持的最大长度

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainConfig:
    """训练超参数。"""

    # --- 数据 ---
    data_dir: str = "data/processed"
    max_src_len: int = 64
    max_tgt_len: int = 64
    batch_size: int = 64
    max_tokens: int = 4096              # 动态 batching 的 token 上限
    num_workers: int = 0
    use_bucket_sampler: bool = True     # 按长度分桶，减少 padding 浪费

    # --- 优化 ---
    epochs: int = 20
    lr_scale: float = 1.0               # Noam 调度的缩放系数（小语料可以调大）
    warmup_steps: int = 800             # warmup 步数（原论文 4000，用于大规模语料）
    label_smoothing: float = 0.1
    clip_grad: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    weight_decay: float = 0.0

    # --- 验证 / 日志 ---
    eval_every_epochs: int = 1
    eval_max_sentences: int = 200       # 验证时最多跑多少句算 BLEU（解码较慢）
    log_every_steps: int = 50
    seed: int = 2024
    device: str = "auto"                # auto / cpu / cuda / cuda:0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Config:
    """把模型与训练配置打包，便于一起存盘 / 从 checkpoint 恢复。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"model": asdict(self.model), "train": asdict(self.train)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            train=TrainConfig.from_dict(d.get("train", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def tiny_config(vocab_size: int = 3000) -> ModelConfig:
    """几秒钟级的小模型，用于单元测试 / 冒烟测试。"""
    return ModelConfig(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=32,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=64,
        dropout=0.0,
        max_len=64,
    )


def base_config(vocab_size: int = 8000) -> ModelConfig:
    """论文 base model 的规模（需要更大的显存和训练语料）。"""
    return ModelConfig(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=512,
    )

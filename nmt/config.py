"""超参数集中管理。

命名沿用《Attention Is All You Need》和后续文献的通用叫法，
方便你对照论文 / 开源实现阅读：

    base 配置 : d_model=512, n_heads=8, 6+6 层, d_ff=2048（论文的 base model）
    tiny 配置 : 本项目的冒烟测试配置，CPU 上也能跑，用来验证代码没写错

所有配置都能存成 JSON 一起塞进 checkpoint，
这样"训练时用的什么结构"永远不会和你记忆里的对不上。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """模型结构超参数。"""

    # --- 词表 ---
    src_vocab_size: int = 16000        # 源语言（英文）词表大小
    tgt_vocab_size: int = 16000        # 目标语言（中文）词表大小
    share_vocab: bool = True           # 中英共用一个 BPE 词表（本项目默认）

    # --- 主体结构 ---
    d_model: int = 512                 # 隐层维度，所有子层输入输出都是它
    n_heads: int = 8                   # 注意力头数，d_k = d_model / n_heads = 64
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048                   # 前馈网络中间层维度，论文用 4 * d_model
    dropout: float = 0.1
    attention_dropout: float = 0.0     # 注意力权重上的 dropout，默认关（论文未提）
    activation: str = "relu"           # relu / gelu / swish
    norm_first: bool = False           # False = post-norm（原论文）；True = pre-norm（现代模型）
    tie_embeddings: bool = True        # 输出层与目标词嵌入共享权重
    max_len: int = 256                 # 位置编码支持的最大长度
    scale_embedding: bool = True       # 词嵌入乘 sqrt(d_model)，论文 3.4 节
    bias: bool = True                  # 线性层是否带 bias（论文原文带）

    @property
    def d_k(self) -> int:
        """每个头的维度。512 / 8 = 64。"""

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} 必须能被 n_heads={self.n_heads} 整除")
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainConfig:
    """训练超参数。"""

    # --- 数据 ---
    data_dir: str = "data/ready"
    max_src_len: int = 192             # 超过就裁掉（数据准备阶段已经过滤过长的句子）
    max_tgt_len: int = 192
    max_tokens: int = 8192             # 动态 batching：一个 batch 里 token 总量上限
    max_sentences: int = 128           # 单个 batch 的句对数上限
    num_workers: int = 0               # Windows 上设 0 最稳；Linux 服务器可以调大
    bucket_size: int = 64              # 长度分桶宽度：长度差不超过它的句子放一起

    # --- 优化 ---
    epochs: int = 30
    lr_scale: float = 1.0              # Noam 学习率的缩放系数，小语料可以调大
    warmup_steps: int = 4000           # 论文用 4000；小语料 1000~2000 更合适
    label_smoothing: float = 0.1
    clip_grad: float = 1.0             # 梯度范数裁剪
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    weight_decay: float = 0.0
    accum_steps: int = 1               # 梯度累积：显存不够就调大它

    # --- 混合精度 ---
    amp: bool = True
    amp_dtype: str = "float16"         # float16 需要 GradScaler；bfloat16 不需要

    # --- 验证 / 日志 ---
    eval_every_epochs: int = 1
    eval_max_sentences: int = 400      # 验证时最多解码多少句（解码比训练慢得多）
    eval_beam_size: int = 1            # 验证阶段先用贪心，快
    log_every_steps: int = 50
    save_every_epochs: int = 1
    patience: int = 0                  # >0 时，验证 BLEU 连续多少轮不涨就早停
    tensorboard: bool = False          # 需要额外 pip install tensorboard

    seed: int = 2024
    device: str = "auto"               # auto / cpu / cuda / cuda:0
    max_steps: int = 0                 # >0 时覆盖 epochs，用于限时训练 / 冒烟测试

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


# --------------------------------------------------------------------------
# 预设：命令行里用 --preset 选择
# --------------------------------------------------------------------------
def tiny_config(vocab_size: int = 2000) -> Config:
    """冒烟测试用：几十秒能在 CPU 上跑完，用来验证代码链路。"""

    return Config(
        model=ModelConfig(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=64,
            n_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=128,
            dropout=0.0,
            max_len=128,
        ),
        train=TrainConfig(
            max_src_len=64,
            max_tgt_len=64,
            max_tokens=2048,
            max_sentences=64,
            epochs=3,
            warmup_steps=100,
            # Noam 的峰值学习率 = scale * d_model^-0.5 * warmup^-0.5。
            # d_model=64、warmup=100 时裸峰值是 1.25e-2 —— 太大了，模型会原地震荡。
            # 乘 0.08 把它压到 1e-3 量级，这是小模型能训得动的范围。
            lr_scale=0.08,
            eval_max_sentences=50,
            log_every_steps=10,
            amp=False,
        ),
    )


def small_config(vocab_size: int = 16000) -> Config:
    """单卡 4~8 GB 显存的笔记本配置。"""

    return Config(
        model=ModelConfig(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            d_ff=1024,
            dropout=0.1,
            max_len=192,
        ),
        train=TrainConfig(
            max_src_len=128,
            max_tgt_len=128,
            max_tokens=4096,
            warmup_steps=2000,
            # 裸峰值 1.4e-3，乘 0.7 落在 1e-3 附近
            lr_scale=0.7,
        ),
    )


def base_config(vocab_size: int = 16000) -> Config:
    """论文 base 规模，4090 单卡几小时的目标配置。"""

    return Config(
        model=ModelConfig(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=512,
            n_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            dropout=0.1,
            max_len=256,
        ),
        train=TrainConfig(
            max_src_len=192,
            max_tgt_len=192,
            # 动态 batching 的 token 预算。中英句对平均约 110 个 token（含 <bos>/<eos>），
            # 16384 大约等于 150 句一批；4090 的 24 GB 显存完全放得下。
            # 如果 OOM：先降到 8192，再用 accum_steps=2 把等效 batch size 补回来。
            max_tokens=16384,
            max_sentences=128,
            epochs=30,
            warmup_steps=4000,
            accum_steps=1,
            eval_max_sentences=400,
        ),
    )


PRESETS = {
    "tiny": tiny_config,
    "small": small_config,
    "base": base_config,
}


def build_config(
    preset: str = "base",
    vocab_size: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """按预设造配置，再套用命令行覆盖值。

    overrides 形如 {"model.d_model": 256, "train.accum_steps": 4}。
    """

    if preset not in PRESETS:
        raise ValueError(f"未知预设 {preset!r}，可选：{', '.join(PRESETS)}")
    cfg = PRESETS[preset](vocab_size) if vocab_size else PRESETS[preset]()

    for key, value in (overrides or {}).items():
        if "." not in key:
            raise ValueError(f"覆盖项 {key!r} 必须写成 'model.xxx' 或 'train.xxx'")
        section, attr = key.split(".", 1)
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, attr):
            raise ValueError(f"配置里没有 {key!r}")
        setattr(target, attr, value)
    return cfg

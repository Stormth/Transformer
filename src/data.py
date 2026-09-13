"""数据集、动态 padding、长度分桶批采样。

翻译任务的数据处理有三个容易踩的坑：

1. **padding 位置**：batch 内长短不齐，必须补齐到同一长度，
   同时记录真实长度，供注意力掩码使用。
2. **解码器输入与标签错位**：
       目标句      : 我 喜欢 这本书 <eos>
       解码器输入  : <bos> 我 喜欢 这本书     <- shift right
       标签        : 我 喜欢 这本书 <eos>     <- shift left
   训练时给解码器"上一个词"，让它预测"下一个词"。
3. **padding 浪费**：如果 batch 里混着 3 个词和 100 个词的句子，
   98% 的计算都在算 <pad>。按长度分桶 + token 预算组 batch 能显著提速。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .bpe import BPE

Pair = Tuple[str, str]


# ---------------------------------------------------------------------- #
# 读写
# ---------------------------------------------------------------------- #
def read_tsv(path: str | Path) -> List[Pair]:
    """读取 `源语言<TAB>目标语言` 的平行语料。

    用 utf-8-sig 读取：Windows 记事本/PowerShell 保存的 UTF-8 文件常带 BOM，
    不处理的话第一行的第一个字会变成乱码。
    """
    pairs: List[Pair] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{lineno} 期望 2 列（中文<TAB>英文），实际 {len(fields)} 列")
        src, tgt = fields[0].strip(), fields[1].strip()
        if src and tgt:
            pairs.append((src, tgt))
    return pairs


def write_tsv(pairs: Sequence[Pair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for src, tgt in pairs:
            f.write(f"{src}\t{tgt}\n")


# ---------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------- #
class TranslationDataset(Dataset):
    """把句对编码成 id。

    返回的每个样本是 dict：
        src    : [Ts] 源语言 id（不含 <bos>/<eos>）
        tgt    : [Tt] 目标语言 id（含 <eos>，不含 <bos>）

    <bos> 由模型用 shift_right 自动加在解码器输入最前面，
    这样"输入 / 标签"两套序列永远来自同一个张量，不容易错位。
    """

    def __init__(
        self,
        pairs: Sequence[Pair],
        tokenizer: BPE,
        max_src_len: int = 64,
        max_tgt_len: int = 64,
        filter_long: bool = True,
    ):
        self.tokenizer = tokenizer
        self.samples: List[Dict[str, List[int]]] = []
        self.skipped = 0
        for src_text, tgt_text in pairs:
            src_ids = tokenizer.encode(src_text)
            tgt_ids = tokenizer.encode(tgt_text, add_eos=True)
            if filter_long and (len(src_ids) > max_src_len or len(tgt_ids) > max_tgt_len):
                self.skipped += 1
                continue
            src_ids = src_ids[:max_src_len]
            tgt_ids = tgt_ids[:max_tgt_len]
            if not src_ids or not tgt_ids:
                self.skipped += 1
                continue
            self.samples.append({"src": src_ids, "tgt": tgt_ids})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.samples[index]

    @property
    def lengths(self) -> List[int]:
        """用于分桶的长度（按源+目标的 token 数估算计算量）。"""
        return [len(s["src"]) + len(s["tgt"]) for s in self.samples]


def make_collate_fn(pad_id: int = 0):
    """返回 collate 函数：把 list[dict] 拼成一个 batch。"""

    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_src = max(len(x["src"]) for x in batch)
        max_tgt = max(len(x["tgt"]) for x in batch)
        B = len(batch)

        src = torch.full((B, max_src), pad_id, dtype=torch.long)
        tgt = torch.full((B, max_tgt), pad_id, dtype=torch.long)
        src_lengths = torch.zeros(B, dtype=torch.long)
        tgt_lengths = torch.zeros(B, dtype=torch.long)

        for i, x in enumerate(batch):
            s, t = x["src"], x["tgt"]
            src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            tgt[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            src_lengths[i] = len(s)
            tgt_lengths[i] = len(t)

        return {"src": src, "tgt": tgt, "src_lengths": src_lengths, "tgt_lengths": tgt_lengths}

    return collate


# ---------------------------------------------------------------------- #
# 批采样：长度分桶 + token 预算
# ---------------------------------------------------------------------- #
class LengthBucketBatchSampler(Sampler):
    """把长度接近的样本放进同一个 batch。

    做法（业界常用做法之一）：
        1. 按长度排序，避免长句和短句混在一起；
        2. 顺序切 batch：一个 batch 的 token 总量不超过 max_tokens（也不超过 batch_size）；
        3. 每个 epoch 打乱 batch 之间的顺序（batch 内部已经按长度排好了，不能再乱）。

    每个 epoch 会重新加一点随机扰动（jitter），避免模型长期看到完全固定的组合。
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int = 64,
        max_tokens: int = 4096,
        shuffle: bool = True,
        seed: int = 0,
        jitter: float = 0.05,
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.jitter = jitter
        self.epoch = 0
        self._batches: List[List[int]] = []
        self._build()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._build()

    def _build(self) -> None:
        rng = random.Random(self.seed + self.epoch)
        n = len(self.lengths)
        # 长度 + 小幅随机扰动，兼顾"桶内长度接近"与"每个 epoch 略有不同"
        keyed = sorted(
            range(n),
            key=lambda i: self.lengths[i] * (1.0 + rng.uniform(-self.jitter, self.jitter)),
        )
        batches: List[List[int]] = []
        cur: List[int] = []
        cur_max = 0
        for i in keyed:
            L = self.lengths[i]
            new_max = max(cur_max, L)
            if cur and (len(cur) >= self.batch_size or new_max * (len(cur) + 1) > self.max_tokens):
                batches.append(cur)
                cur, cur_max = [], 0
                new_max = L
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)

        if self.shuffle:
            rng.shuffle(batches)
        self._batches = batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def create_dataloader(
    dataset: TranslationDataset,
    batch_size: int = 64,
    max_tokens: int = 4096,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_id: int = 0,
    use_bucket_sampler: bool = True,
    seed: int = 0,
) -> Tuple[DataLoader, LengthBucketBatchSampler | None]:
    """构造 DataLoader。训练集用长度分桶采样，验证/测试集按顺序即可。"""
    collate = make_collate_fn(pad_id)
    if shuffle and use_bucket_sampler:
        sampler = LengthBucketBatchSampler(
            dataset.lengths, batch_size=batch_size, max_tokens=max_tokens, shuffle=True, seed=seed
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
        )
        return loader, sampler
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        drop_last=False,
    )
    return loader, None

"""批处理：长度分桶 + 动态 padding。

两个让训练快起来的技巧，都跟"句子长度不一样"有关：

1. **动态 padding**：一个 batch 里只补到本 batch 最长的那句，而不是全局最大长度。
   一批里全是短句时，pad 少、算得快。代价是每个 batch 形状不同，几乎可以忽略。

2. **长度分桶**：把长度接近的句子放进同一个 batch。
   否则一批里 100 个短句配 1 个长句，所有短句都得补到长句的长度，白算一大半。
   分桶会破坏随机性（一批里全是长度相近的句子），
   我们用"桶内打乱 + batch 顺序打乱"来弥补，这是标准做法。

最后是 token 预算：一个 batch 里 token 总数不超过 max_tokens，
batch 大小随句子长度自动伸缩，显存占用稳定 ——
固定 batch size 是 OOM 最常见的元凶。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class ParallelTextDataset(Dataset):
    """读 `corpus.py` 落盘的拍平张量，按句取样本。"""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在，先跑 python -m nmt.corpus 准备数据")
        payload = torch.load(path, map_location="cpu")
        self.src: torch.Tensor = payload["src"]
        self.src_lengths: torch.Tensor = payload["src_lengths"]
        self.tgt: torch.Tensor = payload["tgt"]
        self.tgt_lengths: torch.Tensor = payload["tgt_lengths"]
        self.src_offsets = self._offsets(self.src_lengths)
        self.tgt_offsets = self._offsets(self.tgt_lengths)

    @staticmethod
    def _offsets(lengths: torch.Tensor) -> torch.Tensor:
        """长度数组 -> 前缀和，用来把"第 i 句"翻译成区间 [start, end)。"""

        offsets = torch.zeros(len(lengths) + 1, dtype=torch.long)
        torch.cumsum(lengths.long(), dim=0, out=offsets[1:])
        return offsets

    def __len__(self) -> int:
        return len(self.src_lengths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_start, src_end = int(self.src_offsets[index]), int(self.src_offsets[index + 1])
        tgt_start, tgt_end = int(self.tgt_offsets[index]), int(self.tgt_offsets[index + 1])
        return self.src[src_start:src_end].long(), self.tgt[tgt_start:tgt_end].long()

    def length(self, index: int) -> Tuple[int, int]:
        """句子的 (英文长度, 德文长度)，分桶时用。"""

        return int(self.src_lengths[index]), int(self.tgt_lengths[index])

    @property
    def total_tokens(self) -> int:
        return int(self.src_lengths.sum()) + int(self.tgt_lengths.sum())

    def decode_pair(self, index: int, tokenizer) -> Tuple[str, str]:
        """取回原文（调试、看预测时很有用）。"""

        src_ids, tgt_ids = self[index]
        return tokenizer.decode(src_ids.tolist()), tokenizer.decode(tgt_ids.tolist())


class LengthBucketSampler:
    """先按长度分桶，再按 token 预算组 batch。用法和普通 sampler 一样。"""

    def __init__(
        self,
        dataset: ParallelTextDataset,
        max_tokens: int = 8192,
        max_sentences: int = 128,
        bucket_size: int = 64,
        shuffle: bool = True,
        seed: int = 2024,
        max_src_len: int = 0,
        max_tgt_len: int = 0,
    ) -> None:
        self.dataset = dataset
        self.max_tokens = max(1, max_tokens)
        self.max_sentences = max(1, max_sentences)
        self.bucket_size = max(1, bucket_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # 过滤掉超出长度上限的句子（数据准备时已经卡过一道，这里是双保险）
        indices: List[int] = []
        costs: List[int] = []
        for index in range(len(dataset)):
            src_len, tgt_len = dataset.length(index)
            if max_src_len and src_len > max_src_len:
                continue
            if max_tgt_len and tgt_len > max_tgt_len:
                continue
            indices.append(index)
            # 用 src+tgt 近似一句的算力开销（注意力是 O(T^2)，先不细算）
            costs.append(src_len + tgt_len)
        self.indices = indices
        self.costs = costs
        self._cache: Optional[List[List[int]]] = None

    # ------------------------------------------------------------ 组 batch
    def _build(self) -> List[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # 1) 全局按长度排序：排序后长度相近的句子天然挨在一起
        order = sorted(range(len(self.indices)), key=lambda i: self.costs[i])

        # 2) 切成一桶一桶，桶内打乱（这样同一批里不会总是同样的句子组合）
        buckets: List[List[int]] = []
        for start in range(0, len(order), self.bucket_size):
            bucket = order[start : start + self.bucket_size]
            if self.shuffle:
                permutation = torch.randperm(len(bucket), generator=generator).tolist()
                bucket = [bucket[i] for i in permutation]
            buckets.append(bucket)

        # 3) 按 token 预算把句子装进 batch
        batches: List[List[int]] = []
        current: List[int] = []
        current_max = 0
        for bucket in buckets:
            for position in bucket:
                cost = self.costs[position]
                new_max = max(current_max, cost)
                # 补到本 batch 最长句之后的总 token 数 = 最长句 * 句数
                projected = new_max * (len(current) + 1)
                if current and (projected > self.max_tokens or len(current) >= self.max_sentences):
                    batches.append(current)
                    current, current_max = [], 0
                    new_max = cost
                current.append(position)
                current_max = new_max
        if current:
            batches.append(current)

        # 4) 打乱 batch 顺序
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in permutation]
        return batches

    def set_epoch(self, epoch: int) -> None:
        """每个 epoch 换一个种子，让分桶洗牌每次都不一样。"""

        self.epoch = epoch
        self._cache = None

    def __len__(self) -> int:
        if self._cache is None:
            self._cache = self._build()
        return len(self._cache)

    def __iter__(self) -> Iterator[List[int]]:
        if self._cache is None:
            self._cache = self._build()
        for batch in self._cache:
            yield [self.indices[position] for position in batch]

    # ------------------------------------------------------------ 信息
    @property
    def num_pairs(self) -> int:
        return len(self.indices)


def collate_batch(
    samples: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """把一个 batch 的变长序列补成规则张量。

    返回：
        src          [B, S]  英文 id（含 <eos>）
        tgt          [B, T]  德文 id（含 <bos> 与 <eos>）
        src_lengths  [B]
        tgt_lengths  [B]
    """

    src_list, tgt_list = zip(*samples)
    src_lengths = torch.tensor([len(x) for x in src_list], dtype=torch.long)
    tgt_lengths = torch.tensor([len(x) for x in tgt_list], dtype=torch.long)
    max_src = int(src_lengths.max())
    max_tgt = int(tgt_lengths.max())

    src = torch.full((len(src_list), max_src), pad_id, dtype=torch.long)
    tgt = torch.full((len(tgt_list), max_tgt), pad_id, dtype=torch.long)
    for row, (src_ids, tgt_ids) in enumerate(zip(src_list, tgt_list)):
        src[row, : len(src_ids)] = src_ids
        tgt[row, : len(tgt_ids)] = tgt_ids
    return {"src": src, "tgt": tgt, "src_lengths": src_lengths, "tgt_lengths": tgt_lengths}


def build_dataloader(
    dataset: ParallelTextDataset,
    pad_id: int,
    max_tokens: int,
    max_sentences: int,
    bucket_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 2024,
    max_src_len: int = 0,
    max_tgt_len: int = 0,
) -> Tuple[DataLoader, LengthBucketSampler]:
    """组装 DataLoader：采样器负责组 batch，collate 负责 padding。"""

    sampler = LengthBucketSampler(
        dataset,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        bucket_size=bucket_size,
        shuffle=shuffle,
        seed=seed,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
        num_workers=num_workers,
    )
    return loader, sampler

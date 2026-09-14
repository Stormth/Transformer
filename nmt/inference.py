"""推理入口：把"一句英文"变成"一句德语"。

命令行翻译、评测、可视化都走这里，保证训练和推理用的是同一套前处理
（少一个空格、少一个标点，分数就会莫名其妙地掉）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .bpe import BPE
from .checkpoint import config_from_checkpoint, load_checkpoint
from .decoding import beam_search_decode, greedy_decode
from .dataset import collate_batch, truncate_pair
from .masks import make_encoder_attn_mask
from .model import Transformer
from .utils import get_logger, resolve_device

logger = get_logger()


def select_eval_indices(total: int, limit: int) -> List[int]:
    """从整个 split 里等间隔抽 limit 句。

    为什么不直接取前 limit 句？因为新闻语料的顺序和版面有关，
    取开头几十句可能全是"国际版"的文章，BLEU 会随抽样位置漂。
    等间隔抽样便宜又无偏。
    """

    if limit <= 0 or limit >= total:
        return list(range(total))
    stride = total / limit
    return sorted({int(index * stride) for index in range(limit)})


@torch.no_grad()
def translate_dataset(
    model: Transformer,
    dataset,
    tokenizer: BPE,
    device: torch.device,
    indices: Optional[Sequence[int]] = None,
    batch_size: int = 32,
    beam_size: int = 1,
    max_tokens: int = 4096,
    length_penalty: float = 0.6,
    max_len: int = 192,
    adaptive_max_len: bool = True,
    progress=None,
) -> List[str]:
    """对数据集里的若干句做翻译，返回和 indices 顺序一致的译文列表。

    adaptive_max_len：按输入长度动态限制生成长度（默认开）。
    为什么需要它？译文长度通常和原文相当，但**训练不足的模型不会输出 <eos>**，
    于是每句都会硬生成到 max_len=192 才停 —— 评测 2 万句时这会让耗时翻好几倍，
    而且多出来的部分全是废话。这里按"原文长度的 2 倍 + 10"封顶，
    既能容纳正常的译文膨胀，又能把这些无意义的步数砍掉。
    想看模型"到底会不会自己停"，把它关掉即可。
    """

    model.eval()
    selected = list(range(len(dataset))) if indices is None else list(indices)
    pad_id = tokenizer.pad_id

    # 按长度排序后按 token 预算切 batch：评测时这一步能省掉大量 padding
    order = sorted(selected, key=lambda i: sum(dataset.length(i)))
    batches: List[List[int]] = []
    current: List[int] = []
    current_max = 0
    for index in order:
        cost = sum(dataset.length(index))
        new_max = max(current_max, cost)
        if current and new_max * (len(current) + 1) > max_tokens:
            batches.append(current)
            current, current_max = [], 0
            new_max = cost
        current.append(index)
        current_max = new_max
    if current:
        batches.append(current)

    hypotheses: Dict[int, str] = {}
    truncate_limit = int(getattr(model.config, "max_len", 512))
    truncated = 0
    for done, batch_indices in enumerate(batches, start=1):
        samples = []
        for index in batch_indices:
            src_ids, tgt_ids = dataset[index]
            if src_ids.numel() > truncate_limit:
                truncated += 1
            samples.append(truncate_pair(src_ids, tgt_ids, truncate_limit))
        collated = collate_batch(samples, pad_id)
        src = collated["src"].to(device)
        src_mask = make_encoder_attn_mask(src, pad_id)
        step_max_len = max_len
        if adaptive_max_len:
            source_len = int(collated["src"].size(1))
            step_max_len = min(max_len, max(16, source_len * 2 + 10))

        if beam_size <= 1:
            sequences = greedy_decode(
                model, src, src_mask,
                bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, pad_id=pad_id,
                max_len=step_max_len,
            )
        else:
            sequences, _ = beam_search_decode(
                model, src, src_mask,
                bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, pad_id=pad_id,
                beam_size=beam_size, max_len=step_max_len, length_penalty=length_penalty,
            )
        for index, sequence in zip(batch_indices, sequences):
            hypotheses[index] = tokenizer.decode(sequence)
        if progress is not None:
            progress(done, len(batches))

    if truncated:
        logger.warning(f"有 {truncated} 句原文超过了位置编码上限 {truncate_limit}，已截断")
    return [hypotheses[index] for index in selected]


def find_vocab_path(checkpoint_path: Path, recorded: str) -> Path:
    """按优先级找一个能用的词表文件。"""

    candidates = [
        Path(recorded),                       # checkpoint 里记的路径
        checkpoint_path.parent / "vocab.json",  # 和 checkpoint 放一起（训练脚本会复制一份）
        Path("data/ready/vocab.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "找不到 vocab.json。训练脚本会把词表复制到 checkpoint 目录，"
        "如果你只拷了 .pt 文件，请把 vocab.json 一起拷过来。"
    )


class Translator:
    """加载 checkpoint 后反复调用的翻译器。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        beam_size: int = 4,
        length_penalty: float = 0.6,
        max_len: int = 192,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        self.checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        self.config = config_from_checkpoint(self.checkpoint)

        self.tokenizer = BPE.load(find_vocab_path(checkpoint_path, self.checkpoint.vocab_path))
        self.model = Transformer(self.config.model)
        self.model.load_state_dict(self.checkpoint.model)
        self.model.to(self.device).eval()

        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.max_len = max_len

    # ------------------------------------------------------------------ 编码
    def encode(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.tokenizer.pad_id
        # 位置编码有长度上限，超长的输入必须截断（保留末尾的 <eos>），
        # 否则 forward 会直接抛 ValueError。评测时过长句被截断是常规做法。
        limit = self.config.model.max_len
        sequences = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_eos=True)
            if len(ids) > limit:
                ids = ids[: limit - 1] + [ids[-1]]
            sequences.append(ids)
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
        batch = torch.full((len(sequences), int(lengths.max())), pad_id, dtype=torch.long)
        for row, ids in enumerate(sequences):
            batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        batch = batch.to(self.device)
        return batch, make_encoder_attn_mask(batch, pad_id)

    # ------------------------------------------------------------------ 翻译
    def translate(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        beam_size: Optional[int] = None,
    ) -> List[str]:
        """按 batch 翻译一批句子。句子按长度排序后分堆，能省不少 padding。"""

        beam_size = self.beam_size if beam_size is None else beam_size
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        outputs: List[str] = [""] * len(texts)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            batch_texts = [texts[i] for i in chunk]
            src, src_mask = self.encode(batch_texts)

            if beam_size <= 1:
                sequences = greedy_decode(
                    self.model, src, src_mask,
                    bos_id=self.tokenizer.bos_id,
                    eos_id=self.tokenizer.eos_id,
                    pad_id=self.tokenizer.pad_id,
                    max_len=self.max_len,
                )
            else:
                sequences, _ = beam_search_decode(
                    self.model, src, src_mask,
                    bos_id=self.tokenizer.bos_id,
                    eos_id=self.tokenizer.eos_id,
                    pad_id=self.tokenizer.pad_id,
                    beam_size=beam_size,
                    max_len=self.max_len,
                    length_penalty=self.length_penalty,
                )
            for position, sequence in zip(chunk, sequences):
                outputs[position] = self.tokenizer.decode(sequence)
        return outputs

    def translate_one(self, text: str, beam_size: Optional[int] = None) -> str:
        return self.translate([text], beam_size=beam_size)[0]

    # ------------------------------------------------------------------ 展示
    def describe(self) -> str:
        return (
            f"设备 {self.device} | {self.tokenizer.summary()}\n"
            f"{self.model.describe()}\n"
            f"checkpoint：第 {self.checkpoint.epoch} 轮，验证集最优 "
            f"{self.checkpoint.best_metric} {self.checkpoint.best_score:.2f}"
        )

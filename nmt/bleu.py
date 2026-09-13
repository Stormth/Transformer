"""BLEU 与 chrF 评测，纯标准库手写。

BLEU 在算什么？

它拿机器译文和参考译文比 n-gram（连续 n 个词）的重合度：

    BLEU = BP · exp( Σ_{n=1..4} w_n · log p_n )

    p_n   = 机器译文里"在参考译文中出现过"的 n-gram 占的比例（截断计数，见下）
    w_n   = 通常取 1/4，四元均匀
    BP    = 长度惩罚，译文比参考短就乘一个小于 1 的因子

三个必须知道的细节：

1. **截断计数**：参考里 "the" 只出现 3 次，机器译文出现了 10 次，
   那么最多只算命中 3 次。否则刷同一个词就能刷高分数。
2. **BP 存在的理由**：不惩罚长度的话，"the" 这一个高精度短句就能骗到高分。
   BP = min(1, exp(1 - 参考长度/译文长度))。
3. **切词方式必须固定**：BLEU 不是"跨语言可比"的指标。
   同一个模型，换个切词方式（整词 / 子词 / 字符）分数就变了。
   本项目英德两边都用"转小写 + 只保留字母数字串"的简化 13a 分词，
   和 sacrebleu 的 `13a` 基本一致，所以你算出来的数字大致能和别人对齐。

chrF 则是按**字符 n-gram** 算 F 值，对形态变化多的语言（德语的名词复数、
形容词词尾、动词变位）更宽容：词形错一半时 BLEU 会掉很多，chrF 还能反映出
"主干译对了"。训练早期 BLEU 还是 0 的时候，chrF 往往已经在动了 ——
用它观察"有没有在学"更灵敏。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# 13a 思路的简化版：Unicode 字母串 + 数字串；标点整个丢掉。
# （ä ö ü ß é 都会被 [^\W\d_] 匹配到，不会被误伤）
_WORD = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)*")


def tokenize_for_bleu(text: str) -> List[str]:
    """BLEU 用的切分：转小写，只保留字母串和数字串，标点丢掉。"""

    return _WORD.findall(text.lower())


def _extract_ngrams(tokens: Sequence[str], order: int) -> Counter:
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


@dataclass
class BLEUScore:
    bleu: float
    precisions: List[float] = field(default_factory=list)
    bp: float = 1.0
    sys_len: int = 0
    ref_len: int = 0
    ratio: float = 0.0

    def summary(self) -> str:
        precisions = "/".join(f"{p:.1%}" for p in self.precisions)
        return (
            f"BLEU-4 {self.bleu:.2f}  "
            f"（1~4 gram 命中率 {precisions}，长度比 {self.ratio:.2f}，BP {self.bp:.3f}）"
        )


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
    max_order: int = 4,
) -> BLEUScore:
    """语料级 BLEU。

    注意是**语料级**：先统计整个测试集的 n-gram 命中数，再算一次比值，
    而不是对每句话算 BLEU 再平均（后者是另一个指标，数值会低很多）。
    """

    if len(hypotheses) != len(references):
        raise ValueError("译文和参考译文数量不一致")
    if not hypotheses:
        return BLEUScore(0.0)

    matches = [0] * (max_order + 1)
    totals = [0] * (max_order + 1)
    hyp_len_total = 0
    ref_len_total = 0

    for hypothesis, reference in zip(hypotheses, references):
        hyp_tokens = tokenize_for_bleu(hypothesis)
        ref_tokens = tokenize_for_bleu(reference)
        hyp_len_total += len(hyp_tokens)
        ref_len_total += len(ref_tokens)

        for order in range(1, max_order + 1):
            hyp_ngrams = _extract_ngrams(hyp_tokens, order)
            ref_ngrams = _extract_ngrams(ref_tokens, order)
            # 截断计数：命中数不能超过参考里的出现次数
            clipped = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
            matches[order] += clipped
            totals[order] += sum(hyp_ngrams.values())

    precisions = []
    for order in range(1, max_order + 1):
        if totals[order] == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches[order] / totals[order])

    # 长度惩罚：译文短于参考要罚，长了不奖
    if hyp_len_total == 0 or ref_len_total == 0:
        bp = 0.0
    elif hyp_len_total > ref_len_total:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len_total / hyp_len_total)

    if min(precisions) > 0:
        score = bp * math.exp(sum(math.log(p) for p in precisions) / max_order)
    else:
        score = 0.0

    ratio = hyp_len_total / ref_len_total if ref_len_total else 0.0
    return BLEUScore(
        bleu=score * 100,
        precisions=precisions,
        bp=bp,
        sys_len=hyp_len_total,
        ref_len=ref_len_total,
        ratio=ratio,
    )


def _char_ngrams(text: str, order: int) -> Counter:
    """按字符切 n-gram（去掉空白）。"""

    stripped = "".join(text.split())
    return Counter(stripped[i : i + order] for i in range(len(stripped) - order + 1))


def corpus_chrf(
    hypotheses: Sequence[str],
    references: Sequence[str],
    char_order: int = 6,
    beta: float = 2.0,
) -> float:
    """chrF：字符级 F 值（beta=2 时更看重召回）。返回 0~100。"""

    if len(hypotheses) != len(references):
        raise ValueError("译文和参考译文数量不一致")
    if not hypotheses:
        return 0.0

    scores: List[float] = []
    for order in range(1, char_order + 1):
        matches = 0
        hyp_total = 0
        ref_total = 0
        for hypothesis, reference in zip(hypotheses, references):
            hyp_ngrams = _char_ngrams(hypothesis, order)
            ref_ngrams = _char_ngrams(reference, order)
            matches += sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
            hyp_total += sum(hyp_ngrams.values())
            ref_total += sum(ref_ngrams.values())
        precision = matches / hyp_total if hyp_total else 0.0
        recall = matches / ref_total if ref_total else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            beta_sq = beta * beta
            scores.append((1 + beta_sq) * precision * recall / (beta_sq * precision + recall))
    return sum(scores) / len(scores) * 100


def evaluate_all(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> Dict[str, float]:
    """一次算出训练监控常用的几个数。"""

    bleu = corpus_bleu(hypotheses, references)
    return {
        "bleu": bleu.bleu,
        "chrf": corpus_chrf(hypotheses, references),
        "length_ratio": bleu.ratio,
    }

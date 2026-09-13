"""纯 Python 实现的 BLEU（不依赖 sacrebleu / numpy）。

BLEU 回答的问题是：机器译文里的 n-gram，有多少比例出现在参考译文中？

    BLEU = BP · exp( Σ w_n · log p_n )        n = 1..4

    p_n : 修正后的 n-gram 精确率（分子要"截断"：
          n-gram 在译文中出现 3 次而参考译文里只出现 2 次，最多只算 2 次）
    BP  : 简短惩罚 brevity penalty = min(1, exp(1 - 参考长度/译文长度))
          否则模型只要输出"the"就能得到很高的精确率

关于平滑：如果某个长度的 n-gram 一个都没匹配上，p_n = 0，整个 BLEU = 0。
这里采用 sacrebleu 的 add-k 平滑（默认 k=0，即 exp 平滑的简化版），
避免长句偶尔无 4-gram 匹配时分数被"一刀切"归零。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

_PUNCT_RE = re.compile(r"([.,!?;:()\[\]{}\"'`])")


def tokenize(text: str, lang: str = "en", lowercase: bool = True) -> List[str]:
    """轻量分词：英文按空格 + 标点切分（接近 mteval-13a），中文按字切分。"""
    if lang.startswith("zh"):
        return [c for c in text if not c.isspace()]
    text = text.strip()
    if lowercase:
        text = text.lower()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return [t for t in text.split() if t]


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]] | Sequence[str],
    max_order: int = 4,
    smooth_k: float = 0.0,
    lowercase: bool = True,
    lang: str = "en",
) -> float:
    """语料级 BLEU（0~100）。

    参数
    ----
    hypotheses : 机器译文列表
    references : 参考译文；可以是「每句一个参考」的字符串列表，
                 也可以是「每句多个参考」的 list[list[str]]
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses 与 references 数量不一致")
    if not hypotheses:
        return 0.0

    # 统一成 list[list[str]]，方便支持多参考译文
    refs: List[List[str]] = []
    for r in references:
        refs.append(list(r) if isinstance(r, (list, tuple)) else [r])  # type: ignore[arg-type]

    clipped = [0] * (max_order + 1)
    total = [0] * (max_order + 1)
    ref_len_sum = 0
    hyp_len_sum = 0

    for hyp, ref_group in zip(hypotheses, refs):
        hyp_tokens = tokenize(hyp, lang=lang, lowercase=lowercase)
        ref_tokens = [tokenize(r, lang=lang, lowercase=lowercase) for r in ref_group]
        hyp_len_sum += len(hyp_tokens)

        # 简短惩罚用的参考长度：取与译文长度最接近的参考句长度（sacrebleu 的做法）
        lengths = [len(r) for r in ref_tokens]
        ref_len_sum += min(lengths, key=lambda l: (abs(l - len(hyp_tokens)), l))

        for n in range(1, max_order + 1):
            hyp_ngrams = _ngrams(hyp_tokens, n)
            total[n] += max(0, len(hyp_tokens) - n + 1)
            max_ref = Counter()
            for rt in ref_tokens:
                for gram, c in _ngrams(rt, n).items():
                    max_ref[gram] = max(max_ref[gram], c)
            clipped[n] += sum(min(c, max_ref[gram]) for gram, c in hyp_ngrams.items())

    if hyp_len_sum == 0:
        return 0.0

    # 各阶精确率（带平滑）
    precisions = []
    for n in range(1, max_order + 1):
        if total[n] == 0:
            continue
        num = clipped[n] + smooth_k
        den = total[n] + smooth_k
        if num == 0:
            return 0.0
        precisions.append(num / den)
    if not precisions:
        return 0.0

    weights = [1.0 / len(precisions)] * len(precisions)
    score = sum(w * math.log(p) for w, p in zip(weights, precisions))

    bp = 1.0
    if hyp_len_sum < ref_len_sum:
        bp = math.exp(1.0 - ref_len_sum / hyp_len_sum)

    return 100.0 * bp * math.exp(score)


def corpus_bleu_pair(
    pairs: Iterable[tuple[str, str]],
    lowercase: bool = True,
) -> float:
    """便捷函数：输入 (hypothesis, reference) 迭代器。"""
    hyps, refs = [], []
    for h, r in pairs:
        hyps.append(h)
        refs.append(r)
    return corpus_bleu(hyps, refs, lowercase=lowercase)

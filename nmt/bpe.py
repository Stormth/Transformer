"""手写 BPE 分词器（Byte-Pair Encoding）。

为什么需要分词？

* 按"词"切：英文一个词一个 token，但遇到没见过的词（人名、新词、变形）就只能
  吐 `<unk>`，词表还会膨胀到几十万。
* 按"字符"切：永远不会 <unk>，但序列变得很长，注意力成本是 O(T^2)，
  而且每个字符带的信息太少。
* BPE 是折中：从字符出发，反复把"最常一起出现的相邻符号"合并成新符号。
  常见的词会变成一个 token（the / 一个），罕见词会被拆成几个子词片段
  （unbelievable -> un + believ + able），既压短了序列，又几乎没有 <unk>。

中英差异（本项目的 pre_tokenize 就处理这件事）：

* 英文按空格天然分词，标点单独成 token；
* 中文没有空格，所以把**连续的汉字串当作一个"词"**，再让 BPE 在这串字内部
  学习常见的字组合（"我们" "可以" "问题" 这类高频组合会被合并成一个 token）。

为什么需要词尾标记 `</w>`？

    因为解码时要把子词拼回原词。假设 "committee" 被切成 c + o + mmit + tee，
    如果直接把 token 用空格连起来，就会得到 "c o mmit tee" —— 词被拆散了。
    标准做法（BPE 原论文）是给每个词的**最后一个字符**加上 `</w>` 标记：

        训练时： "committee" -> c o m m i t t e e</w>
        解码时： 遇到带 </w> 的 token 就知道"这个词到这儿结束"，
                 把它前面攒着的碎片直接连起来，再加一个空格。

    这样做还有个附带好处：合并规则自然学到 "e</w>" "tion</w>" 这类词尾片段，
    对英文形态变化的建模更友好。中文不需要这个标记（汉字之间本来就不加空格），
    所以只对不含汉字的"词"加，避免词表被汉字的两份形式浪费掉。

训练算法（频率加权的贪心合并）：

    1. 统计所有"词"的出现次数；
    2. 把每个词表示成字符序列；
    3. 反复：找出当前出现次数最多的相邻符号对 (a, b)，合并成 "ab"，
       并把这一步记录进合并规则表；
    4. 直到词表达到目标大小，或者最高频的符号对出现次数低于阈值。

关键工程点：每一步只重算**受影响的词**的符号对，而不是重扫全部语料。
否则十几万句语料要做上万次合并，纯 Python 会慢到不可用。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 特殊符号
# --------------------------------------------------------------------------
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS: Tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# --------------------------------------------------------------------------
# 文本切分
# --------------------------------------------------------------------------
# 汉字（含扩展 A 区与兼容区）和日文假名一起视为"无空格语言"字符
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff"

# 四类候选，按优先级排列：
#   1. 连续汉字串        他明天在学校
#   2. 英文单词（含撇号） don't
#   3. 数字（含小数/千分位） 3.14
#   4. 其它任意单个非空白字符（标点、符号）
_TOKEN_RE = re.compile(rf"[{_CJK}]+|[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,]\d+)*|\S")

# 需要清掉的不可见字符（零宽空格、BOM、行分隔符等）
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u2028\u2029\ufeff]")

_CJK_CHAR = re.compile(rf"[{_CJK}]")

# 词尾标记：加在每个"英文词"最后一个字符后面，用来还原词边界
WORD_END = "</w>"


def _word_to_symbols(word: str) -> List[str]:
    """把一个"词"变成 BPE 的初始符号序列。

    不含汉字的词给最后一个字符加上 `</w>`；中文串保持逐字，不加标记。
    """

    symbols = list(word)
    if symbols and not _CJK_CHAR.search(word):
        symbols[-1] = symbols[-1] + WORD_END
    return symbols


def normalize_text(text: str) -> str:
    """统一空白、清掉不可见字符。

    刻意**不做**全角转半角：中文全角标点（，。！？）是正文的一部分，
    转成半角会让输出变得不像中文。
    """

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = _CONTROL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def pre_tokenize(text: str) -> List[str]:
    """切分成"词"级别的单元，交给 BPE 在词内部继续合并。"""

    return _TOKEN_RE.findall(text)


def contains_cjk(text: str) -> bool:
    """判断这段文字里有没有汉字（数据清洗时用来验语言）。"""

    return bool(_CJK_CHAR.search(text))


def cjk_ratio(text: str) -> float:
    """汉字占非空白字符的比例。用它过滤"中文侧其实是英文"的脏数据。"""

    stripped = re.sub(r"\s", "", text)
    if not stripped:
        return 0.0
    return len(_CJK_CHAR.findall(stripped)) / len(stripped)


# --------------------------------------------------------------------------
# BPE 正文
# --------------------------------------------------------------------------
class BPE:
    """一个够用的 BPE 实现：训练、编码、解码、存取。"""

    def __init__(
        self,
        merges: Sequence[Tuple[str, str]],
        base_chars: Optional[Sequence[str]] = None,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        self.special_tokens = list(special_tokens)
        self.merges: List[Tuple[str, str]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.merge_rank: Dict[Tuple[str, str], int] = {
            pair: index for index, pair in enumerate(self.merges)
        }

        if base_chars is None:
            chars = set()
            for a, b in self.merges:
                chars.update(a)
                chars.update(b)
            base_chars = sorted(chars)
        self.base_chars: List[str] = list(base_chars)

        # 按"特殊符号 -> 基础字符 -> 合并产物"的顺序编号。
        # 顺序一旦定下就不能变，否则词表 id 会和训练好的权重对不上。
        self.token_to_id: Dict[str, int] = {}
        for token in self.special_tokens:
            self._add_token(token)
        for char in self.base_chars:
            self._add_token(char)
        for a, b in self.merges:
            self._add_token(a + b)

        self.id_to_token: List[str] = [""] * len(self.token_to_id)
        for token, index in self.token_to_id.items():
            self.id_to_token[index] = token

    # ------------------------------------------------------------ 基本属性
    def _add_token(self, token: str) -> None:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    # ------------------------------------------------------------ 训练
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 16000,
        min_frequency: int = 2,
        max_unique_words: int = 300000,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> "BPE":
        """在语料上训出合并规则。

        texts: 可迭代的句子（中英混在一起训练，得到共享词表）
        vocab_size: 目标词表大小（含特殊符号）
        min_frequency: 出现次数低于它的符号对不再合并
        max_unique_words: 只保留最高频的这些"词"，控制训练时间
        """

        # --- 1. 统计词频 ---
        word_counts: Counter[str] = Counter()
        char_counts: Counter[str] = Counter()
        for text in texts:
            for word in pre_tokenize(text):
                word_counts[word] += 1
                # 统计的是"带词尾标记"的符号，这样每个字符的收尾形式也在词表里，
                # 否则一个词以罕见字符结尾时会掉进 <unk>
                char_counts.update(_word_to_symbols(word))

        if not word_counts:
            raise ValueError("语料是空的，没法训练 BPE")

        # --- 2. 词表预算：先给基础字符留位置，剩下的才给合并规则 ---
        budget_for_chars = max(1, vocab_size - len(SPECIAL_TOKENS))
        if len(char_counts) > budget_for_chars:
            # 字符太多（罕见符号）时，只保留最高频的那些，其余的编码成 <unk>
            keep = [ch for ch, _ in char_counts.most_common(budget_for_chars)]
            base_chars = sorted(keep)
        else:
            base_chars = sorted(char_counts)
        # 基础字符占掉多少预算，剩下的才留给合并规则
        budget_for_chars = len(base_chars)

        num_merges = max(0, vocab_size - len(SPECIAL_TOKENS) - budget_for_chars)

        # --- 3. 把词表示成字符序列（只保留够高频的词） ---
        top_words = word_counts.most_common(max_unique_words)
        words: List[List[str]] = [_word_to_symbols(word) for word, _ in top_words]
        freqs: List[int] = [count for _, count in top_words]

        pair_counts: Counter[Tuple[str, str]] = Counter()
        pair_words: Dict[Tuple[str, str], set] = defaultdict(set)

        def add_pairs(index: int, sign: int) -> None:
            word = words[index]
            freq = freqs[index] * sign
            for position in range(len(word) - 1):
                pair = (word[position], word[position + 1])
                pair_counts[pair] += freq
                if sign > 0:
                    pair_words[pair].add(index)
                else:
                    pair_words[pair].discard(index)

        for index in range(len(words)):
            add_pairs(index, sign=1)

        def merge_word(word: List[str], pair: Tuple[str, str]) -> List[str]:
            """把 word 里所有相邻的 pair 合并。"""

            merged: List[str] = []
            position = 0
            while position < len(word):
                if position < len(word) - 1 and word[position] == pair[0] and word[position + 1] == pair[1]:
                    merged.append(pair[0] + pair[1])
                    position += 2
                else:
                    merged.append(word[position])
                    position += 1
            return merged

        # --- 4. 贪心合并 ---
        merges: List[Tuple[str, str]] = []
        for step in range(num_merges):
            if not pair_counts:
                break
            # 取出现次数最多的对；次数相同时按符号顺序决定，保证可复现
            best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if best_count < min_frequency:
                break

            merges.append(best_pair)

            # 只更新包含这个 pair 的词 —— 这是整个实现能跑得动的关键
            for index in list(pair_words[best_pair]):
                add_pairs(index, sign=-1)          # 先撤掉旧符号对的计数
                words[index] = merge_word(words[index], best_pair)
                add_pairs(index, sign=1)           # 再加入合并后的新计数
            pair_counts.pop(best_pair, None)
            pair_words.pop(best_pair, None)

            if progress is not None and (step + 1) % 1000 == 0:
                progress(step + 1, num_merges)

        if progress is not None:
            progress(len(merges), num_merges)

        return cls(merges, base_chars=base_chars)

    # ------------------------------------------------------------ 编码
    def _encode_word(self, word: str) -> List[str]:
        """对单个"词"做 BPE：从字符出发，不断套用优先级最高的合并规则。"""

        symbols: List[str] = _word_to_symbols(word)
        if len(symbols) < 2:
            return symbols

        while True:
            best_rank = None
            best_position = -1
            for position in range(len(symbols) - 1):
                rank = self.merge_rank.get((symbols[position], symbols[position + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_position = position
            if best_rank is None:
                break
            symbols[best_position : best_position + 2] = [
                symbols[best_position] + symbols[best_position + 1]
            ]
        return symbols

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        normalize: bool = True,
    ) -> List[int]:
        """文本 -> id 列表。没见过的字符会变成 <unk>（正常语料里应该极少）。"""

        if normalize:
            text = normalize_text(text)
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for word in pre_tokenize(text):
            for piece in self._encode_word(word):
                ids.append(self.token_to_id.get(piece, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: Sequence[str], **kwargs) -> List[List[int]]:
        return [self.encode(text, **kwargs) for text in texts]

    # ------------------------------------------------------------ 解码
    def detokenize(self, pieces: Sequence[str]) -> str:
        """把 token 拼回人类可读的句子。

        步骤：
          1. 遇到带 `</w>` 的 token，说明一个"词"结束了：把攒下来的碎片
             直接连成一个词，词与词之间用一个空格分开；
          2. 中文汉字之间、中文标点前后都不该有空格，再按规则清理一遍。
        """

        words: List[str] = []
        buffer: List[str] = []
        for piece in pieces:
            if piece.endswith(WORD_END):
                buffer.append(piece[: -len(WORD_END)])
                words.append("".join(buffer))
                buffer = []
            else:
                buffer.append(piece)
        if buffer:
            # 模型可能没生成词尾标记（比如被 max_len 截断），兜底也要拼出来
            words.append("".join(buffer))

        text = " ".join(words)
        # 汉字之间不留空格
        text = re.sub(rf"(?<=[{_CJK}]) (?=[{_CJK}])", "", text)
        # 中文标点前面不留空格
        text = re.sub(r"\s+([，。！？；：、）】》」』…—·])", r"\1", text)
        # 中文标点后面也不留空格（逗号、句号都是单独的 token，拼起来会多出一个空格）
        text = re.sub(r"([，。！？；：、」』…—·])\s+", r"\1", text)
        text = re.sub(r"([）】])\s+", r"\1", text)
        text = re.sub(r"([（【《「『])\s+", r"\1", text)
        # 英文标点同理
        text = re.sub(r"\s+([,.!?;:%)\]])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        text = re.sub(r"\s+([’”])", r"\1", text)
        text = re.sub(r"([‘“])\s+", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        pieces: List[str] = []
        for index in ids:
            token = self.id_to_token[index]
            if skip_special_tokens and token in self.special_tokens:
                continue
            pieces.append(token)
        return self.detokenize(pieces)

    def tokenize(self, text: str) -> List[str]:
        """文本 -> token 字符串列表（调试、观察分词结果时很有用）。

        注意返回的是**原始 token**，所以英文词尾会带 `</w>` 标记，
        这正是解码时还原词边界的依据。
        """

        pieces: List[str] = []
        for word in pre_tokenize(normalize_text(text)):
            pieces.extend(self._encode_word(word))
        return pieces

    # ------------------------------------------------------------ 存取
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "special_tokens": self.special_tokens,
            "base_chars": self.base_chars,
            "merges": [[a, b] for a, b in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPE":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(
            merges=[tuple(pair) for pair in payload["merges"]],  # type: ignore[misc]
            base_chars=payload["base_chars"],
            special_tokens=payload.get("special_tokens", SPECIAL_TOKENS),
        )

    # ------------------------------------------------------------ 展示
    def summary(self) -> str:
        return (
            f"BPE 词表：{len(self)} 个 token"
            f"（{len(self.special_tokens)} 特殊符号 + {len(self.base_chars)} 基础字符"
            f" + {len(self.merges)} 条合并规则）"
        )

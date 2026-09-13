"""纯 Python 实现的 BPE（Byte-Pair Encoding）分词器。

为什么机器翻译需要子词分词？
    * 按「词」分词：词表爆炸、遇到没见过的词只能 <unk>；
    * 按「字符」分词：英文序列太长，模型难以建模；
    * BPE 折中：高频词保持整词，低频词拆成子词，永远不需要 <unk>。

本实现遵循 subword-nmt 的经典做法（也是 GPT/RoBERTa 系列 BPE 的雏形）：
    1. 预处理：把句子切成「单元」(unit)。中文按字切，英文按单词切，标点各自成单元。
    2. 每个单元写成字符序列，末尾加 </w> 表示「这是一个单元的结尾」。
       例如 watching -> ['w', 'a', 't', 'c', 'h', 'i', 'n', 'g</w>']
    3. 统计所有相邻符号对的出现次数（按词频加权），每轮合并最高频的一对，
       把合并规则记进 merges 列表。重复 num_merges 次。
    4. 用学到的 merges 对新文本做同样的合并，得到子词序列。

工程注意：训练时不能每轮重扫全部语料，否则 O(合并次数 × 语料长度)。
这里维护了 `stats`（pair -> 加权频次）和 `index`（pair -> 含该 pair 的单元 id），
每次合并只更新受影响的单元，与主流实现一致。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# PAD 必须是 0：PyTorch 的 padding_idx / ignore_index 默认配合 0 使用最自然
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

END_OF_WORD = "</w>"  # 单元结尾标记

# 中文字符范围（CJK 统一表意文字 + 扩展 A + 兼容区 + 中文标点）
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
    (0x3000, 0x303F),
    (0xFF00, 0xFFEF),
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def pre_tokenize(text: str) -> List[str]:
    """把一句话切成「单元」列表。

    中文 -> 逐字；英文/数字 -> 整词；标点与其它符号 -> 各自成单元。

    >>> pre_tokenize("I like 中文, ok!")
    ['I', 'like', '中', '文', ',', 'ok', '!']
    """
    units: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            units.append("".join(buf))
            buf.clear()

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            flush()
            i += 1
        elif is_cjk(ch):
            flush()  # 中文与前后英文天然断开
            units.append(ch)
            i += 1
        else:
            m = _WORD_RE.match(text, i)
            if m:
                flush()
                units.append(m.group(0))
                i = m.end()
            else:
                flush()
                units.append(ch)
                i += 1
    flush()
    return units


def unit_to_symbols(unit: str) -> Tuple[str, ...]:
    """单元 -> 初始符号序列（末尾符号带 </w>）。"""
    if not unit:
        return ()
    symbols = list(unit)
    symbols[-1] = symbols[-1] + END_OF_WORD
    return tuple(symbols)


class BPE:
    """一个可训练、可保存、可复用的 BPE 分词器。"""

    def __init__(self, merges: Sequence[Tuple[str, str]], vocab: Sequence[str]):
        self.merges: List[Tuple[str, str]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.vocab: List[str] = list(vocab)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        # pair -> 合并优先级：数字越小越先合并
        self._merge_rank: Dict[Tuple[str, str], int] = {p: i for i, p in enumerate(self.merges)}

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.vocab)

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

    # ------------------------------------------------------------------ #
    # 编码 / 解码
    # ------------------------------------------------------------------ #
    def encode_tokens(self, text: str) -> List[str]:
        """文本 -> 子词 token 列表（不含特殊符号）。"""
        out: List[str] = []
        for unit in pre_tokenize(text):
            out.extend(self._encode_unit(unit_to_symbols(unit)))
        return out

    def _encode_unit(self, symbols: Tuple[str, ...]) -> List[str]:
        """对一个单元反复应用 merges（每次合并当前优先级最高的一对）。"""
        symbols = list(symbols)
        while len(symbols) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(symbols, symbols[1:]):
                rank = self._merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            symbols = _merge_pair(symbols, best_pair)
        return symbols

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """文本 -> id 列表。"""
        ids = [self.token_to_id.get(t, self.unk_id) for t in self.encode_tokens(text)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """id 列表 -> 文本。"""
        specials = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
        tokens: List[str] = []
        for i in ids:
            i = int(i)
            if skip_special and i in specials:
                continue
            if i >= len(self.vocab):
                continue
            tokens.append(self.vocab[i])
        return self.detokenize(tokens)

    @staticmethod
    def detokenize(tokens: Sequence[str]) -> str:
        """把子词拼回可读文本。

        </w> 标记一个单元的结束：遇到它就把缓冲区的子词连成一个词，
        单元之间用空格分隔。最后再去掉「汉字之间的空格」。
        """
        units: List[str] = []
        buf = ""
        for tok in tokens:
            if tok.endswith(END_OF_WORD):
                buf += tok[: -len(END_OF_WORD)]
                units.append(buf)
                buf = ""
            else:
                buf += tok
        if buf:
            units.append(buf)
        text = " ".join(units)
        # 去掉汉字/中文标点之间的空格："我 爱 你" -> "我爱你"
        text = re.sub(r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]) +(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])", "", text)
        return text.strip()

    def is_english_like(self, token: str) -> bool:
        return bool(token) and all(ord(c) < 128 for c in token.replace(END_OF_WORD, ""))

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        vocab_size: int = 8000,
        min_pair_freq: int = 2,
        verbose: bool = True,
    ) -> "BPE":
        """在平行语料的文本上学习 merges，并构建词表。

        参数
        ----
        texts         : 所有句子（源语言 + 目标语言放在一起训练「联合词表」）
        vocab_size    : 目标词表大小（含 4 个特殊符号）
        min_pair_freq : 低于该频次的符号对不再合并
        """
        # ---- 1. 统计单元频次 ---------------------------------------- #
        unit_freq: Counter = Counter()
        for text in texts:
            unit_freq.update(pre_tokenize(text))

        words: List[List[str]] = []
        freqs: List[int] = []
        for unit, f in unit_freq.items():
            words.append(list(unit_to_symbols(unit)))
            freqs.append(f)
        # 必须在合并**之前**记录基础字符表：合并会把 'h','e','l','l','o</w>'
        # 逐步变成 'hello</w>'，之后再看 words 就只剩整词了。
        # 基础字符表是"没见过的词还能逐字符回退"的兜底保障。
        alphabet = {s for symbols in words for s in symbols}
        if verbose:
            print(f"[BPE] 唯一单元数 {len(words)}，总单元数 {sum(freqs)}")

        # ---- 2. 初始化 pair 统计（含倒排索引，避免每轮重扫语料）------ #
        stats: Counter = Counter()
        index: Dict[Tuple[str, str], set] = defaultdict(set)
        for wid, symbols in enumerate(words):
            for pair in _pairs(symbols):
                stats[pair] += freqs[wid]
                index[pair].add(wid)

        # 词表 = 特殊符号 + 基础字符表 + 每条合并规则的产物。
        # 因此合并次数的上限要先把"特殊符号 + 基础字符表"扣掉，
        # 否则可能出现：编码时用到了某条合并结果，但该结果被截断出词表 -> 变成 <unk>。
        num_merges = max(0, vocab_size - len(SPECIAL_TOKENS) - len(alphabet))

        merges: List[Tuple[str, str]] = []
        for step in range(num_merges):
            if not stats:
                break
            # 取「频次最高」的 pair；并列时用字典序保证结果可复现
            pair = max(stats.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if stats[pair] < min_pair_freq:
                break
            merges.append(pair)

            for wid in list(index.get(pair, ())):
                symbols = words[wid]
                f = freqs[wid]
                for p in _pairs(symbols):  # 先撤销旧统计
                    stats[p] -= f
                    if stats[p] <= 0:
                        del stats[p]
                    index[p].discard(wid)
                symbols = _merge_pair(symbols, pair)
                words[wid] = symbols
                for p in _pairs(symbols):  # 再写入新统计
                    stats[p] += f
                    index[p].add(wid)
            index.pop(pair, None)
            stats.pop(pair, None)

            if verbose and (step + 1) % 500 == 0:
                nxt = max(stats.items(), key=lambda kv: kv[1]) if stats else ("-", 0)
                print(f"[BPE] 已学习 {step + 1} 条合并规则，最近一次合并 {pair}，下一候选 {nxt[0]}（频次 {nxt[1]}）")

        if verbose:
            print(f"[BPE] 完成：{len(merges)} 条合并规则")

        # ---- 3. 构建词表 ------------------------------------------------ #
        # 顺序：特殊符号 -> 基础字符（按在语料中的出现频次降序）-> 合并产物（按合并顺序）
        alphabet_freq = Counter()
        for unit, f in unit_freq.items():
            for s in unit_to_symbols(unit):
                alphabet_freq[s] += f
        vocab = list(SPECIAL_TOKENS)
        vocab += [s for s, _ in alphabet_freq.most_common()]
        seen = set(vocab)
        for a, b in merges:
            token = a + b
            if token not in seen:
                seen.add(token)
                vocab.append(token)

        if verbose:
            print(f"[BPE] 词表大小 {len(vocab)}（目标 {vocab_size}）："
                  f"基础字符 {len(alphabet_freq)} + 合并产物 {len(vocab) - len(alphabet_freq) - len(SPECIAL_TOKENS)}")
        bpe = cls(merges, vocab)
        # 自检：训练语料重新编码后不应出现 <unk>
        unk = sum(1 for t in texts[:2000] for i in bpe.encode(t) if i == UNK_ID)
        if unk:
            print(f"[BPE][warning] 训练语料自身编码出现 {unk} 个 <unk>，请检查词表构建逻辑")
        return bpe

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "special_tokens": SPECIAL_TOKENS,
                    "merges": [list(m) for m in self.merges],
                    "vocab": self.vocab,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPE":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls([tuple(m) for m in data["merges"]], data["vocab"])


# ---------------------------------------------------------------------- #
# 工具函数
# ---------------------------------------------------------------------- #
def _pairs(symbols: Sequence[str]) -> List[Tuple[str, str]]:
    return list(zip(symbols, symbols[1:]))


def _merge_pair(symbols: Sequence[str], pair: Tuple[str, str]) -> List[str]:
    """把 symbols 中所有 pair 合并成一个符号（贪心、不重叠）。"""
    a, b = pair
    out: List[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_on_files(
    paths: Sequence[str | Path],
    vocab_size: int = 8000,
    min_pair_freq: int = 2,
    verbose: bool = True,
) -> BPE:
    """便捷函数：直接读取若干 TSV 文件的全部列来训练 BPE。"""
    texts: List[str] = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            texts.extend(field for field in line.split("\t") if field.strip())
    return BPE.train(texts, vocab_size=vocab_size, min_pair_freq=min_pair_freq, verbose=verbose)

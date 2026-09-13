"""真实平行语料的下载、清洗、切分和编码。

本项目用两份公开语料（英译德）：

    News-Commentary v16 (de-en)  ~29.4 万句对  训练集，新闻评论域
    WMT-News v2019 (de-en)       ~4.6 万句对    WMT 官方历年测试集

具体切成四份：

    dev      newstest2013          3000 句    WMT14 英德任务的标准验证集
    test2014 newstest2014          3003 句    论文里报 27.3 BLEU 用的就是它
    test2017 newstest2017-ende     3004 句
    test2018 newstest2018-ende     2998 句
    test2019 newstest2019-ende     1997 句

为什么测试集必须是官方的？

因为 BLEU 只在同一份测试集、同一套切词方式下才有可比性。WMT 的 newstest2014
是《Attention Is All You Need》报分数用的那份，也是后来几乎所有 NMT 论文的标配，
你自己的模型跑出来的数字可以直接和它们摆在一起看。自己随手切一个测试集，
分数再高也不知道算高还是低。

真实数据是脏的，这个文件里有相当一部分代码在处理这些事：

    * 空行（原文件里段落之间有空行）
    * 编码坏行（UTF-8 解码失败留下的 U+FFFD）
    * 语言放错（德语那一列其实是英语，或者混进了别的语言）
    * 长度比例失衡（一句英文对着一大段德语，通常是文档级对齐的错位）
    * 重复句对

每一步过滤都会记数并打印出来 —— 数据清洗最忌讳"悄悄扔掉了几十万句"。
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

import torch

from .bpe import BPE, contains_cjk, normalize_text
from .utils import Timer, ensure_dir, get_logger, human_time, save_json, set_seed

logger = get_logger()

# 这一份代码只做英译德。把语言对写进 meta.json，
# 训练脚本启动时会校验 —— 否则你在服务器上很可能拿着上一次英译中留下的
# data/ready 直接开训，发现得晚了就白跑几小时。
DATA_PAIR = "en-de"


# --------------------------------------------------------------------------
# 语料清单
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusSpec:
    """一份 OPUS 语料的元信息。"""

    name: str
    url: str
    domain: str
    note: str = ""
    default_max_pairs: int = 0  # 0 = 全部使用


# 训练语料：新闻评论域，英德约 29.4 万句对（是常见语言对里质量最好的中等规模语料之一）
# 注意名字里带上 de-en：缓存目录按语料名区分，换语言对时不会误用上一次的缓存
NEWS_COMMENTARY = CorpusSpec(
    name="News-Commentary.de-en",
    url="https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/de-en.txt.zip",
    domain="新闻评论",
    note="WMT 官方指定的训练语料之一，294,498 句对",
)

# 验证 / 测试语料：WMT 历年 newstest（2008~2019 全都在这一个包里）
WMT_NEWS = CorpusSpec(
    name="WMT-News.de-en",
    url="https://object.pouta.csc.fi/OPUS-WMT-News/v2019/moses/de-en.txt.zip",
    domain="新闻",
    note="newstest2013（dev）+ newstest2014/2017/2018/2019（测试），含官方参考译文",
)

# 可选加餐：换领域做数据规模实验
EXTRA_CORPORA: Dict[str, CorpusSpec] = {
    "europarl": CorpusSpec(
        name="Europarl.de-en",
        url="https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/de-en.txt.zip",
        domain="欧洲议会辩论",
        note="近 200 万句对，是英德方向上最大的干净语料；语气正式、句式长",
        default_max_pairs=300000,
    ),
    "ted2013": CorpusSpec(
        name="TED2013.de-en",
        url="https://object.pouta.csc.fi/OPUS-TED2013/v1.1/moses/de-en.txt.zip",
        domain="演讲口语",
        note="14 万句对，句子短、口语化，用来观察换领域后 BLEU 怎么变",
        default_max_pairs=140000,
    ),
    "multiun": CorpusSpec(
        name="MultiUN.de-en",
        url="https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/de-en.txt.zip",
        domain="联合国文件",
        note="16 万句对，正式书面语，术语密集",
        default_max_pairs=160000,
    ),
}

# WMT-News 里官方数据集的文档名 -> 我们给它的 split 名
# 注意 OPUS 的命名规则：后缀是"对齐时的源语言"。同一个年份的 -deen 和 -ende
# 是两套不同的句对，这里统一取 -ende（英译德方向），2013/2014 只有一套就直接用。
WMT_SPLITS = {
    "newstest2013": "dev",
    "newstest2014-deen": "test2014",
    "newstest2017-ende": "test2017",
    "newstest2018-ende": "test2018",
    "newstest2019-ende": "test2019",
}


# --------------------------------------------------------------------------
# 下载 / 解压
# --------------------------------------------------------------------------
def download(url: str, dest: Path, force: bool = False) -> Path:
    """下载到 dest。已存在就跳过。"""

    dest = Path(dest)
    if dest.exists() and not force and dest.stat().st_size > 0:
        logger.info(f"已存在，跳过下载：{dest.name}（{dest.stat().st_size / 1024 ** 2:.1f} MB）")
        return dest

    ensure_dir(dest.parent)
    logger.info(f"下载 {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(tmp, "wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(
                        f"\r   {percent:5.1f}%  {downloaded / 1024 ** 2:6.1f}/{total / 1024 ** 2:.1f} MB",
                        end="",
                        flush=True,
                    )
    print()
    tmp.replace(dest)
    return dest


def extract(zip_path: Path, dest_dir: Path, force: bool = False) -> Path:
    """解压 OPUS 的 moses 格式包。

    解压出来通常有三样东西：
        *.en / *.de   每行一句的平行文本
        *.xml         句子级对齐表（能看出哪些句子属于哪个文档）
        LICENSE/README
    """

    dest_dir = Path(dest_dir)
    marker = dest_dir / ".extracted"
    if marker.exists() and not force:
        return dest_dir
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    ensure_dir(dest_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)
    marker.write_text("ok", encoding="utf-8")
    logger.info(f"已解压到 {dest_dir}")
    return dest_dir


def fetch_corpus(spec: CorpusSpec, raw_dir: Path, force: bool = False) -> Path:
    """下载 + 解压，返回解压目录。"""

    zip_path = raw_dir / f"{spec.name}.zip"
    download(spec.url, zip_path, force=force)
    return extract(zip_path, raw_dir / spec.name, force=force)


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def read_moses_pairs(directory: Path) -> List[Tuple[str, str]]:
    """读 moses 格式的 .en / .de 两个文件，按行配成句对。

    返回 (英文, 德文) —— 也就是英译德方向。
    注意 OPUS 的包名是 de-en，但文件是分开的，方向由我们读的顺序决定。
    """

    directory = Path(directory)
    en_files = list(directory.glob("*.en"))
    de_files = list(directory.glob("*.de"))
    if not en_files or not de_files:
        raise FileNotFoundError(f"{directory} 里没有找到 *.en / *.de 文件")

    # errors="replace" 会把坏字节换成 U+FFFD 而不是直接抛异常，
    # 这样我们能"看得见"坏行，并在清洗阶段把它们挑出来。
    def read_lines(path: Path) -> List[str]:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()

    source = read_lines(en_files[0])
    target = read_lines(de_files[0])
    if len(source) != len(target):
        logger.warning(
            f"两侧行数不一致（英文 {len(source)} 行、德文 {len(target)} 行），"
            "按「句子内部换行」修复后再对齐"
        )
    return align_parallel_lines(source, target)


# 单条对齐代价的上限：避免个别长度极端的句子 dominate 整个最优路径
_MAX_PAIR_COST = 3.0


def _pair_cost(short_record: str, long_record: str) -> float:
    """一对句子的代价：长度比例偏离 1 的对数幅度（用 log 是为了左右对称）。"""

    if not short_record or not long_record:
        return _MAX_PAIR_COST
    return min(_MAX_PAIR_COST, abs(math.log(len(long_record) / len(short_record))))


def align_parallel_lines(
    source: Sequence[str],
    target: Sequence[str],
    max_extra: int = 8,
) -> List[Tuple[str, str]]:
    """把两侧的 moses 行对齐成句对，顺手修掉「句子内部换行」。

    为什么需要这一步？moses 格式规定"一行一句"，但真实文件偶尔犯规：
    本项目用的 de-en 包里就有 5 处德语句子被换行拆成两行，
    于是 de 文件比 en 文件多 5 行，直接 zip() 的话从错位点开始全乱。

    做法是一次**单调对齐**（和编辑距离同一类思路）：

        状态：把短侧的 i 条记录配完、并已经多用掉 k 行长侧
        转移：① 一对一      短[i] 配 长[i+k]
             ② 一对二合并  短[i] 配 长[i+k] + " " + 长[i+k+1]   （k += 1）
        代价：|log(长侧长度 / 短侧长度)|，越接近 1 越好
        终点：必须恰好把 k 用到 max_extra（也就是把所有多余行都合并掉）

    因为多出来的行数很少（这里是 5），k 的取值只有 0~5 这几种，
    状态数 = 句子数 × 6，几秒钟就能算完 4.6 万句。

    为什么不用"这一行没有句末标点就合并"这种启发式？
    因为德语新闻里有大量没有标点的标题行，误合并会让**后面所有句子**都错位。
    DP 是在全局找一个总代价最小的方案，稳得多。
    """

    if len(source) == len(target):
        return list(zip(source, target))

    if len(source) > len(target):
        long_side, short_side, long_is_source = list(source), list(target), True
    else:
        long_side, short_side, long_is_source = list(target), list(source), False

    extra_total = len(long_side) - len(short_side)
    if extra_total > max_extra:
        raise ValueError(
            f"两侧行数相差 {extra_total} 行（英文 {len(source)}、德文 {len(target)}），"
            f"超过容忍上限 {max_extra}。这不像是「句子内部换行」，请检查语料文件是否完整。"
        )

    n_short, n_long = len(short_side), len(long_side)
    inf = float("inf")
    # dp[i][k]：配完前 i 条短侧记录、已多用 k 行长侧的最小代价
    dp = [[inf] * (extra_total + 1) for _ in range(n_short + 1)]
    choice: List[List[Optional[bool]]] = [[None] * (extra_total + 1) for _ in range(n_short + 1)]
    dp[0][0] = 0.0

    for i in range(n_short):
        for k in range(extra_total + 1):
            base = dp[i][k]
            if base == inf:
                continue
            j = i + k                                  # 长侧下标
            if j >= n_long:
                continue
            # ① 一对一
            cost = base + _pair_cost(short_side[i], long_side[j])
            if cost < dp[i + 1][k]:
                dp[i + 1][k] = cost
                choice[i + 1][k] = False
            # ② 把长侧相邻两行合并成一句
            if k < extra_total and j + 1 < n_long:
                merged = long_side[j] + " " + long_side[j + 1]
                cost = base + _pair_cost(short_side[i], merged)
                if cost < dp[i + 1][k + 1]:
                    dp[i + 1][k + 1] = cost
                    choice[i + 1][k + 1] = True

    if dp[n_short][extra_total] == inf:
        raise ValueError("对齐失败：找不到合法的单调对齐方案")

    # 回溯，还原每一步是"一对一"还是"合并"
    merges: List[int] = []
    k = extra_total
    for i in range(n_short, 0, -1):
        if choice[i][k]:
            merges.append(i - 1)                        # 第 i-1 条短记录对应一次合并
            k -= 1
    merges.reverse()

    pairs: List[Tuple[str, str]] = []
    short_index = long_index = 0
    merge_at = set(merges)
    while short_index < n_short:
        record = long_side[long_index]
        if short_index in merge_at:
            record = record + " " + long_side[long_index + 1]
            long_index += 1
        counterpart = short_side[short_index]
        pairs.append((record, counterpart) if long_is_source else (counterpart, record))
        short_index += 1
        long_index += 1

    if merges:
        logger.info(
            f"修复了 {len(merges)} 处「句子内部换行」（"
            f"{'英文' if long_is_source else '德文'}侧，位置约 {merges[:8]}）"
        )
    return pairs


def read_wmt_news_splits(directory: Path) -> Dict[str, List[Tuple[str, str]]]:
    """从 WMT-News 里切出官方的 dev / test 集。

    做法：xml 里的每个 <linkGrp> 对应一个文档（newstest2014-deen 之类），
    它们的先后顺序和 .en/.de 文件的行顺序一致。按 linkGrp 的条数累加偏移量，
    就能把整块行区间还原成一个个测试集。

    这个包里同时有 -deen 和 -ende 两种文档：它们是**两套不同的句对**
    （源语不同、译文不同），混在一起会把测试集撑大。这里只取 WMT_SPLITS 里列出的那几个。
    """

    directory = Path(directory)
    xml_files = list(directory.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"{directory} 里没有找到 *.xml")

    pairs = read_moses_pairs(directory)
    tree = ElementTree.parse(xml_files[0])

    splits: Dict[str, List[Tuple[str, str]]] = {}
    offset = 0
    for group in tree.getroot():
        links = list(group)
        doc_name = (group.attrib.get("toDoc") or group.attrib.get("fromDoc") or "").split("/")[-1]
        doc_name = doc_name.replace(".xml.gz", "")

        if doc_name in WMT_SPLITS:
            splits[WMT_SPLITS[doc_name]] = pairs[offset : offset + len(links)]
        offset += len(links)

    if offset != len(pairs):
        raise ValueError(f"xml 的 link 总数 {offset} 与文件行数 {len(pairs)} 不一致，切分会错位")

    missing = set(WMT_SPLITS.values()) - set(splits)
    if missing:
        raise ValueError(f"没能切出这些 split：{sorted(missing)}")
    return splits


# --------------------------------------------------------------------------
# 清洗
# --------------------------------------------------------------------------
# 判断"语言放错"用的功能词表。
# 英德同属拉丁字母，光看字符集分不出谁是谁，只能靠最高频的功能词 ——
# 这也是真实数据清洗里常用的土办法（正式项目会用 fastText 之类的语言识别模型）。
_ENGLISH_MARKERS = {
    "the", "and", "of", "to", "in", "is", "was", "were", "for", "that", "with",
    "on", "as", "by", "at", "from", "it", "be", "has", "have", "this", "not",
}
_GERMAN_MARKERS = {
    "der", "die", "das", "und", "ist", "sind", "nicht", "mit", "sich", "auf",
    "für", "von", "den", "dem", "ein", "eine", "zu", "im", "am", "des", "wird",
    "auch", "als", "dass", "hat", "haben",
}

_LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏ]")
_WORD_SPLIT = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class CleanConfig:
    """清洗阈值。改这些数字前先看一眼被丢掉的原因分布。"""

    max_chars: int = 400              # 单侧字符数上限（编码后还会再按 token 数过滤）
    min_chars: int = 1
    min_latin_ratio: float = 0.50     # 两侧都必须以拉丁字母为主，能筛掉混进来的俄语/希腊语
    min_length_ratio: float = 0.50    # 德文字符数 / 英文字符数
    max_length_ratio: float = 2.50    # 上限
    check_language: bool = True       # 用功能词粗判"哪一侧其实是另一种语言"


@dataclass
class CleanStats:
    """记录每一步丢了多少句，以及为什么。"""

    seen: int = 0
    kept: int = 0
    reasons: Counter = field(default_factory=Counter)

    def drop(self, reason: str) -> None:
        self.reasons[reason] += 1

    def report(self) -> str:
        if self.seen == 0:
            return "（没有数据）"
        lines = [f"输入 {self.seen} 句，保留 {self.kept} 句（{self.kept / self.seen:.1%}）"]
        for reason, count in self.reasons.most_common():
            lines.append(f"    丢掉 {count:>7} 句：{reason}")
        return "\n".join(lines)


def _latin_ratio(text: str) -> float:
    """去掉空白后，拉丁字母占的比例。"""

    stripped = "".join(text.split())
    if not stripped:
        return 0.0
    return len(_LATIN.findall(stripped)) / len(stripped)


def _words(text: str) -> List[str]:
    return _WORD_SPLIT.findall(text.lower())


def _looks_english(text: str) -> bool:
    return any(word in _ENGLISH_MARKERS for word in _words(text))


def _looks_german(text: str) -> bool:
    return any(word in _GERMAN_MARKERS for word in _words(text))


def clean_pair(
    source: str,
    target: str,
    stats: CleanStats,
    config: CleanConfig,
) -> Optional[Tuple[str, str]]:
    """清洗一句句对。返回 None 表示丢弃，并会在 stats 里记下原因。

    source 是英文，target 是德文。
    """

    stats.seen += 1
    source = normalize_text(source)
    target = normalize_text(target)

    if "\ufffd" in source or "\ufffd" in target:
        stats.drop("编码坏行（含替换字符 U+FFFD）")
        return None
    if contains_cjk(source) or contains_cjk(target):
        # 英德语料里出现汉字，基本只有一种可能：抓取串行或者编码坏了
        stats.drop("混入汉字/日文（疑似抓取或编码问题）")
        return None
    if not source or not target:
        stats.drop("空行")
        return None
    if len(source) > config.max_chars or len(target) > config.max_chars:
        stats.drop(f"单侧超过 {config.max_chars} 字符")
        return None
    if len(source) < config.min_chars or len(target) < config.min_chars:
        stats.drop("太短")
        return None

    if _latin_ratio(source) < config.min_latin_ratio:
        stats.drop("英文侧拉丁字母比例过低（可能混了别的语言）")
        return None
    if _latin_ratio(target) < config.min_latin_ratio:
        stats.drop("德文侧拉丁字母比例过低（可能混了别的语言）")
        return None

    # 功能词判语言：只在两边都够长、而且证据明确时才丢。
    # 为什么要加"两边都够长"？因为新闻标题常常一个功能词都没有
    # （"New Questions Over California Water Project"），不能因此误杀。
    if config.check_language:
        if len(_words(source)) >= 6 and len(_words(target)) >= 6:
            if _looks_english(target) and not _looks_german(target):
                stats.drop("德文侧其实是英文（对齐错位）")
                return None
            if _looks_german(source) and not _looks_english(source):
                stats.drop("英文侧其实是德文（对齐错位）")
                return None

    # 长度比例：德语通常比英语长 0~20%（复合词和格变化会多出一些字母）。
    # 偏离太远通常意味着对齐错位（一句英文对着一整段德语）。
    length_ratio = len(target) / max(1, len(source))
    if not config.min_length_ratio <= length_ratio <= config.max_length_ratio:
        stats.drop("德英长度比例失衡（疑似对齐错位）")
        return None

    stats.kept += 1
    return source, target


def clean_pairs(
    pairs: Iterable[Tuple[str, str]],
    config: CleanConfig,
) -> Tuple[List[Tuple[str, str]], CleanStats]:
    stats = CleanStats()
    kept: List[Tuple[str, str]] = []
    for source, target in pairs:
        result = clean_pair(source, target, stats, config)
        if result is not None:
            kept.append(result)
    return kept, stats


def deduplicate(pairs: Sequence[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], int]:
    """去掉完全重复的句对（新闻语料里同一篇稿子被多处收录很常见）。"""

    seen = set()
    unique: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)
    return unique, len(pairs) - len(unique)


# --------------------------------------------------------------------------
# 编码落盘
# --------------------------------------------------------------------------
def encode_split(
    pairs: Sequence[Tuple[str, str]],
    tokenizer: BPE,
    max_src_len: int,
    max_tgt_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], List[Tuple[str, str]]]:
    """把句对编码成"拍平的 int32 张量 + 长度数组"。

    为什么不用 Python 列表存 12 万句？
    每个 int 在 Python 里是 28 字节的对象，加上列表指针，光内存就要上百 MB，
    而且每次取样本都要重新分配。拍平成一根长张量后，取第 i 句就是
    flat[offset[i] : offset[i] + length[i]]，可以整块放进内存、按需切片。

    约定：
        src: 英文 id + <eos>，不加 <bos>
        tgt: <bos> + 德文 id + <eos>
    训练时把 tgt 错开一位：输入 tgt[:, :-1]，标签 tgt[:, 1:]。

    返回值里最后一项是"过滤后保留下来的句对原文"：
    BLEU 必须在原始文本上算，不能拿解码回来的字符串去比，
    否则分词器的空格规则会污染指标。
    """

    src_sequences: List[List[int]] = []
    tgt_sequences: List[List[int]] = []
    kept_pairs: List[Tuple[str, str]] = []
    for source, target in pairs:
        src_ids = tokenizer.encode(source, add_eos=True)
        tgt_ids = tokenizer.encode(target, add_bos=True, add_eos=True)
        if len(src_ids) > max_src_len or len(tgt_ids) > max_tgt_len:
            continue
        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)
        kept_pairs.append((source, target))

    if not src_sequences:
        raise ValueError("过滤后一句都没剩下，检查一下 max_src_len / max_tgt_len 是不是太小")

    def flatten(sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.int32)
        flat = torch.tensor([token for seq in sequences for token in seq], dtype=torch.int32)
        return flat, lengths

    src_flat, src_lengths = flatten(src_sequences)
    tgt_flat, tgt_lengths = flatten(tgt_sequences)

    stats = {
        "pairs": len(src_sequences),
        "src_tokens": int(src_lengths.sum()),
        "tgt_tokens": int(tgt_lengths.sum()),
        "src_len_mean": float(src_lengths.float().mean()),
        "tgt_len_mean": float(tgt_lengths.float().mean()),
        "src_len_p95": float(src_lengths.float().quantile(0.95)),
        "tgt_len_p95": float(tgt_lengths.float().quantile(0.95)),
    }
    return src_flat, src_lengths, tgt_flat, tgt_lengths, stats, kept_pairs


def save_split(
    path: Path,
    src_flat: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_flat: torch.Tensor,
    tgt_lengths: torch.Tensor,
    kept_pairs: Sequence[Tuple[str, str]] = (),
) -> None:
    # 只存张量：torch 2.6 起 torch.load 默认 weights_only=True，
    # 纯张量字典能保证任何版本都能安全读回来。
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "src": src_flat,
            "src_lengths": src_lengths,
            "tgt": tgt_flat,
            "tgt_lengths": tgt_lengths,
        },
        path,
    )
    # 同时存一份纯文本：评测 BLEU 时用得上，也能直接用眼睛扫一遍数据
    if kept_pairs:
        path = Path(path)
        (path.parent / f"{path.stem}.src.en").write_text(
            "\n".join(source for source, _ in kept_pairs) + "\n", encoding="utf-8"
        )
        (path.parent / f"{path.stem}.ref.de").write_text(
            "\n".join(target for _, target in kept_pairs) + "\n", encoding="utf-8"
        )


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build_dataset(
    data_dir: str | Path = "data",
    out_dir: str | Path = "data/ready",
    vocab_size: int = 32000,
    max_src_len: int = 192,
    max_tgt_len: int = 192,
    extras: Sequence[str] = (),
    extra_pairs: int = 0,
    max_train_pairs: int = 0,
    bpe_train_pairs: int = 100000,
    skip_download: bool = False,
    seed: int = 2024,
) -> Dict[str, object]:
    """完整流程：下载 -> 清洗 -> 训练 BPE -> 编码 -> 落盘。"""

    set_seed(seed)
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    raw_dir = ensure_dir(data_dir / "raw")

    clean_config = CleanConfig()
    meta: Dict[str, object] = {
        "pair": DATA_PAIR,
        "format_version": 2,
        "vocab_size": vocab_size,
        "sources": {},
    }

    # --- 1. 训练语料 ---
    logger.info("=" * 68)
    logger.info("第 1 步：准备训练语料")
    with Timer() as timer:
        directory = (
            fetch_corpus(NEWS_COMMENTARY, raw_dir) if not skip_download else raw_dir / NEWS_COMMENTARY.name
        )
        raw_train = read_moses_pairs(directory)
        logger.info(f"读入 {len(raw_train)} 句（{timer.elapsed:.1f}s）")

    train_pairs, train_stats = clean_pairs(raw_train, clean_config)
    logger.info(f"News-Commentary 清洗结果：\n{train_stats.report()}")
    meta["sources"]["news-commentary"] = {"raw": len(raw_train), "kept": len(train_pairs)}

    # --- 2. 可选的加餐语料 ---
    for name in extras:
        if name not in EXTRA_CORPORA:
            raise ValueError(f"未知的加餐语料 {name!r}，可选：{', '.join(EXTRA_CORPORA)}")
        spec = EXTRA_CORPORA[name]
        logger.info(f"加入额外语料：{spec.name}（{spec.domain}）")
        directory = fetch_corpus(spec, raw_dir) if not skip_download else raw_dir / spec.name
        extra_raw = read_moses_pairs(directory)
        limit = extra_pairs if extra_pairs > 0 else spec.default_max_pairs
        if limit > 0:
            extra_raw = extra_raw[:limit]
        extra_clean, extra_stats = clean_pairs(extra_raw, clean_config)
        logger.info(f"{spec.name} 清洗结果：\n{extra_stats.report()}")
        train_pairs.extend(extra_clean)
        meta["sources"][name] = {
            "raw": len(extra_raw),
            "kept": len(extra_clean),
            "domain": spec.domain,
        }

    train_pairs, dropped = deduplicate(train_pairs)
    logger.info(f"去重丢掉 {dropped} 句，训练集共 {len(train_pairs)} 句")

    if max_train_pairs > 0 and len(train_pairs) > max_train_pairs:
        train_pairs = train_pairs[:max_train_pairs]
        logger.info(f"按 --max-train-pairs 截断到 {len(train_pairs)} 句")
    meta["train_pairs"] = len(train_pairs)

    # --- 3. 验证 / 测试语料（WMT 官方） ---
    logger.info("=" * 68)
    logger.info("第 2 步：准备 WMT 官方验证 / 测试集")
    wmt_dir = fetch_corpus(WMT_NEWS, raw_dir) if not skip_download else raw_dir / WMT_NEWS.name
    wmt_splits = read_wmt_news_splits(wmt_dir)

    eval_splits: Dict[str, List[Tuple[str, str]]] = {}
    eval_stats: Dict[str, Dict[str, int]] = {}
    for split_name, pairs in sorted(wmt_splits.items()):
        cleaned, stats = clean_pairs(pairs, clean_config)
        eval_splits[split_name] = cleaned
        eval_stats[split_name] = {"raw": stats.seen, "kept": stats.kept}
        logger.info(f"{split_name}: {stats.kept}/{stats.seen} 句保留")
    meta["splits"] = eval_stats

    # --- 4. 训练 BPE ---
    logger.info("=" * 68)
    logger.info(f"第 3 步：训练 BPE 分词器（目标词表 {vocab_size}）")
    bpe_texts: List[str] = []
    # 只拿前 N 句对训分词器：合并规则学的是"词怎么拼"，10 万句对已经足够，
    # 再往上加只会让这一步更慢（每多一个词，所有涉及它的合并都要多算一次）。
    for source, target in train_pairs[:bpe_train_pairs]:
        bpe_texts.append(source)
        bpe_texts.append(target)

    # BPE 训练是数据准备里最慢的一步，所以给它配一个带预计剩余时间的进度条。
    # 速率按"已完成合并数 / 已用秒数"实时估算，前几十秒的估算会偏乐观，属正常现象。
    merge_start = time.perf_counter()

    def _merge_progress(done: int, total: int) -> None:
        elapsed = time.perf_counter() - merge_start
        rate = done / max(1e-9, elapsed)
        remaining = (total - done) / max(1e-9, rate)
        logger.info(
            f"    BPE 合并 {done}/{total}（已用 {human_time(elapsed)}，"
            f"预计还需 {human_time(remaining)}）"
        )

    with Timer() as timer:
        tokenizer = BPE.train(
            bpe_texts,
            vocab_size=vocab_size,
            min_frequency=2,
            progress=_merge_progress,
        )
    logger.info(f"{tokenizer.summary()}（耗时 {timer.elapsed:.1f}s）")
    tokenizer.save(out_dir / "vocab.json")

    # --- 5. 编码落盘 ---
    logger.info("=" * 68)
    logger.info("第 4 步：编码成 id 并落盘")
    split_stats: Dict[str, Dict[str, float]] = {}
    for name, pairs in [("train", train_pairs)] + sorted(eval_splits.items()):
        src_flat, src_lengths, tgt_flat, tgt_lengths, stats, kept_pairs = encode_split(
            pairs, tokenizer, max_src_len, max_tgt_len
        )
        save_split(out_dir / f"{name}.pt", src_flat, src_lengths, tgt_flat, tgt_lengths, kept_pairs)
        split_stats[name] = stats
        logger.info(
            f"{name:<9} {stats['pairs']:>7} 句  英文 {stats['src_tokens']:>10} token  "
            f"德文 {stats['tgt_tokens']:>9} token  "
            f"（英文均长 {stats['src_len_mean']:.1f}，p95 {stats['src_len_p95']:.0f}）"
        )

    meta["split_stats"] = split_stats
    meta["clean_config"] = dict(clean_config.__dict__)
    meta["bpe_merges"] = len(tokenizer.merges)
    meta["bpe_base_chars"] = len(tokenizer.base_chars)
    meta["special_tokens"] = list(tokenizer.special_tokens)
    meta["vocab_size_actual"] = len(tokenizer)
    meta["max_src_len"] = max_src_len
    meta["max_tgt_len"] = max_tgt_len
    meta["extras"] = list(extras)
    save_json(out_dir / "meta.json", meta)

    logger.info("=" * 68)
    logger.info(f"完成，产物在 {out_dir}")
    logger.info("  vocab.json   分词器")
    logger.info(f"  train.pt     训练集（{split_stats['train']['pairs']} 句）")
    for name in sorted(eval_splits):
        logger.info(f"  {name}.pt     {split_stats[name]['pairs']} 句")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并准备英译中新闻平行语料")
    parser.add_argument("--data-dir", default="data", help="原始语料存放目录")
    parser.add_argument("--out", default="data/ready", help="处理后产物目录")
    parser.add_argument(
        "--vocab-size", type=int, default=32000,
        help="联合 BPE 词表大小。英德这种形态丰富的语言对用 32k 比较标准；"
             "想快一倍可以降到 16000（BPE 训练时间大致与合并次数成正比）",
    )
    parser.add_argument("--max-src-len", type=int, default=192, help="英文侧最大 token 数")
    parser.add_argument("--max-tgt-len", type=int, default=192, help="德文侧最大 token 数")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        choices=sorted(EXTRA_CORPORA),
        help="额外加入的语料（可重复），用来做数据规模/领域实验",
    )
    parser.add_argument("--extra-pairs", type=int, default=0, help="额外语料最多取多少句，0=用默认值")
    parser.add_argument("--max-train-pairs", type=int, default=0, help="训练集上限，0=不限制")
    parser.add_argument(
        "--bpe-train-pairs", type=int, default=100000,
        help="训练 BPE 时最多用多少句对（默认 10 万，约等于 20 万行文本）。"
             "词表 32k 时这一步大约要 1 小时，想快点可以配合 --vocab-size 16000",
    )
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，直接用已解压的目录")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    build_dataset(
        data_dir=args.data_dir,
        out_dir=args.out,
        vocab_size=args.vocab_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        extras=args.extra,
        extra_pairs=args.extra_pairs,
        max_train_pairs=args.max_train_pairs,
        bpe_train_pairs=args.bpe_train_pairs,
        skip_download=args.skip_download,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

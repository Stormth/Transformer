"""真实平行语料的下载、清洗、切分和编码。

本项目用两份公开语料：

    News-Commentary v16 (en-zh)  ~12.6 万句对   训练集。新闻评论域，也就是"新闻联播"体
    WMT-News v2019 (en-zh)       ~2 万句对      WMT 官方 newsdev/newstest，做验证与测试

为什么测试集要用 WMT 官方的那几个？

因为它们就是每年机器翻译评测用的标准测试集。你在自己的模型上跑出来的 BLEU，
可以直接和论文、商业系统、开源模型的公开分数摆在一起比。
如果自己随便切一个测试集，分数再高也不知道好不好。

真实数据是脏的，这个文件里有相当一部分代码在处理这些事：

    * 空行（原文件里段落之间有空行）
    * 编码坏行（UTF-8 解码失败留下的 U+FFFD）
    * 语言放错（中文那一列其实是英文）
    * 长度比例失衡（一句英文对着一大段中文，通常是文档级对齐的错位）
    * 重复句对

每一步过滤都会记数并打印出来 —— 数据清洗最忌讳"悄悄扔掉了几十万句"。
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

import torch

from .bpe import BPE, cjk_ratio, normalize_text
from .utils import Timer, ensure_dir, get_logger, save_json, set_seed

logger = get_logger()


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


# 训练语料：新闻评论域，中英各约 12.6 万句
NEWS_COMMENTARY = CorpusSpec(
    name="News-Commentary",
    url="https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/en-zh.txt.zip",
    domain="新闻评论",
    note="WMT 官方指定的训练语料之一",
)

# 验证 / 测试语料：WMT 历年 newsdev 与 newstest
WMT_NEWS = CorpusSpec(
    name="WMT-News",
    url="https://object.pouta.csc.fi/OPUS-WMT-News/v2019/moses/en-zh.txt.zip",
    domain="新闻",
    note="newsdev2017 + newstest2017/2018/2019，含官方参考译文",
)

# 可选加餐：换领域做数据规模实验
EXTRA_CORPORA: Dict[str, CorpusSpec] = {
    "ted2013": CorpusSpec(
        name="TED2013",
        url="https://object.pouta.csc.fi/OPUS-TED2013/v1.1/moses/en-zh.txt.zip",
        domain="演讲口语",
        note="句子短、口语化，用来观察换领域后 BLEU 怎么变",
        default_max_pairs=150000,
    ),
    "un": CorpusSpec(
        name="UN",
        url="https://object.pouta.csc.fi/OPUS-UN/v20090831/moses/en-zh.txt.zip",
        domain="联合国文件",
        note="正式书面语，语气接近新闻稿",
        default_max_pairs=70000,
    ),
}

# WMT-News 里 4 个官方数据集的文档名 -> 我们给它的 split 名
WMT_SPLITS = {
    "newsdev2017-enzh": "dev",
    "newstest2017-enzh": "test2017",
    "newstest2018-enzh": "test2018",
    "newstest2019-enzh": "test2019",
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
        *.en / *.zh   每行一句的平行文本
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
    """读 moses 格式的 .en / .zh 两个文件，按行配成句对。"""

    directory = Path(directory)
    en_files = list(directory.glob("*.en"))
    zh_files = list(directory.glob("*.zh"))
    if not en_files or not zh_files:
        raise FileNotFoundError(f"{directory} 里没有找到 *.en / *.zh 文件")

    # errors="replace" 会把坏字节换成 U+FFFD 而不是直接抛异常，
    # 这样我们能"看得见"坏行，并在清洗阶段把它们挑出来。
    def read_lines(path: Path) -> List[str]:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()

    source = read_lines(en_files[0])
    target = read_lines(zh_files[0])
    if len(source) != len(target):
        raise ValueError(
            f"两个文件行数不一致：{en_files[0].name}={len(source)}，{zh_files[0].name}={len(target)}"
        )
    return list(zip(source, target))


def read_wmt_news_splits(directory: Path) -> Dict[str, List[Tuple[str, str]]]:
    """从 WMT-News 里切出官方的 dev / test 集。

    做法：xml 里的每个 <linkGrp> 对应一个文档（newsdev2017-enzh 之类），
    它们的先后顺序和 .en/.zh 文件的行顺序一致。按 linkGrp 的条数累加偏移量，
    就能把整块行区间还原成一个个测试集。

    这里只取 "-enzh" 方向的文档；包里还有一份 "-zhen"，是同样的句子反过来，
    混进来会让测试集凭空翻倍。
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
@dataclass
class CleanConfig:
    """清洗阈值。改这些数字前先看一眼被丢掉的原因分布。"""

    max_chars: int = 400             # 单侧字符数上限（编码后还会再按 token 数过滤）
    min_chars: int = 1
    min_cjk_ratio: float = 0.25      # 中文那一列至少要有这么多汉字
    max_src_cjk_ratio: float = 0.20  # 英文那一列不该有多少汉字
    min_latin_ratio: float = 0.40    # 英文那一列至少要有这么多拉丁字母
    min_length_ratio: float = 0.20   # 中文/英文字符数比例的下限
    max_length_ratio: float = 2.50   # 上限


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
    stripped = "".join(text.split())
    if not stripped:
        return 0.0
    latin = sum(1 for ch in stripped if ch.isascii() and ch.isalpha())
    return latin / len(stripped)


def clean_pair(
    source: str,
    target: str,
    stats: CleanStats,
    config: CleanConfig,
) -> Optional[Tuple[str, str]]:
    """清洗一句句对。返回 None 表示丢弃，并会在 stats 里记下原因。"""

    stats.seen += 1
    source = normalize_text(source)
    target = normalize_text(target)

    if "\ufffd" in source or "\ufffd" in target:
        stats.drop("编码坏行（含替换字符 U+FFFD）")
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

    if cjk_ratio(target) < config.min_cjk_ratio:
        stats.drop("中文侧汉字比例过低（可能语言放错）")
        return None
    if cjk_ratio(source) > config.max_src_cjk_ratio:
        stats.drop("英文侧混入大量汉字")
        return None
    if _latin_ratio(source) < config.min_latin_ratio:
        stats.drop("英文侧拉丁字母比例过低")
        return None

    # 长度比例：中文译文的字符数一般是英文的 0.3~0.8 倍。
    # 偏离太远通常意味着对齐错位（一句英文对着一整段中文）。
    length_ratio = len(target) / max(1, len(source))
    if not config.min_length_ratio <= length_ratio <= config.max_length_ratio:
        stats.drop("中英长度比例失衡（疑似对齐错位）")
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
        tgt: <bos> + 中文 id + <eos>
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
    # 同时存一份纯文本：评测 BLEU 时用得上，也能直接打开 human eye 检查数据
    if kept_pairs:
        path = Path(path)
        (path.parent / f"{path.stem}.src.en").write_text(
            "\n".join(source for source, _ in kept_pairs) + "\n", encoding="utf-8"
        )
        (path.parent / f"{path.stem}.ref.zh").write_text(
            "\n".join(target for _, target in kept_pairs) + "\n", encoding="utf-8"
        )


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build_dataset(
    data_dir: str | Path = "data",
    out_dir: str | Path = "data/ready",
    vocab_size: int = 16000,
    max_src_len: int = 192,
    max_tgt_len: int = 192,
    extras: Sequence[str] = (),
    extra_pairs: int = 0,
    max_train_pairs: int = 0,
    bpe_train_lines: int = 200000,
    skip_download: bool = False,
    seed: int = 2024,
) -> Dict[str, object]:
    """完整流程：下载 -> 清洗 -> 训练 BPE -> 编码 -> 落盘。"""

    set_seed(seed)
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    raw_dir = ensure_dir(data_dir / "raw")

    clean_config = CleanConfig()
    meta: Dict[str, object] = {"vocab_size": vocab_size, "sources": {}}

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
    for source, target in train_pairs[:bpe_train_lines]:
        bpe_texts.append(source)
        bpe_texts.append(target)
    with Timer() as timer:
        tokenizer = BPE.train(
            bpe_texts,
            vocab_size=vocab_size,
            min_frequency=2,
            progress=lambda done, total: logger.info(f"    BPE 合并进度 {done}/{total}"),
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
            f"中文 {stats['tgt_tokens']:>9} token  "
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
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--max-src-len", type=int, default=192, help="英文侧最大 token 数")
    parser.add_argument("--max-tgt-len", type=int, default=192, help="中文侧最大 token 数")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        choices=sorted(EXTRA_CORPORA),
        help="额外加入的语料（可重复），用来做数据规模/领域实验",
    )
    parser.add_argument("--extra-pairs", type=int, default=0, help="额外语料最多取多少句，0=用默认值")
    parser.add_argument("--max-train-pairs", type=int, default=0, help="训练集上限，0=不限制")
    parser.add_argument("--bpe-train-lines", type=int, default=200000, help="训练 BPE 时最多看多少句")
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
        bpe_train_lines=args.bpe_train_lines,
        skip_download=args.skip_download,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

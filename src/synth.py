"""模板化中英平行语料生成器。

## 为什么不用真实数据集

WMT / Multi30k / Tatoeba 动辄几十万到上千万句，在笔记本上训练一周也看不到
BLEU 明显上升。学习阶段最重要的是**看见整条流水线跑通、且指标可解释地上升**，
所以这里用一组模板把"真实翻译难点"压缩成可在几分钟内学会的小语料。

## 这个生成器刻意埋进的翻译难点

| 难点 | 中文 | 英文 | 考察的能力 |
| --- | --- | --- | --- |
| 语序 | 时间+地点+动词+宾语 | 动词+宾语+地点+时间 | 全局重排 |
| 形态 | 他学习 | he studies | 第三人称单数 |
| 复数 | 三本书 | three books | 数词与名词一致 |
| 量词 | 一本/三台/两张 | a / three / two | 量词在英文中消失 |
| 冠词 | 一本书 | a book / an email | a vs an |
| 否定 | 他不买 | he does not buy | 助动词 do/does |
| 疑问 | 他买吗？ | does he buy ? | 助动词前置 |
| 疑问词 | 他为什么买？ | why does he buy ? | wh- 移位（中文原地不动） |
| 指代 | 我喜欢这本书，但是他不喜欢 | i like this book , but he does not like it | 省略宾语的代词化（it/them） |
| 关联词 | 因为…所以 / 但是 | but / because | 从句连接 |

生成器是**确定性**的：同样的 seed 永远得到同样的语料，便于对比实验。

## 接入你自己的真实数据

只要写成 `中文<TAB>英文` 的 TSV，就能被 src/data.py 直接读取，
见 README「用你自己的数据训练」一节。
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

Pair = Tuple[str, str]

# ---------------------------------------------------------------------- #
# 词典
# ---------------------------------------------------------------------- #
# (中文主语, 英文主语, 是否第三人称单数)
SUBJECTS: List[Tuple[str, str, bool]] = [
    ("我", "i", False),
    ("你", "you", False),
    ("他", "he", True),
    ("她", "she", True),
    ("我们", "we", False),
    ("他们", "they", False),
    ("李老师", "ms li", True),
    ("王先生", "mr wang", True),
]

# 中文所有格 -> 英文所有格
POSSESSIVES: List[Tuple[str, str]] = [
    ("我的", "my"),
    ("你的", "your"),
    ("他的", "his"),
    ("她的", "her"),
    ("我们的", "our"),
    ("他们的", "their"),
]

# 名词：zh 单数形式, en 单数, en 复数, 冠词, 量词, 是否可数
NOUNS: Dict[str, Dict[str, object]] = {
    "book":     {"zh": "书",   "sg": "book",    "pl": "books",    "article": "a",  "measure": "本", "countable": True},
    "computer": {"zh": "电脑", "sg": "computer", "pl": "computers", "article": "a",  "measure": "台", "countable": True},
    "phone":    {"zh": "手机", "sg": "phone",   "pl": "phones",   "article": "a",  "measure": "部", "countable": True},
    "ticket":   {"zh": "票",   "sg": "ticket",  "pl": "tickets",  "article": "a",  "measure": "张", "countable": True},
    "coffee":   {"zh": "咖啡", "sg": "cup of coffee", "pl": "cups of coffee", "article": "a", "measure": "杯", "countable": True},
    "apple":    {"zh": "苹果", "sg": "apple",   "pl": "apples",   "article": "an", "measure": "个", "countable": True},
    "dress":    {"zh": "裙子", "sg": "dress",   "pl": "dresses",  "article": "a",  "measure": "件", "countable": True},
    "car":      {"zh": "车",   "sg": "car",     "pl": "cars",     "article": "a",  "measure": "辆", "countable": True},
    "cat":      {"zh": "猫",   "sg": "cat",     "pl": "cats",     "article": "a",  "measure": "只", "countable": True},
    "table":    {"zh": "桌子", "sg": "table",   "pl": "tables",   "article": "a",  "measure": "张", "countable": True},
    "photo":    {"zh": "照片", "sg": "photo",   "pl": "photos",   "article": "a",  "measure": "张", "countable": True},
    "movie":    {"zh": "电影", "sg": "movie",   "pl": "movies",   "article": "a",  "measure": "部", "countable": True},
    "report":   {"zh": "报告", "sg": "report",  "pl": "reports",  "article": "a",  "measure": "份", "countable": True},
    "plan":     {"zh": "计划", "sg": "plan",    "pl": "plans",    "article": "a",  "measure": "个", "countable": True},
    "problem":  {"zh": "问题", "sg": "problem", "pl": "problems", "article": "a",  "measure": "个", "countable": True},
    "project":  {"zh": "项目", "sg": "project", "pl": "projects", "article": "a",  "measure": "个", "countable": True},
    "city":     {"zh": "城市", "sg": "city",    "pl": "cities",   "article": "a",  "measure": "座", "countable": True},
    "student":  {"zh": "学生", "sg": "student", "pl": "students", "article": "a",  "measure": "个", "countable": True},
    "teacher":  {"zh": "老师", "sg": "teacher", "pl": "teachers", "article": "a",  "measure": "位", "countable": True},
    "meeting":  {"zh": "会议", "sg": "meeting", "pl": "meetings", "article": "a",  "measure": "个", "countable": True},
    "email":    {"zh": "邮件", "sg": "email",   "pl": "emails",   "article": "an", "measure": "封", "countable": True},
    "key":      {"zh": "钥匙", "sg": "key",     "pl": "keys",     "article": "a",  "measure": "把", "countable": True},
    "exam":     {"zh": "考试", "sg": "exam",    "pl": "exams",    "article": "an", "measure": "场", "countable": True},
    "chinese":  {"zh": "中文", "sg": "chinese", "pl": "chinese",  "article": "",   "measure": "",   "countable": False},
    "english":  {"zh": "英语", "sg": "english", "pl": "english",  "article": "",   "measure": "",   "countable": False},
    "time":     {"zh": "时间", "sg": "time",    "pl": "time",     "article": "",   "measure": "",   "countable": False},
    "help":     {"zh": "帮助", "sg": "help",    "pl": "help",     "article": "",   "measure": "",   "countable": False},
}

# 名词短语里的形容词（用于「一本新书」/「a new book」）
NP_ADJECTIVES: List[Tuple[str, str]] = [
    ("新", "new"),
    ("旧", "old"),
    ("贵", "expensive"),
    ("便宜", "cheap"),
    ("有趣", "interesting"),
    ("重要", "important"),
    ("漂亮", "beautiful"),
]

# 表语形容词（用于「我很累」这类系表结构）
PREDICATES: List[Tuple[str, str]] = [
    ("累", "tired"),
    ("忙", "busy"),
    ("高兴", "happy"),
    ("饿", "hungry"),
    ("困", "sleepy"),
    ("满意", "satisfied"),
    ("紧张", "nervous"),
]

# 数字：中文, 英文, 是否复数
NUMERALS: List[Tuple[str, str, bool]] = [
    ("一", "one", False),
    ("两", "two", True),
    ("三", "three", True),
    ("四", "four", True),
    ("五", "five", True),
    ("六", "six", True),
    ("七", "seven", True),
    ("八", "eight", True),
    ("九", "nine", True),
    ("十", "ten", True),
]

TIMES: List[Tuple[str, str]] = [
    ("今天", "today"),
    ("明天", "tomorrow"),
    ("昨天", "yesterday"),
    ("今天早上", "this morning"),
    ("今天晚上", "tonight"),
    ("每天", "every day"),
    ("每天早上", "every morning"),
    ("这个周末", "this weekend"),
    ("下周一", "next monday"),
    ("现在", "now"),
]

PLACES: List[Tuple[str, str]] = [
    ("在家", "at home"),
    ("在学校", "at school"),
    ("在公司", "at the office"),
    ("在图书馆", "in the library"),
    ("在公园里", "in the park"),
    ("在北京", "in beijing"),
    ("在火车上", "on the train"),
    ("在网上", "online"),
    ("在教室", "in the classroom"),
    ("在机场", "at the airport"),
    ("在餐厅", "at the restaurant"),
]

TIMES_QUES: List[Tuple[str, str]] = [
    ("今天", "today"),
    ("明天", "tomorrow"),
    ("每天晚上", "every evening"),
    ("这个周末", "this weekend"),
]

# 动词：中文, 英文原形, 英文第三人称单数, 可以搭配的名词 keys
VERBS: List[Dict[str, object]] = [
    {"zh": "买",   "en": "buy",      "en3": "buys",       "objs": ["book", "computer", "ticket", "apple", "coffee", "dress", "car", "phone"]},
    {"zh": "看",   "en": "watch",    "en3": "watches",    "objs": ["movie", "photo", "book", "report"]},
    {"zh": "读",   "en": "read",     "en3": "reads",      "objs": ["book", "report", "email", "chinese"]},
    {"zh": "写",   "en": "write",    "en3": "writes",     "objs": ["report", "email", "plan", "book"]},
    {"zh": "讨论", "en": "discuss",  "en3": "discusses",  "objs": ["problem", "plan", "report", "project"]},
    {"zh": "完成", "en": "finish",   "en3": "finishes",   "objs": ["plan", "report", "project", "exam"]},
    {"zh": "准备", "en": "prepare",  "en3": "prepares",   "objs": ["meeting", "plan", "report"]},
    {"zh": "找到", "en": "find",     "en3": "finds",      "objs": ["key", "phone", "book", "computer"]},
    {"zh": "学习", "en": "study",    "en3": "studies",    "objs": ["chinese", "english"]},
    {"zh": "参加", "en": "attend",   "en3": "attends",    "objs": ["meeting", "exam"]},
    {"zh": "需要", "en": "need",     "en3": "needs",      "objs": ["ticket", "time", "help", "computer"]},
    {"zh": "修理", "en": "repair",   "en3": "repairs",    "objs": ["computer", "car", "table", "phone"]},
    {"zh": "喜欢", "en": "like",     "en3": "likes",      "objs": ["book", "movie", "city", "cat", "coffee", "car", "dress"]},
]

# 情态动词：中文, 英文（非第三人称）, 英文（第三人称单数）
MODALS: List[Tuple[str, str, str]] = [
    ("想", "want to", "wants to"),
    ("打算", "plan to", "plans to"),
    ("决定", "decide to", "decides to"),
    ("希望", "hope to", "hopes to"),
    ("需要", "need to", "needs to"),
]

# 疑问词：中文, 英文, 能否跟时间/地点
WH_WORDS: List[Tuple[str, str]] = [("为什么", "why"), ("什么时候", "when"), ("在哪里", "where"), ("怎么", "how")]


# ---------------------------------------------------------------------- #
# 小工具
# ---------------------------------------------------------------------- #
def _be(subject_en: str, is_third: bool) -> str:
    if subject_en == "i":
        return "am"
    return "is" if is_third else "are"


def _do(is_third: bool) -> str:
    return "does" if is_third else "do"


def _build_np(rng: random.Random, noun_key: str) -> Tuple[str, str, bool]:
    """生成一个名词短语，返回 (中文, 英文, 是否复数)。

    四种形式：
        这/那 + 量词 + 名词   ->  this / that + 名词      （这本书 / this book）
        数词 + 量词 + 名词    ->  a + 名词 / 数词 + 复数   （三本书 / three books）
        所有格 + 量词? + 名词 ->  my / his + 名词         （我的书 / my book）
        光杆名词              ->  the + 名词              （书 / the book）
    """
    noun = NOUNS[noun_key]
    zh_noun = str(noun["zh"])
    singular = str(noun["sg"])
    plural = str(noun["pl"])
    article = str(noun["article"])
    measure = str(noun["measure"])
    countable = bool(noun["countable"])

    if countable:
        form = rng.choice(["dem", "dem", "num", "num", "poss", "bare"])
    else:
        form = rng.choice(["poss", "bare", "bare"])

    # 形容词（只挂在可数名词上，且以一定概率出现）
    adj_zh = adj_en = ""
    if countable and rng.random() < 0.35:
        adj_zh, adj_en = rng.choice(NP_ADJECTIVES)

    if form == "dem":
        this_or_that = rng.random() < 0.5
        zh = f"{'这' if this_or_that else '那'}{measure}{adj_zh}{zh_noun}"
        en = f"{'this' if this_or_that else 'that'} {adj_en + ' ' if adj_en else ''}{singular}"
        return zh, en, False

    if form == "num":
        num_zh, num_en, is_plural = rng.choice(NUMERALS)
        if num_zh == "一":
            # 「一本书」-> a book，冠词由名词决定（a / an）
            assert not is_plural
            zh = f"一{measure}{adj_zh}{zh_noun}"
            en = f"{article} {adj_en + ' ' if adj_en else ''}{singular}"
            return zh, en, False
        zh = f"{num_zh}{measure}{adj_zh}{zh_noun}"
        en = f"{num_en} {adj_en + ' ' if adj_en else ''}{plural}"
        return zh, en, True

    if form == "poss":
        poss_zh, poss_en = rng.choice(POSSESSIVES)
        zh = f"{poss_zh}{adj_zh}{zh_noun}"
        en = f"{poss_en} {adj_en + ' ' if adj_en else ''}{singular}"
        return zh, en, False

    # 光杆名词：可数名词英文用定冠词，不可数名词（中文/英语/时间/帮助）不加冠词
    zh = f"{adj_zh}{zh_noun}" if adj_zh else zh_noun
    if countable:
        en = f"the {adj_en + ' ' if adj_en else ''}{singular}"
    else:
        en = f"{adj_en + ' ' if adj_en else ''}{singular}"
    return zh, en, False


def _tail_zh(time_zh: str, place_zh: str) -> str:
    """中文状语顺序：时间 + 地点 + 动词。"""
    return f"{time_zh}{place_zh}"


def _tail_en(place_en: str, time_en: str) -> List[str]:
    """英文状语顺序：动词 + 宾语 + 地点 + 时间。"""
    parts = []
    if place_en:
        parts.append(place_en)
    if time_en:
        parts.append(time_en)
    return parts


# ---------------------------------------------------------------------- #
# 主生成函数
# ---------------------------------------------------------------------- #
def generate_pairs(
    n: int,
    seed: int = 2024,
    with_time_prob: float = 0.8,
    with_place_prob: float = 0.6,
) -> List[Pair]:
    """生成 n 条不重复的 (中文, 英文) 平行句对。"""
    rng = random.Random(seed)
    seen = set()
    pairs: List[Pair] = []
    attempts = 0
    max_attempts = n * 400

    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        pattern = rng.choice(["decl", "decl", "neg", "question", "modal", "copula", "contrast", "wh"])

        subj_zh, subj_en, is3 = rng.choice(SUBJECTS)
        verb = rng.choice(VERBS)
        verb_zh, verb_en, verb_en3 = str(verb["zh"]), str(verb["en"]), str(verb["en3"])
        noun_key = rng.choice(list(verb["objs"]))  # type: ignore[arg-type]
        obj_zh, obj_en, obj_plural = _build_np(rng, noun_key)

        time_zh, time_en = rng.choice(TIMES) if rng.random() < with_time_prob else ("", "")
        place_zh, place_en = rng.choice(PLACES) if rng.random() < with_place_prob else ("", "")

        # ---------------- 1. 陈述句 ---------------- #
        if pattern == "decl":
            zh = f"{subj_zh}{_tail_zh(time_zh, place_zh)}{verb_zh}{obj_zh}。"
            en_parts = [subj_en, verb_en3 if is3 else verb_en, obj_en] + _tail_en(place_en, time_en)
            en = " ".join(en_parts) + " ."

        # ---------------- 2. 否定句 ---------------- #
        elif pattern == "neg":
            zh = f"{subj_zh}{_tail_zh(time_zh, place_zh)}不{verb_zh}{obj_zh}。"
            en_parts = [subj_en, _do(is3), "not", verb_en, obj_en] + _tail_en(place_en, time_en)
            en = " ".join(en_parts) + " ."

        # ---------------- 3. 一般疑问句 ---------------- #
        elif pattern == "question":
            zh = f"{subj_zh}{_tail_zh(time_zh, place_zh)}{verb_zh}{obj_zh}吗？"
            en_parts = [_do(is3), subj_en, verb_en, obj_en] + _tail_en(place_en, time_en)
            en = " ".join(en_parts) + " ?"

        # ---------------- 4. 情态动词 ---------------- #
        elif pattern == "modal":
            modal_zh, modal_en, modal_en3 = rng.choice(MODALS)
            zh = f"{subj_zh}{_tail_zh(time_zh, place_zh)}{modal_zh}{verb_zh}{obj_zh}。"
            en_parts = [subj_en, modal_en3 if is3 else modal_en, verb_en, obj_en] + _tail_en(place_en, time_en)
            en = " ".join(en_parts) + " ."

        # ---------------- 5. 系表结构 ---------------- #
        elif pattern == "copula":
            pred_zh, pred_en = rng.choice(PREDICATES)
            zh = f"{subj_zh}{time_zh}很{pred_zh}。"
            en_parts = [subj_en, _be(subj_en, is3), "very", pred_en]
            if time_en:
                en_parts.append(time_en)
            en = " ".join(en_parts) + " ."

        # ---------------- 6. 转折 + 代词指代 ---------------- #
        elif pattern == "contrast":
            subj2_zh, subj2_en, is3b = rng.choice(SUBJECTS)
            if subj2_zh == subj_zh:  # 避免「我喜欢…，但是我不喜欢」这种同主语句子
                continue
            zh = f"{subj_zh}{_tail_zh(time_zh, place_zh)}{verb_zh}{obj_zh}，但是{subj2_zh}不{verb_zh}。"
            pronoun = "them" if obj_plural else "it"
            en_parts = (
                [subj_en, verb_en3 if is3 else verb_en, obj_en]
                + _tail_en(place_en, time_en)
                + [",", "but", subj2_en, _do(is3b), "not", verb_en, pronoun, "."]
            )
            en = " ".join(en_parts)

        # ---------------- 7. 疑问词（wh- 移位） ---------------- #
        else:
            wh_zh, wh_en = rng.choice(WH_WORDS)
            if wh_zh == "什么时候":
                time_zh, time_en = rng.choice(TIMES_QUES)
                zh = f"{subj_zh}{wh_zh}{place_zh}{verb_zh}{obj_zh}？"
                en_parts = [wh_en, _do(is3), subj_en, verb_en, obj_en]
                if place_en:
                    en_parts.append(place_en)
            elif wh_zh == "在哪里":
                place_zh = ""
                zh = f"{subj_zh}{time_zh}{wh_zh}{verb_zh}{obj_zh}？"
                en_parts = [wh_en, _do(is3), subj_en, verb_en, obj_en]
                if time_en:
                    en_parts.append(time_en)
            else:
                zh = f"{subj_zh}{time_zh}{wh_zh}{place_zh}{verb_zh}{obj_zh}？"
                en_parts = [wh_en, _do(is3), subj_en, verb_en, obj_en] + _tail_en(place_en, time_en)
            en = " ".join(en_parts) + " ?"

        key = (zh, en)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    if len(pairs) < n:
        raise RuntimeError(
            f"只生成了 {len(pairs)}/{n} 条唯一句对，请增大模板空间或降低 n"
        )
    return pairs


def split_pairs(
    pairs: Sequence[Pair],
    dev_size: int,
    test_size: int,
    seed: int = 2024,
) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    """随机切分为 train / dev / test（三者互不重叠）。"""
    shuffled = list(pairs)
    random.Random(seed + 1).shuffle(shuffled)
    if dev_size + test_size >= len(shuffled):
        raise ValueError("dev/test 划分过大")
    test = shuffled[:test_size]
    dev = shuffled[test_size : test_size + dev_size]
    train = shuffled[test_size + dev_size :]
    return train, dev, test


def write_tsv(pairs: Sequence[Pair], path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for zh, en in pairs:
            f.write(f"{zh}\t{en}\n")

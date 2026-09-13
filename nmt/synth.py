"""合成平行语料：不联网、不下载，用来写测试和跑冒烟实验。

它刻意覆盖几个真正会让翻译模型犯错的点：

    * 语序差异：英文的 "at school" 在中文里要挪到动词前面
    * 时态：will / did / is -ing 分别对应 会 / 已经……了 / 正在
    * 疑问句：英文把助动词提到句首，中文用"吗？"结尾
    * 否定：not -> 不
    * 主谓一致：he discusses，而 I discuss

⚠️ 不要用合成语料评估翻译质量。模板句太规律，模型能整句记住，
BLEU 会高得离谱（教学项目里最常见的坑）。真实评测请用 WMT 官方测试集。
"""

from __future__ import annotations

import random
from typing import List, Tuple

# (英文, 中文, 是不是第三人称单数, be 动词形式)
SUBJECTS = [
    ("I", "我", False, "am"),
    ("you", "你", False, "are"),
    ("he", "他", True, "is"),
    ("she", "她", True, "is"),
    ("my colleague", "我的同事", True, "is"),
    ("the engineer", "这位工程师", True, "is"),
    ("the manager", "经理", True, "is"),
    ("our team", "我们团队", True, "is"),
    ("the new intern", "新来的实习生", True, "is"),
]

PLACES = [
    ("at school", "在学校"),
    ("in the office", "在办公室"),
    ("at the conference", "在会议上"),
    ("in Beijing", "在北京"),
    ("at home", "在家"),
]

TIMES = [
    ("tomorrow", "明天"),
    ("next week", "下周"),
    ("this afternoon", "今天下午"),
    ("on Monday", "周一"),
    ("tonight", "今晚"),
]

# (原形, 第三人称单数, 现在分词, 过去式, 中文)
ACTIONS = [
    ("discuss the problem", "discusses the problem", "discussing the problem", "discussed the problem", "讨论这个问题"),
    ("review the report", "reviews the report", "reviewing the report", "reviewed the report", "审阅这份报告"),
    ("finish the project", "finishes the project", "finishing the project", "finished the project", "完成这个项目"),
    ("read the document", "reads the document", "reading the document", "read the document", "阅读这份文件"),
    ("prepare the meeting", "prepares the meeting", "preparing the meeting", "prepared the meeting", "准备会议"),
    ("test the system", "tests the system", "testing the system", "tested the system", "测试系统"),
    ("update the schedule", "updates the schedule", "updating the schedule", "updated the schedule", "更新日程"),
    ("check the numbers", "checks the numbers", "checking the numbers", "checked the numbers", "核对数据"),
]


def _sentence(subject, place, time, action, tense: str, question: bool, negative: bool) -> Tuple[str, str]:
    en_subject, zh_subject, third, be = subject
    en_place, zh_place = place
    en_time, zh_time = time
    en_base, en_s, en_ing, en_past, zh_verb = action

    # ---------------------------------------------------------------- 中文
    # 主语 + 时间 + 地点 + [不] + [会/已经/正在] + 动词 + [了] + 问号
    zh = f"{zh_subject}{zh_time}{zh_place}"
    if negative:
        zh += "不"
    zh += {"future": "会", "past": "已经", "progressive": "正在"}.get(tense, "")
    zh += zh_verb
    if tense == "past":
        zh += "了"
    zh += "吗？" if question else "。"

    # ---------------------------------------------------------------- 英文
    if tense == "present":
        affirmative = en_s if third else en_base
        auxiliary, main = ("does", en_base) if third else ("do", en_base)
    elif tense == "future":
        affirmative = f"will {en_base}"
        auxiliary, main = "will", en_base
    elif tense == "past":
        affirmative = en_past
        auxiliary, main = "did", en_base
    else:  # progressive
        affirmative = f"{be} {en_ing}"
        auxiliary, main = be, en_ing

    if question:
        en = f"{auxiliary.capitalize()} {en_subject} {'not ' if negative else ''}{main} {en_time} {en_place}"
        return f"{en}?", zh
    if negative:
        en = f"{en_subject} {auxiliary} not {main} {en_time} {en_place}"
    else:
        en = f"{en_subject} {affirmative} {en_time} {en_place}"
    return f"{en}.", zh


def toy_pairs(count: int = 200, seed: int = 2024) -> List[Tuple[str, str]]:
    """生成 count 句去重后的 (英文, 中文) 平行句对。"""

    generator = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    seen = set()
    attempts = 0
    while len(pairs) < count and attempts < count * 80:
        attempts += 1
        pair = _sentence(
            generator.choice(SUBJECTS),
            generator.choice(PLACES),
            generator.choice(TIMES),
            generator.choice(ACTIONS),
            generator.choice(["present", "future", "past", "progressive"]),
            generator.random() < 0.25,
            generator.random() < 0.2,
        )
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs

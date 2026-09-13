"""合成英德平行语料：不联网、不下载，用来写测试和跑冒烟实验。

它刻意覆盖几个真正会让翻译模型犯错的点：

    * 语序差异：德语的时间/地点状语顺序、以及"情态动词把实义动词挤到句尾"
      （wird ... besprechen / hat ... besprochen）—— 英语没有这种"动词框"结构
    * 疑问句：英语靠助动词提前（Does he ...），德语直接把变位动词提到句首（Bespricht er ...）
    * 否定：英语 not，德语 nicht，而且位置完全不同（德语放在句末或不定式前）
    * 形态：德语现在时第三人称变位（besprechen -> bespricht）、完成时分词
      （besprochen / gelesen / geprüft），英语则是 s-form 和过去式
    * 大写：德语所有名词首字母大写，英语除了句首和专有名词都小写

为了保持模板正确，这里**只用第三人称单数主语**：德语动词变位跟着人称变
（ich bespreche / du besprichst / er bespricht），把所有变位形式都塞进模板
会让这个夹具变得又长又容易写错。它存在的意义是"给测试提供可控的句对"，
不是覆盖德语语法。

⚠️ 不要用合成语料评估翻译质量。模板句太规律，模型能整句记住，
BLEU 会高得离谱（教学项目里最常见的坑）。真实评测请用 WMT 官方测试集。
"""

from __future__ import annotations

import random
from typing import List, Tuple

# 全部是第三人称单数，英语用 "does / is / has"，德语变位固定
SUBJECTS = [
    ("he", "er"),
    ("she", "sie"),
    ("my colleague", "mein Kollege"),
    ("the engineer", "der Ingenieur"),
    ("the manager", "der Manager"),
    ("our team", "unser Team"),
    ("the new intern", "der neue Praktikant"),
]

PLACES = [
    ("at school", "in der Schule"),
    ("in the office", "im Büro"),
    ("at the conference", "auf der Konferenz"),
    ("in Beijing", "in Peking"),
    ("at home", "zu Hause"),
]

TIMES = [
    ("tomorrow", "morgen"),
    ("next week", "nächste Woche"),
    ("this afternoon", "heute Nachmittag"),
    ("on Monday", "am Montag"),
    ("tonight", "heute Abend"),
]

# (英语原形, 英语三单, 英语现在分词, 英语过去式,
#  德语不定式, 德语三单变位, 德语完成时分词, 德语宾语)
ACTIONS = [
    ("discuss the problem", "discusses the problem", "discussing the problem", "discussed the problem",
     "besprechen", "bespricht", "besprochen", "das Problem"),
    ("review the report", "reviews the report", "reviewing the report", "reviewed the report",
     "prüfen", "prüft", "geprüft", "den Bericht"),
    ("finish the project", "finishes the project", "finishing the project", "finished the project",
     "beenden", "beendet", "beendet", "das Projekt"),
    ("read the document", "reads the document", "reading the document", "read the document",
     "lesen", "liest", "gelesen", "das Dokument"),
    ("prepare the meeting", "prepares the meeting", "preparing the meeting", "prepared the meeting",
     "planen", "plant", "geplant", "das Meeting"),
    ("test the system", "tests the system", "testing the system", "tested the system",
     "testen", "testet", "getestet", "das System"),
    ("update the schedule", "updates the schedule", "updating the schedule", "updated the schedule",
     "aktualisieren", "aktualisiert", "aktualisiert", "den Zeitplan"),
    ("check the numbers", "checks the numbers", "checking the numbers", "checked the numbers",
     "kontrollieren", "kontrolliert", "kontrolliert", "die Zahlen"),
]


def _capitalize(text: str) -> str:
    """句首字母大写（德语名词本身已经是大写，这里只处理第一个字符）。"""

    return text[:1].upper() + text[1:] if text else text


def _sentence(subject, place, time, action, tense: str, question: bool, negative: bool) -> Tuple[str, str]:
    en_subject, de_subject = subject
    en_place, de_place = place
    en_time, de_time = time
    en_base, en_s, en_ing, en_past, de_inf, de_pres, de_participle, de_object = action

    # ---------------------------------------------------------------- 德语
    # 陈述句：主语 + 变位动词 + 宾语 + 时间 + 地点
    # 将来 / 完成时变成"动词框"，实义动词被挤到句尾（wird ... besprechen）
    # 进行时在德语里不存在，用 "gerade + 现在时"，并省掉时间状语
    # （否则会出现 "gerade ... morgen" 这种别扭的组合）
    if tense == "progressive":
        middle, en_tail = f"gerade {de_object}", ""
    else:
        middle, en_tail = f"{de_object} {de_time} {de_place}", f" {en_time} {en_place}"

    if question:
        # 疑问句也要带上否定词，否则英德两侧说的不是同一件事
        if tense == "future":
            de = f"Wird {de_subject} {middle} {'nicht ' if negative else ''}{de_inf}?"
        elif tense == "past":
            de = f"Hat {de_subject} {middle} {'nicht ' if negative else ''}{de_participle}?"
        else:
            # 是非问句：变位动词提前到句首（对应英语的助动词提前）
            de = f"{de_pres} {de_subject} {middle}{' nicht' if negative else ''}?"
    elif negative:
        # nicht 的位置：变位动词之后、不定式/分词之前
        if tense == "future":
            de = f"{de_subject} wird {middle} nicht {de_inf}."
        elif tense == "past":
            de = f"{de_subject} hat {middle} nicht {de_participle}."
        else:
            de = f"{de_subject} {de_pres} {middle} nicht."
    else:
        if tense == "future":
            de = f"{de_subject} wird {middle} {de_inf}."
        elif tense == "past":
            de = f"{de_subject} hat {middle} {de_participle}."
        else:
            de = f"{de_subject} {de_pres} {middle}."
    de = _capitalize(" ".join(de.split()))

    # ---------------------------------------------------------------- 英语
    if tense == "present":
        affirmative = en_s
        auxiliary, main = "does", en_base
    elif tense == "future":
        affirmative = f"will {en_base}"
        auxiliary, main = "will", en_base
    elif tense == "past":
        affirmative = en_past
        auxiliary, main = "did", en_base
    else:  # progressive
        affirmative = f"is {en_ing}"
        auxiliary, main = "is", en_ing

    if question:
        en = f"{auxiliary.capitalize()} {en_subject} {'not ' if negative else ''}{main}{en_tail}?"
    elif negative:
        en = f"{en_subject} {auxiliary} not {main}{en_tail}."
    else:
        en = f"{en_subject} {affirmative}{en_tail}."
    en = _capitalize(" ".join(en.split()))

    return en, de


def toy_pairs(count: int = 200, seed: int = 2024) -> List[Tuple[str, str]]:
    """生成 count 句去重后的 (英文, 德文) 平行句对。"""

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

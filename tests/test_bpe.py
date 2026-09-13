"""分词器测试：编码解码要能往返，特殊符号和 OOV 要处理对。"""

from __future__ import annotations

import torch

from nmt.bpe import BOS_ID, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, BPE


def test_roundtrip_chinese(tokenizer: BPE) -> None:
    text = "他明天在学校讨论这个问题吗？"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_roundtrip_chinese_with_inner_punctuation(tokenizer: BPE) -> None:
    """中文标点是独立的 token，解码时前后都不能留下空格。

    这里用 "。" 而不是 "，"：测试夹具的分词器只在小规模合成语料上训练过，
    语料里没有 "，" 这个字符，用它会被编码成 <unk>，那是另一回事。
    """

    text = "他明天在学校讨论这个问题。我下周在办公室审阅这份报告。"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_english(tokenizer: BPE) -> None:
    # 用合成语料里出现过的词，保证不会因为字符没进词表而变成 <unk>
    text = "he will discuss the problem tomorrow."
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_subwords_are_joined_back_into_words(tokenizer: BPE) -> None:
    """BPE 会把一个词切成几段，解码时必须拼回一个词，而不是用空格分开。"""

    text = "the engineer checks the numbers"
    pieces = tokenizer.tokenize(text)
    assert any(piece.endswith("</w>") for piece in pieces)   # 词尾标记存在
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_have_fixed_ids(tokenizer: BPE) -> None:
    # 0~3 固定是 <pad> <bos> <eos> <unk>，训练好的权重依赖这个约定
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.unk_id == 3
    assert tokenizer.id_to_token[0] == PAD_TOKEN
    assert BOS_ID == tokenizer.bos_id
    assert BOS_TOKEN == "<bos>"


def test_bos_eos_appended(tokenizer: BPE) -> None:
    ids = tokenizer.encode("你好", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


def test_unknown_character_becomes_unk(tokenizer: BPE) -> None:
    # 一个训练语料里绝对没见过的字符
    ids = tokenizer.encode("𠀋")
    assert tokenizer.unk_id in ids


def test_save_and_load_is_lossless(tokenizer: BPE, tmp_path) -> None:
    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    reloaded = BPE.load(path)
    assert len(reloaded) == len(tokenizer)
    text = "我们团队下周在北京准备会议。"
    assert reloaded.encode(text) == tokenizer.encode(text)
    assert reloaded.decode(reloaded.encode(text)) == tokenizer.decode(tokenizer.encode(text))


def test_vocab_size_is_respected(tmp_path) -> None:
    from nmt.synth import toy_pairs

    texts = [text for pair in toy_pairs(60) for text in pair]
    small = BPE.train(texts, vocab_size=120, min_frequency=1)
    assert len(small) <= 120
    bigger = BPE.train(texts, vocab_size=400, min_frequency=1)
    assert len(bigger) > len(small)


def test_chinese_run_is_one_word(tokenizer: BPE) -> None:
    """连续汉字应该先被当成一个"词"，再在词内部做 BPE，而不是一字一个 token。"""

    from nmt.bpe import pre_tokenize

    words = pre_tokenize("他明天在学校讨论这个问题吗？")
    assert words[0] == "他明天在学校讨论这个问题吗"
    assert words[1] == "？"


def test_decode_skips_specials(tokenizer: BPE) -> None:
    text = "我明天在学校讨论这个问题？"
    ids = [tokenizer.bos_id] + tokenizer.encode(text) + [tokenizer.eos_id]
    assert tokenizer.decode(ids) == text

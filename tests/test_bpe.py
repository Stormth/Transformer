"""分词器测试：编码解码要能往返，特殊符号和 OOV 要处理对。

这里的句子都取自 `nmt/synth.py` 的模板表，保证每个词都进过词表 ——
否则会出现"因为字符是 <unk> 所以往返不一致"的假失败。
"""

from __future__ import annotations

from nmt.bpe import BOS_ID, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, BPE


def test_roundtrip_german(tokenizer: BPE) -> None:
    text = "Der Ingenieur bespricht das Problem morgen in der Schule."
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_roundtrip_english(tokenizer: BPE) -> None:
    text = "He will discuss the problem tomorrow at school."
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_roundtrip_with_inner_punctuation(tokenizer: BPE) -> None:
    """句号是独立的 token，解码时前后都不能留下空格。"""

    text = "Der Ingenieur bespricht das Problem morgen in der Schule. Er testet das System heute Abend."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_subwords_are_joined_back_into_words(tokenizer: BPE) -> None:
    """BPE 会把一个词切成几段，解码时必须拼回一个词，而不是用空格分开。"""

    text = "Der Manager aktualisiert den Zeitplan."
    pieces = tokenizer.tokenize(text)
    assert any(piece.endswith("</w>") for piece in pieces)   # 词尾标记存在
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_hyphen_and_apostrophe_stay_inside_the_word() -> None:
    """连字符和撇号不切断词：E-Mail、well-known、don't 都是一个单元。

    德语靠复合词造词，E-Mail 这类写法很常见，切错了会让 BPE 学到没意义的片段。
    """

    from nmt.bpe import pre_tokenize

    assert pre_tokenize("E-Mail") == ["E-Mail"]
    assert pre_tokenize("well-known") == ["well-known"]
    assert pre_tokenize("don't") == ["don't"]
    assert pre_tokenize("Wirtschaftswachstum und E-Mail") == [
        "Wirtschaftswachstum", "und", "E-Mail",
    ]


def test_umlauts_are_single_letters() -> None:
    """ä ö ü ß 是字母，不能被当成符号拆开。"""

    from nmt.bpe import pre_tokenize

    assert pre_tokenize("Größe") == ["Größe"]
    assert pre_tokenize("für nächste Woche") == ["für", "nächste", "Woche"]


def test_special_tokens_have_fixed_ids(tokenizer: BPE) -> None:
    # 0~3 固定是 <pad> <bos> <eos> <unk>，训练好的权重依赖这个约定
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.unk_id == 3
    assert tokenizer.id_to_token[0] == PAD_TOKEN
    assert BOS_ID == tokenizer.bos_id
    assert BOS_TOKEN == "<bos>"
    assert UNK_TOKEN == "<unk>"
    assert EOS_TOKEN == "<eos>"


def test_bos_eos_appended(tokenizer: BPE) -> None:
    ids = tokenizer.encode("Der Manager", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


def test_unknown_character_becomes_unk(tokenizer: BPE) -> None:
    # 一个训练语料里绝对没见过的字符（生僻汉字）
    ids = tokenizer.encode("𠀋")
    assert tokenizer.unk_id in ids


def test_save_and_load_is_lossless(tokenizer: BPE, tmp_path) -> None:
    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    reloaded = BPE.load(path)
    assert len(reloaded) == len(tokenizer)
    text = "Unser Team wird das Meeting nächste Woche im Büro planen."
    assert reloaded.encode(text) == tokenizer.encode(text)
    assert reloaded.decode(reloaded.encode(text)) == tokenizer.decode(tokenizer.encode(text))


def test_vocab_size_is_respected(tmp_path) -> None:
    from nmt.synth import toy_pairs

    texts = [text for pair in toy_pairs(60) for text in pair]
    small = BPE.train(texts, vocab_size=120, min_frequency=1)
    assert len(small) <= 120
    bigger = BPE.train(texts, vocab_size=400, min_frequency=1)
    assert len(bigger) > len(small)


def test_decode_skips_specials(tokenizer: BPE) -> None:
    text = "Der Ingenieur testet das System heute Abend."
    ids = [tokenizer.bos_id] + tokenizer.encode(text) + [tokenizer.eos_id]
    assert tokenizer.decode(ids) == text

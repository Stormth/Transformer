"""评测指标测试。"""

from __future__ import annotations

from nmt.bleu import corpus_bleu, corpus_chrf, tokenize_for_bleu


def test_tokenize_chinese_splits_characters() -> None:
    assert tokenize_for_bleu("我们讨论问题。", lang="zh") == list("我们讨论问题")


def test_tokenize_english_lowercases_and_splits() -> None:
    assert tokenize_for_bleu("The Report, revised!", lang="en") == ["the", "report", "revised"]


def test_perfect_translation_scores_100() -> None:
    reference = ["他明天在学校讨论这个问题。"]
    score = corpus_bleu(reference, reference, lang="zh")
    assert abs(score.bleu - 100.0) < 1e-6
    assert abs(score.bp - 1.0) < 1e-9


def test_completely_wrong_translation_scores_zero() -> None:
    hypotheses = ["完全无关的内容"]
    references = ["他明天在学校讨论这个问题"]
    assert corpus_bleu(hypotheses, references, lang="zh").bleu < 1.0


def test_empty_input_is_safe() -> None:
    assert corpus_bleu([], [], lang="zh").bleu == 0.0
    assert corpus_chrf([], []) == 0.0


def test_brevity_penalty_applies() -> None:
    references = ["他明天在学校讨论这个问题并且会给出结论"]
    short = corpus_bleu(["他"], references, lang="zh")
    assert short.bp < 1.0
    assert short.ratio < 1.0


def test_chrf_range() -> None:
    hypotheses = ["他明天在学校讨论这个问题", "这是另一句话"]
    references = ["他明天在学校讨论这个问题", "这是另一个句子"]
    value = corpus_chrf(hypotheses, references)
    assert 0.0 <= value <= 100.0
    assert value > 50.0     # 两句都大同小异，chrF 应该挺高


def test_equal_detection_is_case_insensitive_for_english() -> None:
    # 句子要足够长，不然 3-gram / 4-gram 命中数都是 0，BLEU 会被定义成 0
    score = corpus_bleu(
        ["The Committee will review the report"], ["the committee will review the report"], lang="en"
    )
    assert abs(score.bleu - 100.0) < 1e-6

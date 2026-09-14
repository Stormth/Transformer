"""评测指标测试（英德，词级 13a 思路的切分）。"""

from __future__ import annotations

from nmt.bleu import corpus_bleu, corpus_chrf, tokenize_for_bleu


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize_for_bleu("Der Ingenieur, prüft!") == ["der", "ingenieur", "prüft"]


def test_tokenize_keeps_umlauts_and_sharp_s() -> None:
    # ä ö ü ß 是字母，不能被当成标点丢掉，否则德语 BLEU 会被人为压低
    assert tokenize_for_bleu("Größe für nächste Woche") == ["größe", "für", "nächste", "woche"]


def test_perfect_translation_scores_100() -> None:
    reference = ["Der Ingenieur bespricht das Problem morgen in der Schule."]
    score = corpus_bleu(reference, reference)
    assert abs(score.bleu - 100.0) < 1e-6
    assert abs(score.bp - 1.0) < 1e-9


def test_case_and_punctuation_are_ignored() -> None:
    score = corpus_bleu(
        ["DER INGENIEUR BESPRICHT DAS PROBLEM MORGEN IN DER SCHULE"],
        ["Der Ingenieur bespricht das Problem morgen in der Schule."],
    )
    assert abs(score.bleu - 100.0) < 1e-6


def test_completely_wrong_translation_scores_low() -> None:
    hypotheses = ["Die Zahlen sind völlig anders"]
    references = ["Der Ingenieur bespricht das Problem morgen in der Schule"]
    assert corpus_bleu(hypotheses, references).bleu < 1.0


def test_very_short_sentences_score_zero() -> None:
    """BLEU 的已知缺陷：句子太短时 3-gram / 4-gram 命中数必然是 0，整体被定义成 0。

    这就是训练早期 BLEU 长时间显示 0.00 的原因之一，也是要同时看 chrF 的原因。
    """

    assert corpus_bleu(["Hallo"], ["Hallo"]).bleu == 0.0


def test_empty_input_is_safe() -> None:
    assert corpus_bleu([], []).bleu == 0.0
    assert corpus_chrf([], []) == 0.0


def test_brevity_penalty_applies() -> None:
    references = ["Der Ingenieur bespricht das Problem morgen in der Schule und gibt danach eine Erklärung ab"]
    short = corpus_bleu(["Der Ingenieur"], references)
    assert short.bp < 1.0
    assert short.ratio < 1.0


def test_chrf_range() -> None:
    hypotheses = [
        "Der Ingenieur bespricht das Problem morgen in der Schule",
        "Das ist ein anderer Satz",
    ]
    references = [
        "Der Ingenieur bespricht das Problem morgen in der Schule",
        "Das ist ein anderer Satz.",
    ]
    value = corpus_chrf(hypotheses, references)
    assert 0.0 <= value <= 100.0
    assert value > 70.0     # 两句几乎一样，chrF 应该很高


def test_case_sensitive_bleu_is_stricter() -> None:
    """区分大小写时，大小写不同的译文会被扣分（WMT14 官方口径就是这样）。"""

    reference = ["Der Ingenieur bespricht das Problem morgen in der Schule."]
    hypothesis = ["der ingenieur bespricht das problem morgen in der schule."]

    loose = corpus_bleu(hypothesis, reference, lowercase=True).bleu
    strict = corpus_bleu(hypothesis, reference, lowercase=False).bleu
    assert loose > 99.0      # 只差大小写，转小写后完全一致
    assert strict < 60.0     # 区分大小写就露馅了

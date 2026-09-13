"""核心正确性测试。

    python -m pytest tests -q

这里测的都是"错了很难发现、但会让模型悄悄学不好"的地方：
    * 掩码方向对不对（因果掩码写成下三角是个经典 bug）
    * padding 是否真的不影响非 padding 位置
    * KV cache 增量解码与整句前向是否一致
    * 束搜索 beam=1 是否退化为贪心
    * 标签平滑的取值边界
    * 小批量能否过拟合（能过拟合说明模型+损失+反传这条链路是通的）
"""

from __future__ import annotations

import math

import pytest
import torch

from src.bleu import corpus_bleu
from src.bpe import BPE, PAD_ID
from src.config import ModelConfig, TrainConfig, tiny_config
from src.data import LengthBucketBatchSampler, make_collate_fn
from src.loss import LabelSmoothedCrossEntropy, build_criterion
from src.masks import make_causal_mask, make_pad_mask
from src.model import Transformer
from src.scheduler import NoamLR
from src.synth import generate_pairs


# ---------------------------------------------------------------------- #
# 分词器
# ---------------------------------------------------------------------- #
def test_bpe_roundtrip():
    corpus = [
        "我喜欢这本书",
        "他每天学习英语",
        "i like this book",
        "he studies english every day",
    ]
    tok = BPE.train(corpus * 20, vocab_size=300, verbose=False)

    for text in ["我喜欢这本书", "he studies english every day", "i like this book"]:
        ids = tok.encode(text)
        assert ids and all(i != tok.unk_id for i in ids), f"{text} 出现了 <unk>"
        assert tok.decode(ids) == text


def test_bpe_never_drops_base_alphabet():
    """词表很小时也必须保留每个字符，否则会退化成 <unk>。"""
    corpus = ["hello world"] * 50
    tok = BPE.train(corpus, vocab_size=8, verbose=False)
    ids = tok.encode("hello world")
    assert tok.unk_id not in ids


# ---------------------------------------------------------------------- #
# 掩码
# ---------------------------------------------------------------------- #
def test_pad_mask():
    lengths = torch.tensor([3, 5])
    mask = make_pad_mask(lengths, max_len=5)
    assert mask.shape == (2, 1, 1, 5)
    assert mask[0, 0, 0].tolist() == [False, False, False, True, True]
    assert mask[1, 0, 0].tolist() == [False] * 5


def test_causal_mask_is_upper_triangular():
    mask = make_causal_mask(4)[0, 0]
    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected)


# ---------------------------------------------------------------------- #
# 模型不变量
# ---------------------------------------------------------------------- #
def _tiny_model(vocab_size: int = 32) -> Transformer:
    cfg = tiny_config(vocab_size)
    return Transformer(cfg, pad_id=0, bos_id=1, eos_id=2)


def _pad_to_tensor(sequences: list[list[int]], pad_id: int) -> torch.Tensor:
    """把变长 id 列表补齐成 [B, T] 张量（测试里替代 DataLoader 的 collate）。"""
    max_len = max(len(s) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(sequences):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def _energize(model: Transformer, scale: float = 30.0) -> Transformer:
    """把输出层放大，让随机初始化的模型也有"鲜明"的 logits。

    这一步很关键：如果 logits 全是 1e-3 量级，那么"缓存实现错了"和"缓存实现对了"
    两种情况的差异也只在 1e-3 量级，用 atol=1e-4 的比较会**误判通过**。
    先把 logits 放大，缓存 bug 就会立刻暴露成巨大的差异。
    """
    with torch.no_grad():
        model.generator.weight.mul_(scale)
        model.generator.bias.add_(torch.randn_like(model.generator.bias) * 3.0)
    return model


def test_padding_does_not_change_encoder_output():
    """在句子后面多补几个 <pad>，前面真实位置的编码结果应完全不变。"""
    torch.manual_seed(0)
    model = _tiny_model().eval()

    ids = [5, 6, 7, 8]
    short = torch.tensor([ids])
    long = torch.tensor([ids + [PAD_ID, PAD_ID]])

    with torch.no_grad():
        out_short = model.encode(short, model.make_src_mask(short))
        out_long = model.encode(long, model.make_src_mask(long))

    assert torch.allclose(out_short, out_long[:, :4], atol=1e-5)


def test_decoder_is_causal():
    """改变第 t 个目标词，不应影响第 < t 个位置的输出。"""
    torch.manual_seed(0)
    model = _tiny_model().eval()
    src = torch.tensor([[5, 6, 7]])
    src_mask = model.make_src_mask(src)
    tgt_a = torch.tensor([[1, 9, 10, 11]])
    tgt_b = tgt_a.clone()
    tgt_b[0, -1] = 20  # 只改最后一个词

    with torch.no_grad():
        logits_a = model(src, tgt_a)
        logits_b = model(src, tgt_b)

    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_kv_cache_matches_full_forward():
    """增量解码（每步 1 个 token + 缓存）必须等价于整句一次前向。"""
    torch.manual_seed(0)
    model = _energize(_tiny_model()).eval()
    src = torch.tensor([[5, 6, 7, 8]])
    tgt = torch.tensor([[1, 9, 10, 11, 2]])
    src_mask = model.make_src_mask(src)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        full = model.decode(memory, src_mask, tgt, model.make_tgt_mask(tgt))

        caches = model._init_caches(memory)
        step_logits = []
        for t in range(tgt.size(1)):
            logits, caches = model._step(tgt[:, t : t + 1], memory, src_mask, caches, offset=t)
            step_logits.append(logits)
        incremental = torch.stack(step_logits, dim=1)  # [1, T, V]

    assert full.abs().max().item() > 1.0, "logits 太小，这个测试会失去分辨力"
    assert torch.allclose(full, incremental, atol=1e-4), "KV cache 实现与全量前向不一致"


def test_greedy_decode_uses_cache_consistently():
    """贪心解码（走 KV cache）应与"逐词重算全序列"的贪心结果完全一致。"""
    torch.manual_seed(0)
    model = _energize(_tiny_model()).eval()
    src = torch.tensor([[5, 6, 7, 8]])
    src_mask = model.make_src_mask(src)
    memory = model.encode(src, src_mask)

    with torch.no_grad():
        cached = model.greedy_decode(src, max_len=6)[0]

        # 参考实现：每步都对完整前缀重新前向一次
        tokens = [model.bos_id]
        for _ in range(6):
            tgt = torch.tensor([tokens])
            logits = model.decode(memory, src_mask, tgt, model.make_tgt_mask(tgt))
            tokens.append(int(logits[0, -1].argmax()))
        reference = tokens[1:]

    assert cached == reference


def test_beam_search_beam1_equals_greedy():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    src = torch.tensor([[5, 6, 7], [8, 9, 1]])

    greedy = model.greedy_decode(src, max_len=8)
    beam = model.beam_search_best_ids(src, beam_size=1, max_len=8)
    assert greedy == beam


def test_beam_search_returns_ranked_hypotheses():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    src = torch.tensor([[5, 6, 7, 8]])
    ranked = model.beam_search(src, beam_size=4, max_len=10)
    assert len(ranked) == 1
    assert 0 < len(ranked[0]) <= 4, "最多返回 beam_size 个候选"
    # 长度归一化后的分数应从高到低
    scores = [s / (((5 + len(ids)) / 6) ** 0.6) for ids, s in ranked[0]]
    assert scores == sorted(scores, reverse=True)


def test_beam_search_larger_beam_is_not_worse():
    """同一个（未训练的）模型上，beam 更大的搜索不应找到更差的候选。"""
    torch.manual_seed(0)
    model = _tiny_model().eval()
    src = torch.tensor([[5, 6, 7, 8]])

    def best_score(beam: int) -> float:
        ids, score = model.beam_search(src, beam_size=beam, max_len=10)[0][0]
        return score / (((5 + len(ids)) / 6) ** 0.6)

    assert best_score(4) >= best_score(1) - 1e-6


def test_tie_embeddings_shares_weight():
    cfg = tiny_config(32)
    cfg.tie_embeddings = True
    model = Transformer(cfg)
    assert model.generator.weight.data_ptr() == model.tgt_embed.embed.weight.data_ptr()


# ---------------------------------------------------------------------- #
# 损失与调度器
# ---------------------------------------------------------------------- #
def test_label_smoothing_bounds():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 10)
    target = torch.tensor([[1, 2, 3], [4, 5, 6]])

    criterion = LabelSmoothedCrossEntropy(smoothing=0.1, ignore_index=0)
    loss = criterion(logits, target)
    assert loss.item() > 0

    # 完全正确的预测：无平滑时 loss≈0，平滑后仍然 > 0（平滑惩罚过度自信）
    perfect = torch.full((1, 2, 10), -20.0)
    perfect[0, :, :] = -20.0
    for t, tok in enumerate([3, 7]):
        perfect[0, t, tok] = 20.0
    target2 = torch.tensor([[3, 7]])
    assert LabelSmoothedCrossEntropy(0.0, 0)(perfect, target2).item() < 1e-6
    assert LabelSmoothedCrossEntropy(0.1, 0)(perfect, target2).item() > 0.01


def test_padding_is_ignored_in_loss():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 10)
    target = torch.tensor([[3, 4, PAD_ID, PAD_ID]])
    base = build_criterion(0.1, PAD_ID)(logits, target).item()
    changed = logits.clone()
    changed[0, 2:] = torch.randn(2, 10) * 50  # 只改 padding 位置
    assert math.isclose(base, build_criterion(0.1, PAD_ID)(changed, target).item(), rel_tol=1e-6)


def test_noam_schedule_shape():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1.0)
    sched = NoamLR(opt, d_model=64, warmup_steps=100, scale=1.0)

    lrs = []
    for _ in range(300):
        opt.step()
        sched.step()
        lrs.append(sched.current_lr())

    peak = max(lrs)
    assert lrs[0] < peak and lrs[-1] < peak, "学习率应先升后降"
    assert abs(lrs.index(peak) + 1 - 100) <= 2, "峰值应出现在 warmup 附近"
    assert peak == pytest.approx(64 ** -0.5 * 100 ** -0.5, rel=1e-3)


# ---------------------------------------------------------------------- #
# 数据
# ---------------------------------------------------------------------- #
def test_bucket_sampler_covers_all_and_respects_budget():
    lengths = [5] * 20 + [50] * 10 + [100] * 5
    sampler = LengthBucketBatchSampler(lengths, batch_size=8, max_tokens=200, shuffle=True, seed=0)
    seen = []
    for batch in sampler:
        seen.extend(batch)
        assert max(lengths[i] for i in batch) * len(batch) <= 200 * 1.01
        assert len(batch) <= 8
    assert sorted(seen) == list(range(len(lengths)))

    sampler.set_epoch(1)
    assert set(sum(list(sampler), [])) == set(range(len(lengths)))


def test_collate_pads_correctly():
    collate = make_collate_fn(PAD_ID)
    batch = [{"src": [5, 6, 7], "tgt": [9, 2]}, {"src": [8], "tgt": [10, 11, 2, 2]}]
    out = collate(batch)
    assert out["src"].shape == (2, 3)
    assert out["tgt"].shape == (2, 4)
    assert out["src"][1].tolist() == [8, PAD_ID, PAD_ID]
    assert out["src_lengths"].tolist() == [3, 1]
    assert out["tgt_lengths"].tolist() == [2, 4]


# ---------------------------------------------------------------------- #
# 合成语料的质量（生成器改坏了要能立刻发现）
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth_pairs():
    return generate_pairs(3000, seed=123)


def test_synth_punctuation_is_aligned(synth_pairs):
    """中文以「。」结尾的句子，英文必须对应「 .」；「？」对应「 ?」。"""
    for zh, en in synth_pairs:
        if zh.endswith("？"):
            assert en.endswith(" ?"), (zh, en)
        else:
            assert zh.endswith("。") and en.endswith(" ."), (zh, en)
        assert "，" not in en and "？" not in en and "。" not in en


def test_synth_uncountable_nouns_have_no_article(synth_pairs):
    """中文/英语/时间/帮助 这类不可数名词前面不能出现 a/the。"""
    bad = [" the chinese", " the english", " the time", " the help",
           " a chinese", " a english", " a time", " a help"]
    for zh, en in synth_pairs:
        for pattern in bad:
            assert f"{pattern} " not in f" {en} ", (zh, en)


def test_synth_numeral_and_plural_agreement(synth_pairs):
    """「三本书」必须译成 three books；「一本书」必须带冠词 a/an。"""
    numerals = {"两": "two", "三": "three", "四": "four", "五": "five",
                "六": "six", "七": "seven", "八": "eight", "九": "nine", "十": "ten"}
    measures = "本台部张杯个件辆只份座位封把场"
    for zh, en in synth_pairs:
        for zh_num, en_num in numerals.items():
            if f"{zh_num}本" in zh or any(f"{zh_num}{m}" in zh for m in measures):
                assert en_num in en, (zh, en)
        if any(f"一{m}" in zh for m in measures):
            assert " a " in f" {en} " or " an " in f" {en} ", (zh, en)


def test_synth_pronoun_substitution_in_contrast_sentences(synth_pairs):
    """「…，但是他不喜欢。」里的宾语被省略，英文要用 it / them 补出来。"""
    found = 0
    for zh, en in synth_pairs:
        if "但是" in zh:
            found += 1
            assert " it " in f" {en} " or " them " in f" {en} ", (zh, en)
    assert found > 50, "对比句样本太少，测试失去意义"


# ---------------------------------------------------------------------- #
# BLEU
# ---------------------------------------------------------------------- #
def test_bleu_perfect_and_degenerate():
    refs = ["i like this book .", "he studies english every day ."]
    assert corpus_bleu(refs, refs) == pytest.approx(100.0, abs=1e-6)
    assert corpus_bleu(["完全不对的译文 ."], ["i like this book ."]) < 5.0
    assert corpus_bleu([""], ["something"]) == 0.0


def test_bleu_penalizes_short_output():
    ref = ["the cat sat on the mat and looked around ."]
    short = corpus_bleu(["the cat ."], ref)
    long_ok = corpus_bleu(ref, ref)
    assert short < long_ok


# ---------------------------------------------------------------------- #
# 端到端：能不能过拟合一个极小数据集
# ---------------------------------------------------------------------- #
def test_model_can_overfit_tiny_batch():
    """8 句语料上训练 120 步，loss 应显著下降（否则说明某处链路是坏的）。"""
    torch.manual_seed(0)
    pairs = generate_pairs(8, seed=7)
    tok = BPE.train([p[0] for p in pairs] + [p[1] for p in pairs], vocab_size=300, verbose=False)

    src = _pad_to_tensor([tok.encode(p[0]) for p in pairs], tok.pad_id)
    tgt = _pad_to_tensor([tok.encode(p[1], add_eos=True) for p in pairs], tok.pad_id)

    cfg = ModelConfig(
        src_vocab_size=len(tok), tgt_vocab_size=len(tok),
        d_model=64, n_heads=4, num_encoder_layers=2, num_decoder_layers=2,
        d_ff=128, dropout=0.0, max_len=64,
    )
    model = Transformer(cfg, pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.98), eps=1e-9)
    criterion = build_criterion(0.1, tok.pad_id)

    losses = []
    for _ in range(120):
        tgt_in = model.shift_right(tgt, tok.bos_id)
        logits = model(src, tgt_in)
        loss = criterion(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, f"loss 没有明显下降: {losses[0]:.3f} -> {losses[-1]:.3f}"

    # 至少有一半的句子能被贪心解码完全复现（小模型不必强求 100%）
    model.eval()
    generated = model.greedy_decode(src, max_len=tgt.size(1))
    expected = [row[row != tok.pad_id].tolist() for row in tgt]
    exact = sum(1 for g, e in zip(generated, expected) if g == e)
    assert exact >= len(expected) // 2, f"只有 {exact}/{len(expected)} 句被复现"

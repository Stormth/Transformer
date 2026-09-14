"""配置与论文预设的测试：论文那些关键数字不能被手滑改掉。"""

from __future__ import annotations

from nmt.config import PRESETS, build_config, paper_base_config, paper_big_config


def test_paper_base_matches_the_paper() -> None:
    """《Attention Is All You Need》base 的关键超参。"""

    config = paper_base_config()
    model = config.model
    assert (model.d_model, model.n_heads) == (512, 8)
    assert (model.num_encoder_layers, model.num_decoder_layers) == (6, 6)
    assert model.d_ff == 2048
    assert model.dropout == 0.1
    assert model.share_vocab and model.tie_embeddings     # 论文用共享词表

    train = config.train
    assert train.warmup_steps == 4000
    assert train.label_smoothing == 0.1
    assert train.max_tokens == 50000                      # 25k 源 + 25k 目标
    assert train.keep_last_checkpoints == 5               # 论文平均最后 5 个
    assert train.bleu_lowercase is False                  # 对齐 sacrebleu 的 WMT14 口径


def test_paper_base_peak_learning_rate_matches_the_paper() -> None:
    """论文 base 的峰值学习率是 7e-4，这是从 Noam 公式直接算出来的。"""

    from nmt.scheduler import NoamScheduler
    import torch

    config = paper_base_config()
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = NoamScheduler(
        optimizer, config.model.d_model, config.train.warmup_steps, config.train.lr_scale
    )
    peak = scheduler.lr_at(config.train.warmup_steps)
    assert abs(peak - 7.0e-4) < 1e-5


def test_paper_big_is_three_times_wider() -> None:
    config = paper_big_config()
    assert (config.model.d_model, config.model.n_heads) == (1024, 16)
    assert config.model.d_ff == 4096
    assert config.model.dropout == 0.3      # 论文 big 用的 dropout


def test_all_presets_are_buildable() -> None:
    for name in PRESETS:
        config = build_config(name)
        assert config.model.d_model % config.model.n_heads == 0


def test_override_can_turn_off_lowercase_bleu() -> None:
    config = build_config("base", overrides={"train.bleu_lowercase": False})
    assert config.train.bleu_lowercase is False

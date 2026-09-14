"""checkpoint 平均的测试（论文报的分数就是最后 5 个 checkpoint 的平均）。"""

from __future__ import annotations

import torch

from nmt.average import average_checkpoints, find_snapshots
from nmt.checkpoint import load_checkpoint, save_checkpoint
from nmt.config import tiny_config
from nmt.model import Transformer


def _save(path, model, config, seed: int) -> None:
    torch.manual_seed(seed)
    for param in model.parameters():
        param.data.normal_(0, 0.1)
    save_checkpoint(path, model=model, config=config, epoch=1, save_optimizer=False)


def test_average_of_two_checkpoints_is_the_mean(tmp_path) -> None:
    config = tiny_config(vocab_size=64)
    model_a, model_b = Transformer(config.model), Transformer(config.model)
    path_a, path_b = tmp_path / "epoch_0001.pt", tmp_path / "epoch_0002.pt"
    _save(path_a, model_a, config, seed=1)
    _save(path_b, model_b, config, seed=2)

    weights_a = {k: v.clone() for k, v in load_checkpoint(path_a).model.items()}
    weights_b = {k: v.clone() for k, v in load_checkpoint(path_b).model.items()}

    averaged = average_checkpoints([path_a, path_b], output=tmp_path / "avg.pt")["model"]
    for key in weights_a:
        expected = (weights_a[key] + weights_b[key]) / 2
        assert torch.allclose(averaged[key], expected, atol=1e-6), key

    # 平均出来的文件必须能当普通 checkpoint 读
    reloaded = load_checkpoint(tmp_path / "avg.pt")
    assert reloaded.config == config.to_dict()


def test_find_snapshots_sorts_numerically(tmp_path) -> None:
    """字符串排序会把 epoch_10 排在 epoch_2 前面，必须按数字排。"""

    for name in ("epoch_0001.pt", "epoch_0002.pt", "epoch_0010.pt"):
        (tmp_path / name).write_bytes(b"x")
    names = [p.name for p in find_snapshots(tmp_path)]
    assert names == ["epoch_0001.pt", "epoch_0002.pt", "epoch_0010.pt"]

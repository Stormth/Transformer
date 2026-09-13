"""pytest 公共夹具。

所有测试都刻意不联网、不依赖已下载的数据：
需要语料的地方现场造几十句假数据，这样在笔记本上跑 pytest 也能秒级完成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# 让 `python -m pytest tests` 能直接 import nmt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nmt.bpe import BPE  # noqa: E402
from nmt.config import tiny_config  # noqa: E402
from nmt.model import Transformer  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer() -> BPE:
    """在一个小的合成语料上训一个 BPE，整个测试会话共用。"""

    from nmt.synth import toy_pairs

    texts = []
    for source, target in toy_pairs(120):
        texts.extend([source, target])
    return BPE.train(texts, vocab_size=500, min_frequency=1)


@pytest.fixture()
def model(tokenizer: BPE) -> Transformer:
    torch.manual_seed(0)
    config = tiny_config(vocab_size=len(tokenizer))
    return Transformer(config.model).eval()


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")

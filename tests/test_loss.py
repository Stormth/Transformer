"""损失函数测试：标签平滑的数值要和手算一致，padding 不能参与。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from nmt.loss import LabelSmoothingLoss, PlainCrossEntropy, build_criterion


def test_smoothing_zero_equals_cross_entropy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 10)
    target = torch.tensor([[1, 2, 3], [4, 0, 0]])
    ours = build_criterion(pad_id=0, vocab_size=10, smoothing=0.0)
    reference = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1), ignore_index=0)
    assert torch.allclose(ours(logits, target), reference, atol=1e-6)


def test_label_smoothing_matches_manual_computation() -> None:
    logits = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])   # [1, 1, 4]
    target = torch.tensor([[0]])
    loss = LabelSmoothingLoss(vocab_size=4, pad_id=3, smoothing=0.2)(logits, target)

    log_probs = F.log_softmax(logits[0, 0], dim=-1)
    distribution = torch.tensor([0.8, 0.2 / 3, 0.2 / 3, 0.2 / 3])
    expected = -(distribution * log_probs).sum()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_padding_positions_are_ignored() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 6)
    target = torch.tensor([[1, 2, 0, 0]])          # 后两个是 <pad>
    criterion = LabelSmoothingLoss(vocab_size=6, pad_id=0, smoothing=0.1)
    loss_short = criterion(logits[:, :2], target[:, :2])
    loss_long = criterion(logits, target)
    assert torch.allclose(loss_short, loss_long, atol=1e-6)


def test_loss_is_never_zero_with_smoothing() -> None:
    """标签平滑让正确答案只占 0.9，所以损失有下界。

    均匀分布时损失就是 -log(1/V) = log(V)，这里 V=5。
    """

    logits = torch.full((1, 1, 5), 100.0)
    target = torch.tensor([[0]])
    loss = LabelSmoothingLoss(vocab_size=5, pad_id=4, smoothing=0.1)(logits, target)
    assert abs(loss.item() - math.log(5)) < 1e-5


def test_smoothing_has_a_positive_floor() -> None:
    """把正确答案的概率推到极限时，平滑项仍然会产生损失：模型永远不会"太自信"。"""

    logits = torch.zeros(1, 1, 11)
    logits[0, 0, 0] = 100.0
    target = torch.tensor([[0]])
    loss = LabelSmoothingLoss(vocab_size=11, pad_id=10, smoothing=0.1)(logits, target)

    log_probs = F.log_softmax(logits[0, 0], dim=-1)
    distribution = torch.full((11,), 0.1 / 10)
    distribution[0] = 0.9
    expected = -(distribution * log_probs).sum()
    assert torch.allclose(loss, expected, atol=1e-5)
    assert loss.item() > 0.1


def test_chunked_loss_equals_single_pass() -> None:
    """分块计算必须和一次性计算逐位一致 —— 分块只是为了省显存，不该改变数值。"""

    torch.manual_seed(0)
    logits = torch.randn(2, 7, 13)
    target = torch.randint(0, 13, (2, 7))
    target[0, 5:] = 0      # 混入 padding

    whole = LabelSmoothingLoss(vocab_size=13, pad_id=0, smoothing=0.1, chunk_tokens=1024)
    chunked = LabelSmoothingLoss(vocab_size=13, pad_id=0, smoothing=0.1, chunk_tokens=3)
    assert torch.allclose(whole(logits, target), chunked(logits, target), atol=1e-6)

    plain_whole = PlainCrossEntropy(pad_id=0, chunk_tokens=1024)
    plain_chunked = PlainCrossEntropy(pad_id=0, chunk_tokens=2)
    assert torch.allclose(plain_whole(logits, target), plain_chunked(logits, target), atol=1e-6)


def test_chunked_loss_gradients_flow() -> None:
    logits = torch.randn(1, 6, 11, requires_grad=True)
    target = torch.randint(0, 11, (1, 6))
    LabelSmoothingLoss(vocab_size=11, pad_id=0, smoothing=0.1, chunk_tokens=2)(logits, target).backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0

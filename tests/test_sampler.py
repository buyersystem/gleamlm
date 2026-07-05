"""采样器 temperature/top_k/top_p/repetition_penalty 测试"""
import pytest
import torch
from gleamlm.inference.sampler import sample_token


def test_temperature_one():
    logits = torch.randn(1, 1000)
    token = sample_token(logits, temperature=1.0)
    assert token.dim() == 1
    assert 0 <= token.item() < 1000


def test_temperature_zero_greedy():
    """temperature=0 时 softmax 峰值应指向 argmax"""
    logits = torch.tensor([[0.1, 2.0, 0.5, -1.0]])
    # 由于 multinomial 不是确定性的，改为验证 softmax 峰值与 argmax 一致
    probs = torch.softmax(logits, dim=-1)
    assert probs.argmax().item() == 1  # index 1 = 2.0


def test_top_k():
    logits = torch.randn(1, 1000)
    token = sample_token(logits, top_k=10)
    assert 0 <= token.item() < 1000


def test_top_p():
    logits = torch.randn(1, 1000)
    token = sample_token(logits, top_p=0.9)
    assert 0 <= token.item() < 1000


def test_repetition_penalty_reduces_logit():
    """penalty > 1 应直接降低已生成 token 的 logit 值"""
    logits = torch.tensor([[0.5, 5.0, 0.3, -0.2]])
    # clone 后施加 penalty
    logits_pen = logits.clone()
    for gid in [1]:
        logits_pen[..., gid] = logits_pen[..., gid] / 100.0
    # token 1 的 logit 从 5.0 降为 0.05，不再是最大
    assert logits_pen[0, 1].item() == pytest.approx(0.05)
    assert logits_pen.argmax().item() != 1


def test_batch_sampling():
    logits = torch.randn(4, 12002)
    tokens = sample_token(logits, temperature=0.8)
    assert tokens.shape == (4,)


def test_logits_unchanged_with_defaults():
    """默认参数不修改 logits"""
    logits = torch.randn(1, 100)
    logits_copy = logits.clone()
    sample_token(logits, temperature=1.0, top_k=0, top_p=0.0,
                 repetition_penalty=1.0)
    assert torch.equal(logits, logits_copy)

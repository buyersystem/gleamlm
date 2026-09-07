"""MambaHybridModel 块级混合解码器测试"""

import pytest
import torch

from gleamlm.models.mamba_hybrid import MambaBlock, MambaHybridModel
from gleamlm.models.model import DecoderLayer, GleamLMModel

VOCAB_SIZE = 12002


def make_hybrid(max_seq_len: int = 64, **kwargs) -> MambaHybridModel:
    """小尺寸混合模型: 4 层 × 64 维, pattern/max_seq_len 可覆写"""
    return MambaHybridModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        d_ff=128,
        max_seq_len=max_seq_len,
        dropout=0.0,
        pad_token_id=0,
        **kwargs,
    )


# pattern 展开与校验


def test_pattern_expansion():
    model = make_hybrid(pattern="amm")
    # 4 层 = "amm" 平铺取前 4 个: [a, m, m, a]
    assert model.block_kinds == ["a", "m", "m", "a"]
    assert model.num_attn_blocks == 2
    assert model.num_mamba_blocks == 2
    assert isinstance(model.layers[0], DecoderLayer)
    assert isinstance(model.layers[1], MambaBlock)
    assert isinstance(model.layers[3], DecoderLayer)


def test_invalid_pattern():
    with pytest.raises(ValueError, match="pattern"):
        make_hybrid(pattern="")
    with pytest.raises(ValueError, match="pattern"):
        make_hybrid(pattern="axm")


# 前向契约


def test_forward_shapes_and_kv_placeholders():
    model = make_hybrid(pattern="ammm")
    model.eval()
    x = torch.randint(1, 1000, (2, 12))
    logits, kv_list, aux, hidden = model(x)
    assert logits.shape == (2, 12, VOCAB_SIZE)
    assert len(kv_list) == 4
    # attention 层 (idx 0) 产 (K, V); kv cache 存未 expand 的 K/V (num_kv_heads=2)
    assert kv_list[0] is not None
    k, v = kv_list[0]
    assert k.shape == (2, 2, 12, 16)  # num_kv_heads × head_dim
    assert v.shape == (2, 2, 12, 16)
    for i in (1, 2, 3):
        assert kv_list[i] is None
    assert aux.item() == 0.0  # MLP 无 aux_loss
    assert hidden is None


def test_output_hidden_states():
    model = make_hybrid(pattern="a")
    model.eval()
    x = torch.randint(1, 1000, (2, 12))
    _, _, _, hidden = model(x, output_hidden_states=True)
    assert hidden.shape == (2, 12, 64)


def test_use_cache_false_empties_kv_list():
    model = make_hybrid(pattern="ammm")
    model.eval()
    x = torch.randint(1, 1000, (2, 12))
    _, kv_list, _, _ = model(x, use_cache=False)
    assert kv_list == []


def test_incremental_generation_not_implemented():
    model = make_hybrid(pattern="ammm")
    x = torch.randint(1, 1000, (2, 12))
    dummy_kv = [((torch.randn(2, 4, 1, 16), torch.randn(2, 4, 1, 16)),) for _ in range(4)]
    with pytest.raises(NotImplementedError):
        model(x, past_kv_list=dummy_kv)


def test_seq_len_exceeds_rope_cache():
    model = make_hybrid(pattern="a", max_seq_len=16, rope_factor=1.0)
    model.eval()
    x = torch.randint(1, 1000, (1, 32))
    with pytest.raises(ValueError, match="RoPE"):
        model(x)


# 与主模型的一致性


def test_all_attention_pattern_matches_gleam_model():
    """pattern='a' 时应与同参数 GleamLMModel 逐位一致 (架构等价回归)。"""
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    ref = GleamLMModel(**kwargs)
    torch.manual_seed(0)
    hyb = make_hybrid(pattern="a")
    x = torch.randint(1, 1000, (2, 12))
    ref.eval()
    hyb.eval()
    logits_ref, kv_ref, aux_ref, _ = ref(x)
    logits_hyb, kv_hyb, aux_hyb, _ = hyb(x)
    assert torch.equal(logits_ref, logits_hyb)
    assert torch.equal(kv_ref[0][0], kv_hyb[0][0])
    assert aux_ref.item() == aux_hyb.item()


# 训练路径


def test_backward_reaches_both_block_types():
    model = make_hybrid(pattern="ammm")
    x = torch.randint(1, 1000, (2, 12))
    model.train()
    logits, _, _, _ = model(x)
    logits.float().pow(2).mean().backward()
    # attention 块 (DecoderLayer.GQA.W_q) 与 mamba 块 (in_proj) 都收到梯度
    assert model.layers[0].attn.W_q.weight.grad is not None
    assert model.layers[1].in_proj.weight.grad is not None
    assert torch.isfinite(model.layers[1].in_proj.weight.grad).all()


def test_gradient_checkpointing_logits_and_grads_match():
    """开/关 gradient checkpointing 的混合模型前向与反向应完全一致。"""
    model = make_hybrid(pattern="ammm")
    x = torch.randint(1, 1000, (2, 12))

    def run(cp: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        model.train()
        model.use_gradient_checkpointing = cp
        logits, _, _, _ = model(x)
        logits.float().pow(2).mean().backward()
        grads = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        return logits, grads

    logits_cp, grads_cp = run(True)
    logits_plain, grads_plain = run(False)
    assert torch.allclose(logits_cp, logits_plain)
    for g_cp, g_plain in zip(grads_cp, grads_plain, strict=True):
        assert torch.allclose(g_cp, g_plain)


# 纯 mamba 对照 (pattern='m')


def test_pure_mamba_pattern():
    model = make_hybrid(pattern="m")
    model.eval()
    x = torch.randint(1, 1000, (2, 12))
    logits, kv_list, aux, _ = model(x)
    assert logits.shape == (2, 12, VOCAB_SIZE)
    assert kv_list == [None, None, None, None]
    assert aux.item() == 0.0


def test_num_params_reported():
    model = make_hybrid(pattern="ammm")
    total, with_buffers = model.get_num_params()
    assert total > 0
    assert with_buffers >= total

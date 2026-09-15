"""hybrid_variants 装配件单测: partial RoPE / 输出门控 / 混合层布局 / 端到端装配。

零侵入验证: 组件仅显式装配时生效, 不涉及现有模型默认路径。
"""

import math
from typing import cast

import pytest
import torch

from gleamlm.models.attention_variants import SlidingWindowGQA
from gleamlm.models.hybrid_variants import (
    OutputGateGQA,
    OutputGateSlidingWindowGQA,
    apply_partial_rope,
    build_hybrid_layer_configs,
    compute_partial_rope_cache,
)
from gleamlm.models.model import GQA, DecoderLayer, GleamLMModel, apply_rope, precompute_freqs_cis


class TestPartialRopeCache:
    """缓存构建: rot_dim 偶数化 / 锁定区 cos=1/sin=0 / 频率分母用 rot_dim。"""

    def test_locked_dimensions(self):
        cos, sin = compute_partial_rope_cache(128, 8, base=10000.0, rotary_percent=0.334)
        # rot_dim=42 → 21 对; 锁定区 = 21..63 与 85..127
        assert cos.shape == (8, 128) and sin.shape == (8, 128)
        assert torch.all(cos[:, 21:64] == 1.0) and torch.all(sin[:, 21:64] == 0.0)
        assert torch.all(cos[:, 85:] == 1.0) and torch.all(sin[:, 85:] == 0.0)

    def test_frequency_matches_rot_dim_denominator(self):
        cos, _ = compute_partial_rope_cache(128, 4, base=10000.0, rotary_percent=0.334)
        for j in (0, 5, 20):
            expected = math.cos(1.0 * 10000.0 ** (-2.0 * j / 42.0))
            assert cos[1, j].item() == pytest.approx(expected, rel=1e-5)

    def test_percent_one_matches_standard_rope(self):
        cos_p, sin_p = compute_partial_rope_cache(64, 32, base=10000.0, rotary_percent=1.0)
        cos_m, sin_m = precompute_freqs_cis(64, 32, base=10000.0)
        assert torch.allclose(cos_p, cos_m) and torch.allclose(sin_p, sin_m)

    def test_invalid_percent_raises(self):
        with pytest.raises(ValueError):
            compute_partial_rope_cache(64, 8, rotary_percent=0.0)
        with pytest.raises(ValueError):
            compute_partial_rope_cache(64, 8, rotary_percent=0.01)  # 不足一对旋转维


class TestPartialRopeApply:
    """apply_rope 数值: 锁定维原样直通, 旋转对按 (c, s) 旋转。"""

    def test_locked_dims_passthrough(self):
        cos, sin = compute_partial_rope_cache(128, 8, base=10000.0, rotary_percent=0.25)
        q = torch.randn(1, 1, 8, 128)
        q_out, _ = apply_rope(q, q.clone(), cos, sin)
        # rot_dim=32 → 16 对; 锁定 16..63 与 80..127 原样
        assert torch.equal(q_out[..., 16:64], q[..., 16:64])
        assert torch.equal(q_out[..., 80:], q[..., 80:])

    def test_rotary_pair_values(self):
        cos, sin = compute_partial_rope_cache(128, 8, base=10000.0, rotary_percent=0.25)
        q = torch.randn(1, 1, 8, 128)
        q_out, _ = apply_rope(q, q.clone(), cos, sin)
        j = 3
        # cos/sin 按行对应各时刻, 先广播成 [1, 1, S, 1] 再与 (j, j+64) 对比较
        c, s = cos[:, j].view(1, 1, -1), sin[:, j].view(1, 1, -1)
        expected_lo = q[..., j] * c - q[..., j + 64] * s
        expected_hi = q[..., j + 64] * c + q[..., j] * s
        assert torch.allclose(q_out[..., j], expected_lo, atol=1e-6)
        assert torch.allclose(q_out[..., j + 64], expected_hi, atol=1e-6)


class TestApplyPartialRope:
    """装配: 替换模型缓存 / 装配后前向可跑 / 外推组合拒绝。"""

    @staticmethod
    def _tiny_model(rope_scale: float = 1.0) -> GleamLMModel:
        return GleamLMModel(
            vocab_size=64,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            d_ff=64,
            max_seq_len=16,
            rope_scale=rope_scale,
        )

    def test_replaces_cache(self):
        model = self._tiny_model()
        apply_partial_rope(model, rotary_percent=0.5)
        cos, sin = compute_partial_rope_cache(
            8, model.rope_max_len, base=model.rope_theta, rotary_percent=0.5
        )
        assert torch.equal(model.rope_cos, cos) and torch.equal(model.rope_sin, sin)
        # head_dim=8 → rot_dim=4 → 锁定 2..3 与 6..7
        assert torch.all(model.rope_cos[:, 2:4] == 1.0)  # type: ignore[attr-defined]

    def test_forward_runs_after_apply(self):
        model = self._tiny_model()
        apply_partial_rope(model, rotary_percent=0.5)
        logits, _, _, _ = model(torch.randint(0, 64, (2, 16)))
        assert logits.shape == (2, 16, 64)

    def test_rejects_extrapolation(self):
        model = self._tiny_model(rope_scale=2.0)
        with pytest.raises(ValueError):
            apply_partial_rope(model, rotary_percent=0.5)


class TestOutputGate:
    """门控: σ(W_gate(x)) 语义 / 梯度回传 / 窗口层组合前向。"""

    def test_gate_halves_output_when_zero(self):
        attn = OutputGateGQA(32, 4, 2, dropout=0.0)
        attn.eval()
        with torch.no_grad():
            attn.W_gate.weight.zero_()
        x = torch.randn(2, 5, 32)
        cos, sin = torch.ones(5, 8), torch.zeros(5, 8)
        out, _, _ = attn(x, cos, sin)
        inner, _, _ = GQA.forward(attn, x, cos, sin)
        assert torch.allclose(out, 0.5 * inner, atol=1e-6)

    def test_gate_grad_flows(self):
        attn = OutputGateGQA(32, 4, 2, dropout=0.0)
        x = torch.randn(2, 5, 32)
        cos, sin = torch.ones(5, 8), torch.zeros(5, 8)
        out, _, _ = attn(x, cos, sin)
        out.sum().backward()
        assert attn.W_gate.weight.grad is not None
        assert attn.W_gate.weight.grad.abs().sum() > 0

    def test_swa_gate_forward_with_mask(self):
        attn = OutputGateSlidingWindowGQA(32, 4, 2, dropout=0.0, window_size=3)
        attn.eval()
        x = torch.randn(1, 6, 32)
        cos, sin = torch.ones(6, 8), torch.zeros(6, 8)
        mask = torch.triu(torch.full((6, 6), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)
        out, _, _ = attn(x, cos, sin, mask)
        assert out.shape == (1, 6, 32) and torch.isfinite(out).all()


class TestBuildHybridLayerConfigs:
    """布局: 1-based 整除判定 / 27+5 格局 / 门控开关。"""

    def test_pattern_32_layers(self):
        configs = build_hybrid_layer_configs(32, global_every=6, window_size=127)
        assert len(configs) == 32
        global_idx = [i for i, c in enumerate(configs) if c["attn_variant"] is GQA]
        assert global_idx == [5, 11, 17, 23, 29]
        swa = [c for c in configs if c["attn_variant"] is OutputGateSlidingWindowGQA]
        assert len(swa) == 27
        assert all(c["window_size"] == 127 for c in swa)

    def test_gate_switch(self):
        configs = build_hybrid_layer_configs(4, global_every=6, gate_swa=False)
        assert all(c["attn_variant"] is SlidingWindowGQA for c in configs)

    def test_no_global_when_every_exceeds_layers(self):
        configs = build_hybrid_layer_configs(3, global_every=64)
        assert all(c["attn_variant"] is OutputGateSlidingWindowGQA for c in configs)


class TestHybridAssembly:
    """端到端: 布局 + partial RoPE 装配后前向/反向跑通, 门控参与反传。"""

    def test_small_model_end_to_end(self):
        configs = build_hybrid_layer_configs(num_layers=4, global_every=2, window_size=4)
        model = GleamLMModel(
            vocab_size=100,
            d_model=64,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            d_ff=128,
            max_seq_len=32,
            layer_configs=configs,
        )
        apply_partial_rope(model, rotary_percent=0.5)

        layer0 = cast(DecoderLayer, model.layers[0])
        layer1 = cast(DecoderLayer, model.layers[1])
        assert isinstance(layer0.attn, OutputGateSlidingWindowGQA)
        assert type(layer1.attn) is GQA  # 全局层原生无门控

        logits, _, _, _ = model(torch.randint(0, 100, (2, 16)))
        assert logits.shape == (2, 16, 100)
        logits.float().sum().backward()
        gate = cast(OutputGateSlidingWindowGQA, layer0.attn).W_gate
        assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0

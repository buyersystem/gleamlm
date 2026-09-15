"""混合注意力装配套件: 滑窗/全局层布局、输出门控、partial RoPE。

零侵入: 不改动现有模型的默认行为——布局经 layer_configs 显式注入,
partial RoPE 经 apply_partial_rope 显式替换缓存, 门控变体显式选用。

装配::

    from gleamlm.models.hybrid_variants import (
        apply_partial_rope, build_hybrid_layer_configs)

    configs = build_hybrid_layer_configs(num_layers=32, global_every=6, window_size=127)
    model = GleamLMModel(..., layer_configs=configs)
    apply_partial_rope(model, rotary_percent=0.334)
"""

from typing import Any

import torch
from torch import nn

from gleamlm.models.attention_variants import SlidingWindowGQA
from gleamlm.models.model import GQA, GleamLMModel
from gleamlm.types import PastKeyValue


def compute_partial_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    rotary_percent: float = 1.0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """partial RoPE 的 cos/sin 缓存, 形状与全维缓存一致 [max_seq_len, head_dim]。

    前 rot_dim=int(head_dim×percent) 维 (向下偶数化, 旋转按维度对进行) 旋转,
    其余维度锁定: 频率置 0 → cos=1/sin=0 → apply_rope 输出等于输入。
    旋转段的频率分母用 rot_dim (percent<1 时与全维频率值不同);
    percent=1.0 时与标准 RoPE 数值一致。
    """
    if not 0.0 < rotary_percent <= 1.0:
        raise ValueError(f"rotary_percent 需在 (0, 1] 内, 实际 {rotary_percent}")
    rot_dim = (int(head_dim * rotary_percent) // 2) * 2
    if rot_dim == 0:
        raise ValueError(f"rotary_percent={rotary_percent} 在 head_dim={head_dim} 上不足一对旋转维")

    half = head_dim // 2
    n_pairs = rot_dim // 2
    freq = 1.0 / (base ** (torch.arange(n_pairs, dtype=torch.float) * 2.0 / rot_dim))
    if n_pairs < half:
        freq = torch.cat([freq, torch.zeros(half - n_pairs, dtype=torch.float)])

    t = torch.arange(max_seq_len, dtype=torch.float) / scale
    emb = torch.outer(t, freq)
    emb = torch.cat([emb, emb], dim=-1)  # 每对维度共享同一频率 (half-split 布局)
    return emb.cos(), emb.sin()


def apply_partial_rope(model: GleamLMModel, rotary_percent: float) -> None:
    """把 partial RoPE 装配到已构建的模型: 按模型既有 RoPE 参数重建 cos/sin 缓存。

    与长上下文外推 (rope_scale > 1) 的组合未定义, 显式拒绝而非静默走错路径。
    """
    if model.rope_scale > 1.0:
        raise ValueError("partial RoPE 与线性缩放/YaRN 外推 (rope_scale>1) 的组合未实现")
    cos, sin = compute_partial_rope_cache(
        model.head_dim,
        model.rope_max_len,
        base=model.rope_theta,
        rotary_percent=rotary_percent,
    )
    device = next(model.parameters()).device
    model.register_buffer("rope_cos", cos.to(device), persistent=False)
    model.register_buffer("rope_sin", sin.to(device), persistent=False)


def _gated_output(gate_proj: nn.Linear, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out × sigmoid(W_gate(x)): 门控在 fp32 计算后回投输出 dtype。"""
    return out * torch.sigmoid(gate_proj(x).float()).to(out.dtype)


class OutputGateGQA(GQA):
    """GQA + 输出门控 (全局层的带门控变体)。门控与 QKV 并行投影, 常规线性初始化。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(d_model, num_heads, num_kv_heads, dropout, use_flash_attn, **kwargs)
        self.W_gate = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: PastKeyValue | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PastKeyValue]:
        output, attn_weights, current_kv = super().forward(x, rope_cos, rope_sin, mask, past_kv)
        return _gated_output(self.W_gate, x, output), attn_weights, current_kv


class OutputGateSlidingWindowGQA(SlidingWindowGQA):
    """滑动窗口 GQA + 输出门控 (窗口层的带门控变体)。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = False,
        window_size: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            window_size=window_size,
            **kwargs,
        )
        self.W_gate = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: PastKeyValue | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PastKeyValue]:
        output, attn_weights, current_kv = super().forward(x, rope_cos, rope_sin, mask, past_kv)
        return _gated_output(self.W_gate, x, output), attn_weights, current_kv


def build_hybrid_layer_configs(
    num_layers: int,
    global_every: int = 6,
    window_size: int = 127,
    gate_swa: bool = True,
) -> list[dict]:
    """滑窗/全局交替的 layer_configs: 层号 (1-based) 整除 global_every 的层为全局层。

    - 全局层: 原生 GQA, 不加窗、不挂门控
    - 窗口层: 滑动窗口 GQA, window_size=127 时窗口 128 (含自身);
      gate_swa 开启时挂输出门控 (门控只装窗口层)
    """
    swa_variant: type = OutputGateSlidingWindowGQA if gate_swa else SlidingWindowGQA
    configs: list[dict] = []
    for layer_idx in range(num_layers):
        if global_every > 0 and (layer_idx + 1) % global_every == 0:
            configs.append({"attn_variant": GQA})
        else:
            configs.append({"attn_variant": swa_variant, "window_size": window_size})
    return configs

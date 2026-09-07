"""Mamba-1 教学块与 Mamba × GQA 块级混合解码器 — 与 GleamLMModel 并列的实验入口(不改动 model.py)。

本文件自含两部分, 单文件即可做纯 SSM → 块级混合的对照:

1) Mamba-1 教学块 (selective_scan + MambaBlock):
   用 SSM 替代 attention, 推理每 token O(1)(只维护状态 h_t)。
   选择性扫描让 A/B/C/Δ 依赖输入, 相当于给 SSM 加可学习的门控。

2) MambaHybridModel (Jamba 式块级混合):
   动机: 纯 attention 精确检索强但注意力矩阵平方复杂度; 纯 SSM 线性复杂度
   但隐状态是有损压缩, 精确回溯弱。Jamba / Nemotron-H / Zamba2 等实证:
   按块交替堆叠 attention 与 SSM, 可在长序列成本与精确任务能力之间取平衡。

   - attention 块 = model.DecoderLayer (PreNorm GQA + PreNorm SwiGLU, 可换 MoE)
   - mamba 块   = 本文件的 MambaBlock (conv1d + SiLU gate + selective SSM)
   - pattern 字符串逐字符平铺 num_layers: 'a' = attention 块, 'm' = mamba 块。
     Jamba 1:7 风格 → "ammmmmmm"; 全 attention → "a"(≈GleamLMModel 对照);
     全 mamba → "m"(纯 SSM 对照)。

统一块契约: 两种块都是"完整层"(自带 norm + 残差), 可直接按 pattern 交替
堆叠; forward 签名一致 (x, rope_cos, rope_sin, mask, past_kv), 返回三元组
(out, kv_or_None, aux_or_None) — DecoderLayer 返回 (x, current_kv, aux),
MambaBlock 返回 (out, None, None)。模型循环据此统一汇总 aux_loss (MoE) 与
kv (attention 层), 无需按块类型分叉。

已知限制 (教学/实验定位):
- MambaBlock 用显式 for 循环 scan, 长序列训练慢 — 并行 scan 属内核工作。
- 无增量生成: mamba 需要 SSM 隐状态 h 而非 KV cache, 混合缓存
  (attention KV + mamba state) 尚未实现; past_kv_list 传入即 NotImplementedError。
- attention_mask 的 padding 只对 attention 块生效: mamba 块的 conv/scan 天然
  因果、无 mask 概念, pad 位置仍会被计算 — 请使用长度一致的样本 (教学简化)。

用法示例:
    model = MambaHybridModel(
        vocab_size=12002, d_model=768, num_layers=24,
        num_heads=12, num_kv_heads=6, d_ff=2048, max_seq_len=4096,
        pattern="ammmmmmm",   # 24 层 → [a,m,m,m,m,m,m,m] × 3
    )
    logits, kv_list, aux, hidden = model(input_ids)  # 返回签名同 GleamLMModel
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from gleamlm.models.model import GQA, MLP, DecoderLayer, RMSNorm, precompute_freqs_cis
from gleamlm.types import PastKeyValue, PastKeyValueList


# Mamba: 用 SSM 替代 attention, 推理每 token O(1)(只维护状态 h_t)。
# 选择性扫描让 A/B/C/Δ 依赖输入, 相当于给 SSM 加可学习的门控。
def selective_scan(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """选择性扫描 — SSM 的核心计算。

    离散化 (zero-order hold):
      A_bar = exp(Δ * A)
      B_bar = Δ * B   (简化)
      h_t = A_bar * h_{t-1} + B_bar * x_t
      y_t = C_t * h_t

    形状约定:
      x:     [B, S, d_inner]    每个通道一个 SSM
      A:     [d_inner, d_state] 每个通道独立的 state 转移矩阵
      B/C:   [B, S, d_state]    输入相关的选择向量
      delta: [B, S, d_inner]    每个通道独立的步长
    返回:    [B, S, d_inner]

    本实现用显式循环 (无并行 scan)。
    """
    batch_size, seq_len, d_inner = x.shape
    d_state = A.size(-1)
    dtype = x.dtype

    delta = F.softplus(delta)  # [B, S, d_inner]
    A_bar = torch.exp(delta.unsqueeze(-1) * A)  # [B, S, d_inner, d_state]
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2)  # [B, S, d_inner, d_state]
    u = B_bar * x.unsqueeze(-1)  # [B, S, d_inner, d_state]

    # 显式循环 scan (非并行版本)
    h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=dtype)
    ys = []
    for t in range(seq_len):
        h = A_bar[:, t] * h + u[:, t]  # [B, d_inner, d_state]
        y = (C[:, t].unsqueeze(-2) * h).sum(dim=-1)  # [B, d_inner]
        ys.append(y)

    return torch.stack(ys, dim=1)  # [B, S, d_inner]


class MambaBlock(nn.Module):
    """Mamba 层 — conv1d + SiLU gate + selective SSM (Mamba-1 教学版)。

    架构:
      x → RMSNorm → conv1d → SiLU → SSM → gate(out) → output

    SSM 参数:
      d_state = d_model 的 "state size" — 通常 16-64
      d_conv = 卷积核大小 — 通常 4
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        use_gate: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        d_inner = int(d_model * expand_factor)
        # forward 的 split/缩放引用 self.d_inner; 原独立文件版只存了局部变量,
        # 任何一次前向都会 AttributeError (此前无测试覆盖, 并入后由混合模型测试兜底)
        self.d_inner = d_inner
        self.use_gate = use_gate

        # 输入投影: x → [x', z]，z 为控制门
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # 因果 padding: 只看过去
            groups=d_inner,  # depthwise conv
            bias=False,
        )

        # B/C/Δ 由 x' 投影得到 (selective)；A 固定, log 参数化保证 exp 后为正
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_inner * d_state + 1, dtype=torch.float)).view(
                d_inner, d_state
            )
        )
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        """MambaBlock 前向。作为 'm' 块参与块级混合。

        调用签名与 DecoderLayer 一致, 但 SSM 不需要 RoPE / attention mask /
        past_kv — 这些参数仅用于占位, 被忽略 (conv/scan 天然因果)。
        返回 (output, None, None): 后两位占位对齐统一块契约 (kv, aux),
        SSM 无 KV cache 也无 aux_loss。
        """
        batch_size, seq_len, _ = x.shape

        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)  # [B, S, d_inner] each

        x_conv = x_inner.transpose(1, 2)  # [B, d_inner, S]
        x_conv = self.conv1d(x_conv)[..., :seq_len]  # 因果裁切
        x_conv = F.silu(x_conv)  # [B, d_inner, S]
        x_conv = x_conv.transpose(1, 2)  # [B, S, d_inner]

        # SSM: A 取负保证衰减, B/C/Δ 由投影得到
        A = -torch.exp(self.A_log)  # [d_inner, d_state] (负 → 衰减)
        bc_delta = self.x_proj(x_conv)  # [B, S, d_state*2 + d_inner]
        B, C, delta = bc_delta.split([self.d_state, self.d_state, self.d_inner], dim=-1)

        y_ssm = selective_scan(x_conv, A, B, C, delta)  # [B, S, d_inner]

        y = y_ssm + self.D * x_conv

        if self.use_gate:
            y = y * F.silu(z)

        y = self.out_proj(y)
        output = residual + y

        return output, None, None


# 两种块的统一静态类型: ModuleList 索引在 torch 存根里是 nn.Module,
# 用 cast 收窄到这两种具体层后, forward 联合签名可推导出三元组类型。
HybridBlock = DecoderLayer | MambaBlock
HybridBlockOut = tuple[torch.Tensor, PastKeyValue | None, torch.Tensor | None]


class MambaHybridModel(nn.Module):
    """块级混合解码器: pattern 指定 attention / mamba 块的交替排布。

    embedding / RoPE 缓存 / 因果 mask / 权重初始化 与 GleamLMModel 同款
    (Linear 系统一 1/√d 初始化; MambaBlock 的 conv1d 与 A_log/D 保持其默认),
    便于与主模型做等条件对照实验。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        pad_token_id: int = 0,
        tie_weights: bool = True,
        use_flash_attn: bool = False,
        use_gradient_checkpointing: bool = False,
        # 块交替排布: 'a' = attention 块, 'm' = mamba 块; 按 pattern 循环平铺
        pattern: str = "ammmmmmm",
        # attention 块变体 (透传 DecoderLayer)
        attn_variant: type = GQA,
        ffn_variant: type = MLP,
        norm_variant: type = RMSNorm,
        num_experts: int = 8,
        top_k: int = 2,
        # mamba 块超参 (透传 MambaBlock)
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand_factor: int = 2,
        mamba_use_gate: bool = True,
        # YaRN 长度外推参数 (attention 层 RoPE 用)
        rope_scale: float = 1.0,
        rope_factor: float = 8.0,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.head_dim = d_model // num_heads
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.rope_scale = rope_scale
        self.rope_factor = rope_factor
        self.rope_theta = rope_theta
        self.rope_original_max_seq_len = max_seq_len
        # RoPE 缓存 = 基础长度 × 扩展因子 × 缓冲乘数 (与 GleamLMModel 同款)
        self.rope_max_len = int(max_seq_len * max(rope_scale, 1.0) * rope_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self._use_flash_attn = use_flash_attn

        if not pattern or any(c not in "am" for c in pattern):
            raise ValueError(
                "pattern 只允许 'a'(attention 块) 与 'm'(mamba 块), Jamba 1:7 风格用 'ammmmmmm'"
            )
        self.pattern = pattern
        self.block_kinds = [pattern[i % len(pattern)] for i in range(num_layers)]
        self.num_attn_blocks = sum(k == "a" for k in self.block_kinds)
        self.num_mamba_blocks = num_layers - self.num_attn_blocks

        # Token Embedding: padding_idx 让 pad token 在反向传播中梯度为 0
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embed_dropout = nn.Dropout(dropout)

        # 按 pattern 交替堆叠两种完整层 (各自 PreNorm + 残差, 无需外层包装)
        self.layers = nn.ModuleList()
        for kind in self.block_kinds:
            if kind == "a":
                self.layers.append(
                    DecoderLayer(
                        d_model,
                        num_heads,
                        num_kv_heads,
                        d_ff,
                        dropout,
                        use_flash_attn,
                        attn_variant=attn_variant,
                        ffn_variant=ffn_variant,
                        norm_variant=norm_variant,
                        num_experts=num_experts,
                        top_k=top_k,
                    )
                )
            else:
                self.layers.append(
                    MambaBlock(
                        d_model,
                        d_state=mamba_d_state,
                        d_conv=mamba_d_conv,
                        expand_factor=mamba_expand_factor,
                        use_gate=mamba_use_gate,
                    )
                )

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: 输入映射与输出映射共享权重 (与 GleamLMModel 同款)
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        # RoPE / YaRN 预计算 (persistent=False: 不写入 state_dict, 加载时重算)
        cos, sin = precompute_freqs_cis(
            self.head_dim,
            self.rope_max_len,
            base=self.rope_theta,
            rope_scale=self.rope_scale,
            rope_factor=self.rope_factor,
            original_max_seq_len=self.rope_original_max_seq_len,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _recompute_rope_cache(self) -> None:
        """meta device 物化后重建非持久 RoPE buffer (与 GleamLMModel 同款)。"""
        device = next(self.parameters()).device
        cos, sin = precompute_freqs_cis(
            head_dim=self.head_dim,
            max_seq_len=self.rope_max_len,
            base=self.rope_theta,
            rope_scale=self.rope_scale,
            rope_factor=self.rope_factor,
            original_max_seq_len=self.rope_original_max_seq_len,
        )
        self.register_buffer("rope_cos", cos.to(device), persistent=False)
        self.register_buffer("rope_sin", sin.to(device), persistent=False)

    # 初始化与主模型同款: Embedding/Linear 用 1/√d 使各层方差保持 O(1),
    # LM Head 用小标准差防止 softmax 饱和。注意: MambaBlock 的三个 Linear
    # (in_proj/x_proj/out_proj) 会被本循环重初始化为同尺度 — 与 MambaBlock
    # 单独使用时的默认 init 不同, 此处是有意对齐主模型的方差尺度;
    # conv1d 与 A_log/D 不是 nn.Linear, 保持 MambaBlock 默认不受影响。
    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=self.d_model**-0.5)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=module.weight.size(1) ** -0.5)
        if self.lm_head.weight is not self.token_embed.weight:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # 因果掩码 [1, 1, S, total_len]: 上三角 -inf; offset>0 时已有前缀不 mask
    def _create_causal_mask(
        self, seq_len: int, offset: int = 0, device: torch.device | None = None
    ) -> torch.Tensor:
        total = offset + seq_len
        device = device or torch.device("cpu")
        # diagonal=offset+1: 让前 offset 列全 0 (已有 KV), 当前序列内上三角 -inf
        mask = torch.triu(
            torch.full((seq_len, total), float("-inf"), device=device), diagonal=offset + 1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv_list: PastKeyValueList | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[PastKeyValue | None], torch.Tensor, torch.Tensor | None]:
        """前向。返回 (logits, kv_list, aux_loss, hidden)，签名同 GleamLMModel。

        - attention 块产 (K, V) 进 kv_list; mamba 块无 KV, 以 None 占位,
          使 kv_list 长度恒等于 num_layers (与主模型逐层对齐)。
        - mamba 无跨步隐状态缓存 → 不支持 past_kv_list 增量输入 (传 None)。
        """
        if past_kv_list is not None:
            raise NotImplementedError(
                "MambaHybridModel 暂不支持增量生成: mamba 块需要 SSM 隐状态缓存, "
                "混合 KV + state 缓存尚未实现; 请整段前向 (训练/评测均可)。"
            )
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # cast: torch 存根把动态注册 buffer 的类型记为 Tensor | Module,
        # 局部变量收窄后 .forward 参数检查才通过 (mypy strict)
        rope_cos = cast(torch.Tensor, self.rope_cos)
        rope_sin = cast(torch.Tensor, self.rope_sin)

        x = self.token_embed(input_ids)
        x = self.embed_dropout(x)

        if seq_len > rope_cos.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds pre-allocated RoPE cache "
                f"({rope_cos.size(0)}). Increase max_seq_len in config or "
                f"set a larger multiplier in MambaHybridModel.__init__."
            )

        attn_mask = self._create_causal_mask(seq_len, offset=0, device=device)

        # HF 的 attention_mask (1=keep 0=pad): 只对 attention 块生效
        # (mamba 块的 conv/scan 天然因果且不感知 padding)
        if attention_mask is not None:
            pad = attention_mask.to(dtype=attn_mask.dtype, device=device)
            if pad.size(1) > seq_len:
                pad = pad[:, :seq_len]
            pad = pad[:, None, None, :]
            attn_mask = attn_mask.masked_fill(pad == 0, float("-inf"))

        new_kv_list: list[PastKeyValue | None] = []
        aux_loss_total = torch.tensor(0.0, device=device)
        for i in range(self.num_layers):
            # cast: ModuleList 索引的静态类型是 nn.Module, 收窄到两种具体层后
            # .forward 才有完整联合签名 (mypy strict 下 __call__ 返回 Any)
            block = cast(HybridBlock, self.layers[i])
            # 反向时重算激活值换显存 (与主模型同款); aux 恒经返回值传递 —
            # no-reentrant checkpoint 只对 fn 返回值回传梯度, 属性逃逸会断链
            if self.training and self.use_gradient_checkpointing:

                def _run_block(
                    x: torch.Tensor,
                    rope_cos: torch.Tensor,
                    rope_sin: torch.Tensor,
                    mask: torch.Tensor | None,
                    past_kv: PastKeyValue | None = None,
                    block: HybridBlock = block,
                ) -> HybridBlockOut:
                    return block.forward(x, rope_cos, rope_sin, mask, past_kv)

                x, current_kv, aux = torch.utils.checkpoint.checkpoint(
                    _run_block,
                    x,
                    rope_cos,
                    rope_sin,
                    attn_mask,
                    use_reentrant=False,
                )
            else:
                x, current_kv, aux = block.forward(x, rope_cos, rope_sin, attn_mask)
            new_kv_list.append(current_kv)
            if aux is not None:
                aux_loss_total = aux_loss_total + aux

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        if not use_cache:
            new_kv_list = []

        if output_hidden_states:
            return logits, new_kv_list, aux_loss_total, hidden

        return logits, new_kv_list, aux_loss_total, None

    def get_num_params(self) -> tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        total_buffers = sum(b.numel() for b in self.buffers())
        return total_params, total_params + total_buffers

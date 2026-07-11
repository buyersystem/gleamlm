# ADR-0008: Real-Number RoPE Implementation

RoPE uses real-number operations (`x * cos + rotate_half(x) * sin`) instead of PyTorch's `view_as_complex`. This avoids FP32/FP16 type conversion overhead and is 2-3× faster on GPU.

The complex-number path also caused a dimension-pairing bug in V1-V3 where RoPE was applied to incorrectly paired dimensions.

Consequence: The current `apply_rotary_emb` function in `model.py` uses real-number ops. The earlier dimension-pairing bug is documented in `docs/GleamLM排坑记录.md` entry #22.

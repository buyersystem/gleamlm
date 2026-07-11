"""GleamLM model deployment utilities.

quantize.py  — FP32 → FP16 (current)
(future)     — INT8, GPTQ, GGUF, ONNX export, model packing
"""

from gleamlm.deploy.quantize import quantize_to_fp16

__all__ = ["quantize_to_fp16"]

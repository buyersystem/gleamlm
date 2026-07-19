"""Checkpoint architecture validation.

Prevents silent loading of mismatched model architectures when
load_state_dict(strict=False) could swallow shape mismatches.
"""

from __future__ import annotations


def assert_same_architecture(
    checkpoint_config: dict[str, int],
    current_config: dict[str, int],
    source: str = "checkpoint",
) -> None:
    """比对 checkpoint 与当前模型的架构参数，不匹配时抛出 ValueError。"""
    critical_keys = [
        "vocab_size",
        "d_model",
        "num_layers",
        "num_heads",
        "num_kv_heads",
    ]

    mismatches: list[str] = []
    for key in critical_keys:
        ckpt_val = checkpoint_config.get(key)
        cur_val = current_config.get(key)
        if ckpt_val is not None and cur_val is not None and ckpt_val != cur_val:
            mismatches.append(f"  {key}: {source}={ckpt_val}, current={cur_val}")

    if mismatches:
        raise ValueError(
            f"Architecture mismatch between {source} and current model:\n"
            + "\n".join(mismatches)
            + "\nRefusing to load. Verify model architecture matches the checkpoint."
        )

"""PyTorch utility functions."""

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager

import torch


def get_lr_cosine(
    step: int, total_steps: int, warmup_ratio: float = 0.01, min_lr_ratio: float = 0.1
) -> float:
    """Cosine annealing with warmup. Returns multiplier in [0, 1]."""
    warmup_steps = int(total_steps * warmup_ratio)

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def get_lr_wsd(
    step: int,
    total_steps: int,
    warmup_ratio: float = 0.02,
    stable_ratio: float = 0.80,
    min_lr_ratio: float = 0.05,
) -> float:
    """WSD scheduler. Returns multiplier in [0, 1]."""
    warmup_steps = int(total_steps * warmup_ratio)
    stable_steps = int(total_steps * stable_ratio)
    decay_steps = total_steps - warmup_steps - stable_steps

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    elif step < warmup_steps + stable_steps:
        return 1.0
    else:
        progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@contextmanager
def safe_autocast(
    enabled: bool = True, *, dtype: torch.dtype = torch.bfloat16
) -> Generator[None, None, None]:
    """Safe autocast context manager with automatic backend selection."""
    if not enabled:
        yield
        return

    if torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=dtype):
            yield
        return

    if (
        dtype == torch.bfloat16
        and hasattr(torch, "cpu")
        and callable(getattr(torch.cpu, "is_bf16_supported", None))
        and torch.cpu.is_bf16_supported()  # type: ignore[attr-defined]
    ):
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            yield
        return

    yield

"""Shared path constants for GleamLM. Eliminates duplicate _SCRIPT_DIR patterns."""

from __future__ import annotations

import os

from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH  # noqa: F401 — single source of truth

__all__ = [
    "DEFAULT_TOKENIZER_PATH",
    "get_root_dir",
    "get_config_dir",
    "get_default_checkpoint_dir",
    "get_default_data_dir",
]


def get_root_dir() -> str:
    """Project root directory (parent of gleamlm/ package)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_config_dir() -> str:
    """configs/ directory."""
    return os.path.join(get_root_dir(), "configs")


def get_default_checkpoint_dir(variant: str = "nano") -> str:
    """Checkpoint directory for a model variant (nano/lite/pro)."""
    return os.path.join(get_root_dir(), f"gleamlm-{variant}", "checkpoints")


def get_default_data_dir(variant: str = "nano") -> str:
    """Returns data dir relative to repo root (unlike config.py's cwd-based default)."""
    return os.path.join(get_root_dir(), "data", f"{variant}_data")

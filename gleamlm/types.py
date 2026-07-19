"""Shared type aliases."""

from __future__ import annotations

import torch


class ConfigValidationError(Exception):
    """Invalid configuration format or value."""


PastKeyValue = tuple[torch.Tensor, torch.Tensor]
PastKeyValueList = list[PastKeyValue]

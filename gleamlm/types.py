"""GleamLM 共享类型别名"""
from __future__ import annotations
from typing import Tuple, List
import torch

PastKeyValue = Tuple[torch.Tensor, torch.Tensor]
PastKeyValueList = List[PastKeyValue]

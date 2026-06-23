"""Dataset and collate_fn tests"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gleamlm_dataset import LMDataset, collate_fn
from tokenizer.bbpe_tokenizer import BBPETokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return BBPETokenizer.load("./tokenizer/checkpoints/bbpe_12k")


def test_dataset_creation(tokenizer):
    ds = LMDataset("./data/splits", tokenizer, 512, "valid")
    assert len(ds) > 0


def test_getitem_shape(tokenizer):
    ds = LMDataset("./data/splits", tokenizer, 512, "valid")
    sample = ds[0]
    assert sample.dim() == 1
    assert 256 <= sample.size(0) <= 513  # max_seq_len + 1


def test_collate_fn(tokenizer):
    ds = LMDataset("./data/splits", tokenizer, 512, "valid")
    samples = [ds[i] for i in range(min(4, len(ds)))]
    input_ids, target_ids = collate_fn(samples, pad_id=tokenizer.pad_id)
    assert input_ids.dim() == 2
    assert target_ids.dim() == 2
    assert input_ids.size(0) == len(samples)
    assert target_ids.size(0) == len(samples)
    assert target_ids.size(1) == input_ids.size(1)  # same seq len

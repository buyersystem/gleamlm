"""GleamLM data module — preprocessing pipeline + dataset."""

from gleamlm.data.dataset import LMDataset, collate_fn
from gleamlm.data.preprocess import (
    SimHashIndex,
    clean_file,
    clean_text,
    dedup_file,
    filter_qa,
    hamming_distance,
    normalize,
    parse_qa,
    simhash,
    stream_split,
)

__all__ = [
    "clean_text",
    "clean_file",
    "normalize",
    "simhash",
    "hamming_distance",
    "SimHashIndex",
    "dedup_file",
    "parse_qa",
    "filter_qa",
    "stream_split",
    "LMDataset",
    "collate_fn",
]

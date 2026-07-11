"""GleamLM data preprocessing library.

Core preprocessing algorithms (clean, dedup, filter, build). Moved from data_tools/.
The orchestration script (prepare_data.py) lives in data_tools/ per ADR-0011.
"""

from gleamlm.preprocessing.build_dataset import main as build_dataset_main
from gleamlm.preprocessing.clean_text import main as clean_text_main
from gleamlm.preprocessing.dedup_text import main as dedup_text_main
from gleamlm.preprocessing.filter_qa import main as filter_qa_main

__all__ = [
    "build_dataset_main",
    "clean_text_main",
    "dedup_text_main",
    "filter_qa_main",
]

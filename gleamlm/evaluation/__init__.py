"""GleamLM 统一评估框架 — PPL / 知识探针 / CEVAL / CMMLU"""

from .ppl import evaluate_ppl, evaluate_multiple, PPLResult
from .knowledge import evaluate_knowledge, KnowledgeResult
from .benchmark import evaluate_ceval, evaluate_cmmlu, BenchmarkResult

__all__ = [
    "evaluate_ppl", "evaluate_multiple", "PPLResult",
    "evaluate_knowledge", "KnowledgeResult",
    "evaluate_ceval", "evaluate_cmmlu", "BenchmarkResult",
]

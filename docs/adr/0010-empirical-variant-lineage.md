# ADR-0010: Empirical-Driven Variant Lineage

Each model variant addresses a specific finding from the previous variant:

| Variant | Key learning from predecessor | Response |
|---------|-------------------------------|----------|
| Nano (40M) | — | Establish the Deep-Narrow baseline |
| Lite (87M) | Nano had 4% factual accuracy | 3.4× FFN expansion, same 12 layers |
| Pro (126M) | (planned) Verify deeper scaling | 18 layers at 768d |
| 0.6B (597M) | (planned) Industrial-scale verification | 37×1024, 64K vocab, Linux stack |

Architecture changes are driven by empirical results (ablation testing, evaluation metrics), not arbitrary scaling rules.

Consequence: Each variant's architecture has a documented rationale traceable to specific experiments in `docs/测试报告.md`.

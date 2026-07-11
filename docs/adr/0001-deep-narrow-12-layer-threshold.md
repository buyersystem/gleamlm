# ADR-0001: Deep-Narrow — 12 Layers is the Minimum Viable Threshold for Chinese

All models must have ≥12 layers. Increasing depth consistently outperforms increasing width at small scales.

Layer dropout experiments on GleamLM-Nano (12L × 512d): dropping to 11 layers caused a 60% output diversity cliff; 8 layers produced unreadable output (PPL 34.93 vs 13.65 at 12 layers). The 12-layer minimum is the single most important architectural constraint.

Consequence: Lite (87M) stayed at 12 layers while expanding FFN. Pro (126M) goes to 18 layers. 0.6B uses 37 layers.

# ADR-0009: ChatML Format with Loss Mask for SFT

Supervised fine-tuning uses ChatML format with `<|im_start|>role\n...<|im_end|>` delimiters. The loss is computed only on assistant segments; prompt and system tokens are masked out (`label = -100`).

This switches model behavior from raw text continuation to structured assistant response. The BBPE tokenizer natively registers `<|im_start|>` and `<|im_end|>` as single tokens (IDs 12000, 12001), ensuring format tokens never get broken by BPE merges.

Consequence: SFT and DPO training loops both use this ChatML + loss mask pattern. The tokenizer's `encode_chatml()` helper constructs correctly formatted sequences.

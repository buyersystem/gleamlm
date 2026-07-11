# ADR-0002: BBPE 12K Over SentencePiece 32K BPE

Replaced SentencePiece 32K BPE with a self-developed Byte-Level BPE tokenizer (12K vocab, pure Python, zero dependencies).

The SentencePiece tokenizer consumed 42% of total parameters (16.4M out of ~39M) in the embedding layer. BBPE 12K reduced this to 15% (6.1M), freeing 10.3M parameters for Transformer layers. The tokenizer also natively registers ChatML tokens as single token IDs.

Consequence: Vocabulary fixed at 12K for Nano / Lite / Pro. Only 0.6B plans to jump to 64K with an independent `lm_head`.

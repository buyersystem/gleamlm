# ADR-0004: Zero HuggingFace Dependency

`GleamLMModel` inherits from `nn.Module`, not from HuggingFace's `PreTrainedModel`. Training loop, DDP, checkpoint management, and AMP are all hand-written.

This was an explicit educational decision: the codebase should be fully transparent with zero hidden abstractions. The entire model is ~2500 lines of readable PyTorch.

Consequence: Cannot use vLLM, ollama, or llama.cpp without format conversion. The project prioritizes understanding over ecosystem compatibility.

# ADR-0005: Mono-Repo with Shared Core Library

Single repository. `gleamlm/` is the shared core library (model, tokenizer, dataset, inference, evaluation, utils). `gleamlm-nano/` and `gleamlm-lite/` are variant-specific scripts that import from `gleamlm/`.

A multi-package split was explicitly considered and rejected. The core library enforces a single source of truth for all shared components while leaving training scripts, variant-specific configs, and smoke tests to each variant directory.

Consequence: No package fragmentation. Adding a new variant (Pro, 0.6B) means creating a new directory with a training script and a config YAML, not a new package.

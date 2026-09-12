# GleamLM — Domain Glossary

## Core Architecture

- **Decoder-only Transformer**: Autoregressive architecture, no encoder, no cross-attention.
- **Pre-Norm**: RMSNorm applied _before_ each sublayer, not after.
- **RMSNorm**: Root Mean Square Layer Normalization; replaces LayerNorm.
- **RoPE**: Rotary Position Embedding, implemented via real-number ops (`x*cos + rotate_half(x)*sin`), not complex numbers.
- **GQA (Grouped Query Attention)**: Q heads > KV heads (e.g. 8Q / 4KV). KV heads repeat across groups.
- **QK-Norm**: Applying RMSNorm to Q and K vectors before RoPE (LLaMA 3 / Qwen3 standard).
- **SwiGLU**: Gated activation: `silu(W_gate * x) * (W_up * x)`, output via `W_down`.
- **FFN capacity**: The intermediate dimension `d_ff` of SwiGLU, typically `8/3 * d_model` by the standard formula.
- **Weight tying**: Embedding table reused as output projection (`lm_head`), reducing parameters.
- **KV Cache**: Store past key/value tensors for incremental generation, avoiding recomputation.

## Tokenizer

- **BBPE (Byte-Level BPE)**: Works on UTF-8 byte sequences; 256 base tokens + BPE merges. Self-developed, zero dependencies.
- **ChatML tokens**: `<|endoftext|>`=0 (pad/unk, only for aligning sample lengths in batches), `<|im_start|>`=1 (bos), `<|im_end|>`=2 (eos). Pre-train document boundary token = eos(id 2) — each doc in `.bin` ends with `<|im_end|>`, same token also terminates SFT turns. `<|buffer1|>`-`<|buffer10|>` at IDs 3-12 for future extensions.
- **CJK pre-tokenization**: Each Chinese character is a separate token unit; non-CJK kept as contiguous segments.
- **Frequency-aggregated training**: `train_from_files` dedupes repeated words via `Counter` (bytes → count) and indexes pairs as `pair → {(word_idx, pos)}` with frequency weighting. Memory drops from O(total words) to O(unique words) — 200M chars uses ~1.2GB vs 40GB+, ~9× faster, merge results bit-identical to the old position-level algorithm (verified by equivalence test).
- **Variant-driven training entry**: `manual/train_tokenizer.py --variant nano` reads `data_sources` ratios from `manual/configs/{variant}.yaml` (edu 55 / news 27 / wiki 12 / baike 6) and trains from `data/raw/{name}_dedup.txt`; `--data_dir` / `--base_tokenizer` / `--verify_only` modes remain backward-compatible.

## Architecture Philosophy

- **Deep-Narrow**: More layers (≥12) with narrow hidden dimension (512/768). Proven superior for small Chinese models.
- **"Embedding is the gatekeeper, FFN is the brain"**: Minimize embedding parameters to maximize Transformer capacity. All factual knowledge resides in FFN weights.
- **12 layers is the minimum viable threshold for Chinese generation**: Dropping from 12→11 layers causes a 60% output diversity cliff.

## Training

- **AMP (Automatic Mixed Precision)**: BF16/FP16 training with `GradScaler`.
- **DDP (Distributed Data Parallel)**: Single-command multi-GPU via `torchrun`.
- **memmap dataset**: Tokenized data stored as Megatron-standard `.bin/.idx`, memory-mapped via `np.memmap` (`IndexedMMapDataset`). ~1 MB RAM for any dataset size.
- **Gradient accumulation**: `effective_batch = micro_batch × accumulate_grad`.
- **Z-Loss**: Regularizer `1e-4 * mean(logsumexp(logits)^2)`, prevents logit explosion.
- **WSD scheduler**: Warmup → Stable → Decay (3-phase learning rate).
- **Cosine scheduler**: Cosine Annealing + Warmup (2-phase learning rate).
- **Chinchilla optimal**: `tokens ≈ 20 × params` for compute-optimal training.

## Inference

- **Streaming generation**: Yield tokens incrementally (every 4 tokens by default).
- **Repetition penalty**: Divide logits of already-generated tokens to reduce loops.
- **Sampling strategies**: temperature, top-k, top-p, greedy (temperature=0).

## Data

- **Character-weighted mixing**: Convert target character% ratios to line-count ratios based on average characters-per-line.
- **Sliding window**: Overlapping windows with `stride = 3/4 * max_seq_len`.

## Repo Architecture

- **Core library (`gleamlm/`)**: Provides importable, variant-agnostic functions and classes (model, tokenizer, dataset, training loops, inference engine, evaluation, preprocessing, deployment). All symbols are parameterized — no hardcoded variant-specific defaults.
- **Variant configuration (`manual/configs/nano.yaml`, `manual/configs/lite.yaml`, `manual/configs/pro.yaml`)**: Manual-track-only YAML config files (industrial track has its own `industrial/configs/`); each variant is fully self-contained — no `extends` inheritance, a snapshot of public defaults + variant values — defining model architecture, training hyperparameters, data paths, and SFT/DPO settings. `base.yaml` serves as the public-default reference / template for new configs (copy & modify).
- **Recipe test**: Copy a variant script to a fresh directory, change a few parameters, and it should run. If imports break or the reader can't trace what each step does, the script fails the test.
- **Orchestration-visible principle**: `main()` must be readable without jumping to other files. Steps are listed explicitly; implementation details are delegated to `gleamlm/` imports.
- **Deletion test**: If removing a function from `gleamlm/` would break variant scripts, it belongs in `gleamlm/`. If it would only affect one variant, it belongs in that variant's directory.
- **GUI training console (`webui/`)**: Browser console (pretrain / posttrain / inference tabs) that launches `manual/` scripts as subprocesses and streams logs + metrics. It is a tool layer (like `serve/`), may import `gleamlm/` + `hf/`, and must **not** be a default-value authority — its `/api/train/defaults` fallbacks mirror the manual scripts' naming conventions, guarded by `tests/test_webui_api.py::test_artifact_conventions_match_manual`.
- **User config copies (`manual/my_configs/`)**: WebUI-writable per-user config snapshots (gitignored); built-in `manual/configs/*.yaml` stay read-only. "Save as" copies a built-in template here.
- **Sentinel metrics contract (`gleamlm/utils/metrics.py`)**: Training scripts emit an extra machine-readable line `@@GLEAM_METRIC {json}` (via `emit_metric`) alongside human-readable logs; WebUI parses it with zero regex as the primary channel; logs without sentinel lines fall back to the regex chain (`--no-pbar` pretrain lines → val lines → tqdm frames). The key allow-list is the single source of truth — human-readable lines no longer carry machine-interface duty.

See `adr/0011-library-vs-recipe-architecture.md` for the full design rationale (ADR archive lives at repo root `adr/`, local-only like the design doc).

## Model Variants

- **GleamLM-Nano (~40M)**: 12L × 512d, BBPE 12K. Baseline, complete (v0.1.0).
- **GleamLM-Lite (~87M)**: 12L × 768d, d_ff=2048 (3.4× FFN). Basic lifecycle complete; advanced training deferred.
- **GleamLM-Pro (~126M)**: 18L × 768d, BBPE 12K, d_ff=2048. In development.
- **GleamLM-0.6B (~597M)**: 37L × 1024d, BBPE 24K, YaRN, Linux-only. Planned.

## _Avoid_

- Do not import from HuggingFace (`transformers`, `datasets`, `tokenizers`). GleamLM is self-contained.
- Do not use `torch.view_as_complex` for RoPE. Use real-number operations.
- Do not assume GPU is available; code paths must handle CPU fallback.
- Do not add comments explaining _what_ the code does; comments are for _why_.

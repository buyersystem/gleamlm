"""Unified autoregressive token generator — the single KV-cache loop used by all callers.

Replaces the duplicated per-file generate loops in generate_response(), TextStreamer.generate(),
knowledge._simple_generate(), and scripts/eval_knowledge.generate().
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from gleamlm.inference.sampler import sample_token


def generate_tokens(
    model: torch.nn.Module,
    prompt_ids: list[int],
    device: torch.device,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    stop_ids: set[int] | None = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype | None = None,
) -> Iterator[int]:
    """Autoregressive token generation with KV cache.

    Yields one token ID at a time. Stops when a token in `stop_ids` is produced
    or `max_new_tokens` is reached.

    Args:
        model: The GleamLMModel (or DDP-wrapped).
        prompt_ids: Pre-encoded prompt token IDs (without BOS/EOS padding).
        device: torch device.
        max_new_tokens: Maximum tokens to generate.
        temperature, top_k, top_p, repetition_penalty: Sampling parameters.
        stop_ids: Set of token IDs that trigger generation stop. None = no early stop.
        use_amp: If True, wraps forward pass in torch.amp.autocast.
        amp_dtype: AMP dtype (bfloat16, float16). None = default autocast behaviour.
    """
    model.eval()
    generated_ids: list[int] = prompt_ids.copy()

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, past_kv = _forward(model, input_ids, use_amp, amp_dtype)

    for _ in range(max_new_tokens):
        next_token = sample_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids,
        )
        token_id = int(next_token.item())

        if stop_ids and token_id in stop_ids:
            return

        yield token_id
        generated_ids.append(token_id)
        next_input = torch.tensor([[token_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, past_kv = _forward(model, next_input, use_amp, amp_dtype, past_kv)


def _forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    if use_amp:
        amp_device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.amp.autocast(amp_device, dtype=amp_dtype):  # type: ignore[attr-defined]
            return model(input_ids, past_kv_list=past_kv)  # type: ignore[no-any-return]
    return model(input_ids, past_kv_list=past_kv)  # type: ignore[no-any-return]

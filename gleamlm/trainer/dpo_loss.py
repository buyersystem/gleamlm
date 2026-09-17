"""DPO (Direct Preference Optimization) loss functions.

Provides compute_log_probs, dpo_loss, get_reference_logps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gleamlm.utils.torch_utils import safe_autocast


def compute_log_probs(
    logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute per-token log probabilities, masked. Returns [B]."""
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs_token = log_probs_all[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (log_probs_token * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    term = (policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp)
    return -F.logsigmoid(beta * term).mean()


@torch.no_grad()
def get_reference_logps(
    ref_model: torch.nn.Module,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute chosen and rejected log-probs from frozen reference model."""
    ref_model.eval()
    with safe_autocast():
        c_logits, _, _, _ = ref_model(chosen_ids)
        r_logits, _, _, _ = ref_model(rejected_ids)
    ref_cho = compute_log_probs(c_logits.float(), chosen_ids, chosen_mask)
    ref_rej = compute_log_probs(r_logits.float(), rejected_ids, rejected_mask)
    return ref_cho, ref_rej


# DPO held-out 验证（K8）：面板 val 曲线的数据源


@torch.no_grad()
def evaluate_dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    data_loader: DataLoader,
    beta: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """DPO val：与训练同式的 dpo_loss / margin / acc → (loss, margin, acc)。

    口径与 dpo.py 训练循环严格一致（get_reference_logps + compute_log_probs +
    dpo_loss, bf16 autocast）—— train/val 同口径是硬约束。按 pair 数加权聚合
    （各批对数不等时防小批权重被放大）；margin=β·mean(term)、acc=mean(term>0)
    与训练侧监控量定义同源。模型模式由调用方管理（前后自行 eval/train）。
    """
    total_loss = 0.0
    total_term = 0.0
    total_pos = 0
    total_pairs = 0
    for batch in data_loader:
        chosen_ids = batch["chosen_ids"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)
        ref_cho, ref_rej = get_reference_logps(
            ref_model, chosen_ids, rejected_ids, chosen_mask, rejected_mask
        )
        with safe_autocast():
            c_logits, _, _, _ = policy_model(chosen_ids)
            r_logits, _, _, _ = policy_model(rejected_ids)
            policy_cho = compute_log_probs(c_logits.float(), chosen_ids, chosen_mask)
            policy_rej = compute_log_probs(r_logits.float(), rejected_ids, rejected_mask)
            loss = dpo_loss(policy_cho, policy_rej, ref_cho, ref_rej, beta)
        term = (policy_cho - ref_cho) - (policy_rej - ref_rej)
        n = term.numel()
        total_loss += loss.item() * n
        total_term += term.sum().item()
        total_pos += int((term > 0).sum().item())
        total_pairs += n
    if total_pairs == 0:
        raise ValueError("DPO val 集为空: 检查 val_data 与 dpad_collate")
    return total_loss / total_pairs, beta * total_term / total_pairs, total_pos / total_pairs

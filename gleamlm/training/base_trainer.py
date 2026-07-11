"""Pre-training shared utilities. Extracted from nano/train.py and lite/train.py.

Covers: set_seed, evaluate, checkpoint save/load, GradScaler creation,
DataLoader setup, optimizer/scheduler creation, model building.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gleamlm.dataset.dataset import LMDataset, collate_fn
from gleamlm.models.model import GleamLMModel
from gleamlm.utils.torch_utils import get_lr_cosine


def set_seed(seed: int) -> None:
    """Fixed random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_scaler() -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:  # pyright: ignore[reportDeprecated]
    """AMP GradScaler with CPU fallback (compatible with PyTorch 1.x / 2.x)."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[name-defined]
    return torch.cuda.amp.GradScaler()  # pyright: ignore[reportDeprecated]


def create_optimizer_and_scheduler(
    model: nn.Module,
    train_loader: DataLoader,
    args: Any,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    total_steps = math.ceil(len(train_loader) / args.accumulate_grad) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_cosine(
            step,
            total_steps,
            getattr(args, "warmup_ratio", 0.02),
            getattr(args, "min_lr_ratio", 0.1),
        ),
    )
    return optimizer, scheduler


def create_dataloaders(
    args: Any,
    train_dataset: LMDataset,
    val_dataset: LMDataset,
    pad_id: int,
) -> tuple[DataLoader, DataLoader]:
    if args.world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
            pin_memory=True,
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
            num_workers=0,
            pin_memory=False,
        )
    return train_loader, val_loader


def create_model(
    args: Any,
    pad_token_id: int,
    device: torch.device,
) -> GleamLMModel:
    return GleamLMModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_token_id=pad_token_id,
        tie_weights=True,
        use_flash_attn=getattr(args, "use_flash_attn", False),
    ).to(device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    pad_token_id: int = 0,
    world_size: int = 1,
) -> tuple[float, float]:
    """Validate and return (avg_loss, ppl). Aggregates across DDP ranks."""
    torch.cuda.empty_cache()

    from gleamlm.evaluation.ppl import _compute_raw_loss

    total_loss, total_tokens, _ = _compute_raw_loss(model, val_loader, device, pad_token_id)

    if world_size > 1 and dist.is_initialized():
        t_loss = torch.tensor(total_loss, device=device)
        t_tokens = torch.tensor(total_tokens, device=device)
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_tokens, op=dist.ReduceOp.SUM)
        total_loss = t_loss.item()
        total_tokens = int(t_tokens.item())

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    path: str,
    epoch: int,
    global_step: int,
    world_size: int,
    extra: dict[str, Any] | None = None,
) -> None:
    state_dict = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        **(extra or {}),
    }
    torch.save(state_dict, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    path: str,
    device: torch.device,
    world_size: int,
) -> dict[str, Any]:
    """Returns {start_epoch, global_step, best_val_loss}."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if world_size > 1:
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return {
        "start_epoch": checkpoint.get("epoch", 0) + 1,
        "global_step": checkpoint.get("global_step", 0),
        "best_val_loss": checkpoint.get("val_loss", float("inf")),
    }

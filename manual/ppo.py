"""
PPO (Proximal Policy Optimization) — 经典 RLHF 对齐。

PPO 四件套:
  1) policy clip loss:   L_clip = E[min(r*A, clip(r, 1-ε, 1+ε)*A)]
  2) value loss:         L_value = MSE(V(s), R - V(s))
  3) entropy bonus:      鼓励探索，防止过早收敛
  4) old policy sync:    每步把 current policy 复制到 old policy

用法:
  python manual/ppo.py --model checkpoints/nano/sft/sft_best.pt \
      --data data/rlhf.jsonl --output_dir checkpoints/nano/ppo
"""

import argparse
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from gleamlm.data.rl_data import RLHFDataset, tokenize_prompts
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.trainer.rl_trainer import ValueHead, compute_reward, ppo_loss
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config
from gleamlm.utils.meter import MetricWindow
from gleamlm.utils.metrics import emit_metric
from gleamlm.utils.torch_utils import clean_state_dict, safe_autocast


def train(args):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # 当前未实现 DDP (无 init_process_group / 梯度 all-reduce)，多卡直接禁用，
    # 避免各 rank 用不同数据分片静默训练出分歧副本。
    if world_size > 1:
        raise SystemExit("PPO 未实现 DDP，请单进程运行 (不要用 torchrun)")
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = BBPETokenizer.load(args.tokenizer_path or DEFAULT_TOKENIZER_PATH)
    dataset = RLHFDataset(args.data, max_seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: b,
    )

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = extract_checkpoint_config(ckpt)

    policy_model = GleamLMModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg.get("dropout", 0.0),
        max_seq_len=args.seq_len,
        pad_token_id=tokenizer.pad_id,
        use_flash_attn=cfg.get("use_flash_attn", False),
    ).to(device)
    policy_model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=False)
    policy_model.train()

    value_head = ValueHead(cfg["d_model"]).to(device)

    old_model = deepcopy(policy_model).eval()
    for p in old_model.parameters():
        p.requires_grad = False

    policy_optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=args.lr)

    if rank == 0:
        total = sum(p.numel() for p in policy_model.parameters())
        print(f"PPO — model: {total / 1e6:.2f}M, epsilon={args.epsilon}")
        os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    # 监控累积器（生成吞吐 / 显存 / 窗口 loss 均值 / 窗口 grad_norm max）——
    # 口径与实现见 gleamlm/utils/meter.py，7 个训练脚本共用同一份。
    # 生成型的 tok/s = 本迭代**新生成**的 token ÷ 整个迭代耗时（含 rollout）。
    win = MetricWindow(device)
    for _ in range(args.epochs):
        for batch_items in loader:
            win.start()
            # batch_items: list[{"prompt", "ground_truth"}]（RLHFDataset 返回 dict）
            batch_prompts = [it["prompt"] for it in batch_items]
            batch_ground_truth = [it.get("ground_truth") for it in batch_items]
            prompt_ids = tokenize_prompts(batch_prompts, tokenizer, args.seq_len).to(device)
            prompt_len = prompt_ids.size(1)

            gen_ids = None
            with torch.no_grad():
                ids = prompt_ids.clone()
                for _ in range(args.max_new_tokens):
                    logits, _, _, _ = old_model(ids)
                    nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    ids = torch.cat([ids, nxt], dim=-1)
                    if (nxt == tokenizer.eos_id).all():
                        break
                    if ids.size(1) >= args.seq_len:
                        break
                gen_ids = ids

            # 本步新生成 token = (总长 − prompt 长) × batch
            gen_tokens = (int(gen_ids.size(1)) - prompt_len) * len(batch_prompts)
            decoded = [
                tokenizer.decode(gen_ids[i, prompt_len:].tolist(), skip_special=True)
                for i in range(len(batch_prompts))
            ]
            rewards = torch.tensor(
                [compute_reward(d, gt) for d, gt in zip(decoded, batch_ground_truth, strict=True)],
                device=device,
                dtype=torch.float,
            )

            with safe_autocast():
                p_logits, _, _, hidden = policy_model(gen_ids, output_hidden_states=True)
                with torch.no_grad():
                    o_logits, _, _, _ = old_model(gen_ids)
                values = value_head(hidden)

            loss = ppo_loss(
                p_logits, o_logits, values, rewards, gen_ids, prompt_len, epsilon=args.epsilon
            )

            loss.backward()
            # K4: 双 head（policy / value）各裁一次，取大值上报 —— 任一 head 的尖峰都不漏。
            # （clip_grad_norm_ 返回裁剪前总范数；实参从左到右求值，裁剪顺序与原实现一致）
            grad_norm = max(
                float(torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.clip)),
                float(torch.nn.utils.clip_grad_norm_(value_head.parameters(), args.clip)),
            )
            win.add_grad_norm(grad_norm)
            policy_optimizer.step()
            policy_optimizer.zero_grad()
            value_optimizer.step()
            value_optimizer.zero_grad()
            win.stop(gen_tokens)  # 一个训练迭代（含 rollout）的生成吞吐
            win.add_loss(loss.item())

            for p, old_p in zip(policy_model.parameters(), old_model.parameters(), strict=True):
                old_p.data.copy_(p.data)

            if rank == 0 and (global_step + 1) % args.log_interval == 0:
                window_loss = win.loss
                # WebUI 面板可解析进度行 (与 sft.py tqdm postfix 同构); flush 保证实时
                print(
                    f"{global_step}/{len(loader) * args.epochs} [loss={window_loss:.4f}, lr={args.lr:.2e}]",
                    flush=True,
                )
                # 哨兵指标行（契约见 gleamlm/utils/metrics.py）: 面板优先消费。
                # loss 记窗口均值（自上次记录至今的全部 step 平均）
                win.refresh_gpu()  # 显存按上报节奏采样（查询较重，不逐步查）
                emit_metric(
                    split="train",
                    step=global_step,
                    total=len(loader) * args.epochs,
                    lr=args.lr,
                    **win.emit_kwargs(),
                )
                win.reset()
            global_step += 1

    if rank == 0:
        torch.save(
            {
                "model_state_dict": policy_model.state_dict(),
                "value_head": value_head.state_dict(),
                "_config": extract_checkpoint_config(ckpt),
                "step": global_step,
            },
            os.path.join(args.output_dir, "ppo_final.pt"),
        )


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM PPO alignment")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./checkpoints/ppo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tokenizer_path", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    from gleamlm.utils.logging_utils import setup_cli_logging

    setup_cli_logging()
    args = parse_args()
    train(args)

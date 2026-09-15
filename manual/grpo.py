"""
GRPO (Group Relative Policy Optimization) — DeepSeek 风格的 RLHF 对齐。

核心公式:
  loss = -E[log π_θ(y|x) * A] + β * KL(π_θ || π_ref)
  其中 A = (r_i - mean(r_group)) / std(r_group)

GRPO vs PPO:
  PPO:     value network + clip + GAE + entropy — 3 个 loss 项
  GRPO:    无 value network，优势 = group 内归一化奖励 — 1 个 loss 项
          更简单、更稳定、收敛更快

增强:
  1a 截断奖励守卫: 未以 eos 收尾的截断回答 clamp(max=0) —— 半截文本碰巧
     包含 ground_truth 不再拿 +1.0, 只惩罚不受益
  1b 零方差组动态重采样: 组内奖励全同 (std≈0) 的 prompt 无优势梯度, 只剩
     KL 拉扯 (推理预算白花); 从样本流取新 prompt 替换 (最多 --max_resample
     次), 末轮仍零方差的行不进 loss。--max_resample 0 = 关闭 (行为同旧版)

用法:
  python manual/grpo.py --model checkpoints/nano/sft/sft_best.pt \
      --data data/rlhf.jsonl --output_dir checkpoints/nano/grpo
"""

# RLHF 流水线: SFT → RM → RL。PPO 需 4 个模型 (policy+ref+reward+value)，
# value network 与 policy 同尺寸显存翻倍；GRPO 砍掉 value network，
# 用 group 内奖励统计量做 MC baseline: A_i = (r_i - mean(r_group)) / std(r_group)。
# 代价是推理预算增加 group_size 倍 (DeepSeek-R1 用 group_size=64)。

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
from gleamlm.trainer.rl_trainer import compute_reward, grpo_loss, sample_responses
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config
from gleamlm.utils.metrics import emit_metric
from gleamlm.utils.torch_utils import clean_state_dict, safe_autocast


def train(args):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # 当前未实现 DDP (无 init_process_group / 梯度 all-reduce)，多卡直接禁用，
    # 避免各 rank 用不同数据分片静默训练出分歧副本。
    if world_size > 1:
        raise SystemExit("GRPO 未实现 DDP，请单进程运行 (不要用 torchrun)")
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

    # π_ref 必须是训练开始时的快照: 用自己当前参数算 KL 等于自己约束自己；
    # ref 只推理 (autocast + no_grad)，显存开销约 1 份模型。
    ref_model = deepcopy(policy_model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if rank == 0:
        total = sum(p.numel() for p in policy_model.parameters())
        print(f"GRPO — model: {total / 1e6:.2f}M, group_size={args.group_size}, beta={args.beta}")
        os.makedirs(args.output_dir, exist_ok=True)

    # 统一样本流: 主循环取 batch 与零方差重采样补题共用同一迭代器,
    # loader 耗尽自动回绕 (重新 iter → 新 shuffle 顺序)
    loader_iter = iter(loader)
    buffer: list[dict] = []

    def next_item() -> dict:
        nonlocal loader_iter
        while not buffer:
            try:
                buffer.extend(next(loader_iter))
            except StopIteration:
                loader_iter = iter(loader)
                buffer.extend(next(loader_iter))
        return buffer.pop(0)

    total_steps = len(loader) * args.epochs
    trunc_rows = 0  # 1a: 触达上限仍未出 eos 的样本数 (累计)
    gen_rows = 0  # 1a: rollout 样本总数 (累计分母)
    zero_replaced = 0  # 1b: 累计被替换的零方差 prompt 数
    zero_skipped = 0  # 1b: 末轮仍零方差、裁掉不进 loss 的 prompt 数
    zero_skipped_steps = 0  # 1b: 整批无学习信号而被跳过的 step 数
    # 记录窗口: 自上次指标行（log_interval step）以来的 loss/reward 累计。
    # 曲线记窗口均值而非瞬时单 step 值 —— 单 step 采样噪声大, 直接记锯齿化
    log_loss_sum = 0.0
    log_reward_sum = 0.0
    log_steps = 0

    for global_step in range(total_steps):
        batch_items = [next_item() for _ in range(args.batch_size)]
        pending_prompts = [it["prompt"] for it in batch_items]
        pending_gt = [it.get("ground_truth") for it in batch_items]

        # Rollout (+ 零方差组动态重采样): 组内奖励全同 (std≈0) 时优势项无梯度,
        # KL 项仍在把这组样本拉向 ref —— 推理预算白花 (group_size 倍开销)。
        # 1b: 从样本流取新 prompt 替换零方差行 (最多 --max_resample 次);
        # --max_resample 0 = 关闭 (单轮 rollout, 行为同旧版)。
        rewards = torch.zeros(args.batch_size, args.group_size, device=device)
        zero_mask = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
        for attempt in range(args.max_resample + 1):
            prompt_ids = tokenize_prompts(pending_prompts, tokenizer, args.seq_len).to(device)
            prompt_len = prompt_ids.size(1)

            with torch.no_grad():
                # rollout 必须 eval 模式: dropout 会往采样解码注入噪声，
                # 且会让 ref 的 log-prob 分布与推理时不一致
                policy_model.eval()
                gen_seqs, truncated = sample_responses(
                    policy_model,
                    tokenizer,
                    prompt_ids,
                    group_size=args.group_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seq_len=args.seq_len,
                )
                policy_model.train()

            # reward 只针对回答部分打分 (prompt 是固定条件，不计入)。
            for g_idx in range(args.group_size):
                resp_ids = gen_seqs[g_idx][:, prompt_len:]
                decoded = [
                    tokenizer.decode(resp_ids[i].tolist(), skip_special=True)
                    for i in range(len(pending_prompts))
                ]
                rewards[:, g_idx] = torch.tensor(
                    [compute_reward(d, gt) for d, gt in zip(decoded, pending_gt, strict=True)],
                    device=device,
                    dtype=torch.float,
                )

            # 1a: 截断响应不给正奖励 —— 半截文本碰巧包含 ground_truth 会拿
            # +1.0; clamp(max=0) 只惩罚不受益 (负奖励保留)。逐样本掩码:
            # 同列内未截断样本的奖励不受影响。修整须在零方差判定之前:
            # "全组 +1 实为截断假命中" 也要被识别为零方差。
            for g_idx, trunc_mask in enumerate(truncated):
                if trunc_mask.any():
                    rewards[:, g_idx] = torch.where(
                        trunc_mask,
                        rewards[:, g_idx].clamp(max=0.0),
                        rewards[:, g_idx],
                    )

            zero_mask = rewards.std(dim=-1) < 1e-6
            if attempt == args.max_resample or not zero_mask.any():
                break
            # 替换零方差行: 旧题无优势信号不再使用, 从样本流取新题
            for i in zero_mask.nonzero().flatten().tolist():
                item = next_item()
                pending_prompts[i] = item["prompt"]
                pending_gt[i] = item.get("ground_truth")
                zero_replaced += 1

        trunc_rows += sum(int(t.sum().item()) for t in truncated)
        gen_rows += args.group_size * args.batch_size

        # 1b: 末轮仍零方差的行不进 loss (纯 KL 拉扯无学习价值); 整批全零 →
        # 本 step 无任何有效样本, 跳过更新。旧版行为 (--max_resample 0) 不裁剪。
        if args.max_resample > 0 and zero_mask.any():
            zero_skipped += int(zero_mask.sum())
            keep = (~zero_mask).nonzero().flatten().to(device)
            if keep.numel() == 0:
                zero_skipped_steps += 1
                if rank == 0:
                    print(f"{global_step}/{total_steps} [整批零方差, 跳过更新]", flush=True)
                continue
            rewards = rewards[keep]
            gen_seqs = [seqs[keep] for seqs in gen_seqs]

        # 优势必须按 prompt 分组归一化: 不同 prompt 的奖励分布不同，
        # 跨 prompt 归一化会引入噪声；loss 按 group_size 平均累加。
        adv = (rewards - rewards.mean(dim=-1, keepdim=True)) / (
            rewards.std(dim=-1, keepdim=True) + 1e-8
        )

        all_losses = []
        for g_idx in range(args.group_size):
            gen_ids = gen_seqs[g_idx]

            with safe_autocast():
                # loss 前向与 rollout 同一分布: eval 模式算 log-prob，
                # 否则 dropout 会让重要性比在随机函数上计算
                # (当前 config dropout=0.0，此改动为理论一致性)
                policy_model.eval()
                p_logits, _, _, _ = policy_model(gen_ids)
                policy_model.train()
                with torch.no_grad():
                    r_logits, _, _, _ = ref_model(gen_ids)

            loss = grpo_loss(
                p_logits,
                r_logits,
                gen_ids,
                prompt_len,
                adv[:, g_idx],
                beta=args.beta,
            )
            all_losses.append(loss / args.group_size)

        total_loss = torch.stack(all_losses).sum()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad()
        log_loss_sum += total_loss.item()
        log_reward_sum += rewards.mean().item()
        log_steps += 1

        if rank == 0 and global_step % args.log_interval == args.log_interval - 1:
            window_loss = log_loss_sum / max(log_steps, 1)
            window_reward = log_reward_sum / max(log_steps, 1)
            # WebUI 面板可解析进度行 (与 sft.py tqdm postfix 同构); flush 保证实时
            print(
                f"{global_step}/{total_steps} [loss={window_loss:.4f}, "
                f"lr={args.lr:.2e}, trunc={trunc_rows}/{gen_rows}]",
                flush=True,
            )
            # 哨兵指标行（契约见 gleamlm/utils/metrics.py）: 面板优先消费,
            # 不再依赖手工帧格式; reward 为组内平均奖励（§7.3 的 GRPO 监控量,
            # 与优势同源, 无额外前向开销）。loss/reward 记窗口均值
            emit_metric(
                split="train",
                step=global_step,
                total=total_steps,
                loss=window_loss,
                lr=args.lr,
                reward=window_reward,
            )
            log_loss_sum = 0.0
            log_reward_sum = 0.0
            log_steps = 0

    if rank == 0:
        # 1a/1b 累计统计 (方案对齐: trunc=n/N 与 zero_var replaced/kept)
        print(
            f"完成: {total_steps} steps, zero_var: replaced={zero_replaced}, "
            f"skipped={zero_skipped} (整批跳过 {zero_skipped_steps} steps), "
            f"trunc={trunc_rows}/{gen_rows}",
            flush=True,
        )
        torch.save(
            {
                "model_state_dict": policy_model.state_dict(),
                "_config": extract_checkpoint_config(ckpt),
                "step": global_step + 1,  # 已完成步数 (与旧版保存值一致)
            },
            os.path.join(args.output_dir, "grpo_final.pt"),
        )


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM GRPO alignment")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./checkpoints/grpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="rollout 采样温度 (0 = 贪心, 组内轨迹将完全相同)",
    )
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument(
        "--max_resample",
        type=int,
        default=2,
        help="零方差组最大重采样次数 (0 = 关闭, 行为同旧版)",
    )
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tokenizer_path", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    from gleamlm.utils.logging_utils import setup_cli_logging

    setup_cli_logging()
    args = parse_args()
    train(args)

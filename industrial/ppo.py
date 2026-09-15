"""
PPO / RLOO — TRL 工业版 RLHF 对齐。

PPO 的五个核心概念（TRL 1.x 中 PPO 已被 RLOO 替代，原理不变）：
  1. Policy clip：r(θ) = π_θ/π_old 限制在 [1-ε, 1+ε]，防止一步崩坏
  2. Advantage Estimation：RLOO 用 leave-one-out baseline，比 PPO 的 GAE 更简单
  3. KL penalty：防止 π_θ 偏离 π_ref 太远（β 系数控制）
  4. Mini-batch updates：一个 rollout batch 分多个 mini-batch 更新
  5. Reward function：工业上 reward 来自 Reward Model

TRL 1.x 变化:
  - TRL 0.x 时代：PPOTrainer + ValueHead → 标准 PPO
  - TRL 1.0+：PPO 被移除，推荐 RLOO 和 GRPO（DeepSeek-R1 同款）
  - RLOO vs PPO：去掉 Value Network，用 leave-one-out baseline → 更简单、更稳定
  - TRL 1.x 不再支持 PPOTrainer：工业界 RLHF 演进方向是 GRPO 和 RLOO，
    两者都不需要 Value Network，训练更稳定

对比手动版 (manual/ppo.py)：
  - 手动版：手写 ValueHead、GAE、clip loss、entropy bonus、old policy sync
  - 工业版：RLOOConfig + RLOOTrainer 一行 → 聚焦 reward 设计
用法:
  # 0.6B: SFT/DPO 产物 → RLOO 强化对齐
  python industrial/ppo.py \
    --model_path checkpoints/0.6b/sft_lora_hf \
    --data_path data/0.6b/rlhf.jsonl \
    --output_dir checkpoints/0.6b/ppo_hf \
    --tokenizer_path checkpoints/bbpe_24k/hf_export

  # 多卡
  accelerate launch industrial/ppo.py \
    --model_path checkpoints/0.6b/sft_lora_hf \
    --data_path data/0.6b/rlhf.jsonl \
    --output_dir checkpoints/0.6b/ppo_hf \
    --tokenizer_path checkpoints/bbpe_24k/hf_export

数据格式 (JSONL，推荐带 ground_truth 做规则 reward):
  {"prompt": "请解释质能方程"}
  {"prompt": "2+2=?", "ground_truth": "4"}

守门 (与 manual/grpo.py 1a/1b 同口径, 见 industrial/rl_reward.py):
  - 1a 截断奖励守卫: 未以 eos 收尾的回答 clamp(max=0) —— 半截文本碰巧
    包含 ground_truth 不再拿 +1.0, 只惩罚不受益
  - 1b 零方差审计: 训练结束打印零方差组占比; RLOO rollout 内嵌于 trainer,
    无法像 manual 轨动态重采样 (组内全同 → leave-one-out 优势≈0)
  - 奖励口径: 有 ground_truth 规则匹配 (+1/0/-1); 无 gt 启发式分级
    (与 manual 轨 compute_reward 同口径), 不再是"非空全 +1.0"的常数兜底
  - 过易题预剔除用 data_tools/rl/filter_by_difficulty.py（零方差根治）
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import RLOOConfig, RLOOTrainer

from gleamlm.utils.config import extract_checkpoint_config
from hf.hf_config import GleamLMConfig, gleamlm_config_from_core
from hf.hf_model import GleamLMForCausalLM, load_from_checkpoint
from industrial.rl_reward import build_reward_fn


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO/RLOO RLHF for GleamLM (TRL 1.x)")
    parser.add_argument(
        "--model_path", type=str, required=True, help="GleamLM checkpoint (.pt) or HF model dir"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Directory with config.json (if non-.pt model)",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="RLHF queries (JSONL: {prompt/instruction})"
    )
    parser.add_argument("--output_dir", type=str, default="./ppo_out")
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="HF-format tokenizer dir (tokenizer.json)"
    )
    # RLHF 核心超参
    parser.add_argument("--lr", type=float, default=1e-6, help="RLOO learning rate (通常 1e-6)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument(
        "--num_generations", type=int, default=4, help="每个 prompt 采样数（RLOO 的 group_size）"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1, help="KL penalty 系数 (PPO/RLOO: 推荐 0.01-0.1)"
    )
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 模型加载 ──
    if args.model_path.endswith(".pt"):
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
        if args.config_path:
            config = GleamLMConfig.from_pretrained(args.config_path)
        else:
            config = gleamlm_config_from_core(extract_checkpoint_config(ckpt))
        model = GleamLMForCausalLM(config)
        missing, unexpected = load_from_checkpoint(model, ckpt)
        if missing or unexpected:
            print(f"[warn] ppo load — missing={len(missing)} unexpected={len(unexpected)}")
        # TRL 需要从 config._name_or_path 重建 ref 模型；.pt 构造的模型没有该字段，
        # 导出为本地 HF 目录作为 ref 来源
        model.save_pretrained(args.output_dir)
        model.config._name_or_path = args.output_dir
    else:
        model = GleamLMForCausalLM.from_pretrained(args.model_path)

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1a/1b 奖励守卫 (口径见 industrial/rl_reward.py): 截断回答 clamp(max=0),
    # 零方差组审计; eos id 缺失 (非 BBPE 导出词表) 时 1a 自动停用
    reward_fn, reward_stats = build_reward_fn(tokenizer.eos_token_id, args.num_generations)
    if tokenizer.eos_token_id is None:
        print("[warn] tokenizer.eos_token_id 未定义 — 1a 截断守卫停用")

    # ── 数据集 ──
    raw_data = load_jsonl(args.data_path)
    for item in raw_data:
        if "prompt" not in item and "instruction" in item:
            item["prompt"] = item.pop("instruction")
    dataset = Dataset.from_list(raw_data)
    no_gt = sum(1 for item in raw_data if not str(item.get("ground_truth") or "").strip())
    if no_gt:
        print(f"数据: 无 ground_truth {no_gt}/{len(raw_data)} 行 — 这些行用启发式 reward")
    if no_gt * 2 > len(raw_data):
        print(
            "WARN: 无 gt 行过半, 规则信号弱 — 建议数据提供 ground_truth, 并先用 "
            "data_tools/rl/filter_by_difficulty.py 剔除过易题 (零方差组无优势梯度)"
        )

    # RLOO 无 ValueHead（leave-one-out baseline），无需调 GAE λ，
    # 参数只有 num_generations + beta
    rloo_config = RLOOConfig(
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        bf16=torch.cuda.is_available(),
        fp16=False,
        output_dir=args.output_dir,
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # RLOO 内部: generate → reward → leave-one-out baseline → clip update
    # （去掉了 ValueHead 和 GAE）
    trainer = RLOOTrainer(
        model=model,
        args=rloo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    # ── 训练 ──
    trainer.train()

    # ── 保存 ──
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"PPO/RLOO model saved to {args.output_dir}")
    print(reward_stats.summary())


if __name__ == "__main__":
    main()

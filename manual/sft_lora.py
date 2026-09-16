"""
LoRA SFT 微调 — 冻结预训练权重，只更新低秩 adapter。

用法:
  python manual/sft_lora.py --variant nano \\
    --model checkpoints/nano/final.pt \\
    --output_dir checkpoints/nano/lora
  (超参默认取 manual/configs/nano.yaml 的 lora 段, CLI 同名参数可覆写;
   --model 为预训练基座, 数据默认 data/nano/sft/sft_mix.jsonl)
"""

import argparse
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.trainer.base_trainer import create_scaler, optimizer_step
from gleamlm.trainer.lora import LoraConfig, apply_lora_to_model, merge_lora_weights
from gleamlm.trainer.schedulers import get_lr_cosine, get_lr_wsd
from gleamlm.utils.chatml import format_chatml
from gleamlm.utils.config import (
    DEFAULT_TOKENIZER_PATH,
    extract_checkpoint_config,
    load_config,
)
from gleamlm.utils.metrics import emit_metric
from gleamlm.utils.torch_utils import clean_state_dict, safe_autocast

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """统一 ChatML：兼容单轮 {instruction,output} 与多轮 {messages}（与 core SFT 同语义）。

    返回 (prompt_text, resp_text)：prompt 含到 assistant 头为止的历史+指令，
    resp 为最后一条 assistant 内容；collate 里 assistant 起始之前全部 -100。

    加载期传入 tokenizer 时预检剔除「prompt 过长导致回答被保头截断切掉」的
    病理样本 (label 全 -100), 统计见 self.precheck_stats。
    """

    def __init__(self, data_path: str, max_seq_len: int = 1024, tokenizer=None):
        self.max_seq_len = max_seq_len
        self.data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "messages" in item:
                    msgs = item["messages"]
                    if not msgs or msgs[-1].get("role") != "assistant":
                        continue
                    history = msgs[:-1]
                    prompt_text = (
                        format_chatml(history, add_generation_prompt=True) if history else ""
                    )
                    resp_text = msgs[-1]["content"] + "<|im_end|>"
                else:
                    prompt_text = format_chatml(
                        [{"role": "user", "content": item.get("instruction", "")}],
                        add_generation_prompt=True,
                    )
                    resp_text = item.get("output", "") + "<|im_end|>"
                if not prompt_text.strip() or not resp_text.strip():
                    continue
                self.data.append((prompt_text, resp_text))

        # 加载期预检: collate 保头截断 (ids[:max_seq_len]) 后, prompt token 数
        # >= max_seq_len 的样本 label 全 -100 (回答被整段切掉) → CE 无有效
        # token (nan) 白训, 直接剔除; 不传 tokenizer 时跳过 (兼容旧调用)
        self.precheck_stats: dict = {}
        if tokenizer is not None:
            kept: list[tuple[str, str]] = []
            dropped_idx: list[int] = []
            for i, (prompt_text, _resp) in enumerate(self.data):
                p_len = len(tokenizer.encode(prompt_text, add_bos=True))
                # collate 里 label = [-100] * (p_len - 1) + ids[p_len - 1:] 且 ids
                # 已截到 max_seq_len; p_len - 1 >= max_seq_len 时 label 全 -100
                if p_len - 1 >= max_seq_len:
                    dropped_idx.append(i)
                    continue
                kept.append(self.data[i])
            self.precheck_stats = {
                "total": len(self.data),
                "ok": len(kept),
                "all_masked": len(dropped_idx),
            }
            if dropped_idx:
                logger.warning(
                    f"precheck 剔除 {len(dropped_idx)} 条病理样本 (prompt 过长, "
                    f"保头截断后回答归零); 样例索引: {dropped_idx[:3]}"
                )
            if not kept:
                raise ValueError("precheck 后无健康样本: 检查 prompt 长度与 seq_len 设置")
            self.data = kept

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer, max_seq_len):
    prompts, responses = list(zip(*batch, strict=True))
    input_ids, labels_list = [], []
    for p, r in zip(prompts, responses, strict=True):
        ids = tokenizer.encode(p + r, add_bos=True)
        ids = ids[:max_seq_len]
        p_len = len(tokenizer.encode(p, add_bos=True))
        label = [-100] * (p_len - 1) + ids[p_len - 1 :]
        label = label[:max_seq_len]
        if len(label) < len(ids):
            label = label + [-100] * (len(ids) - len(label))
        input_ids.append(ids)
        labels_list.append(label)
    max_len = max(len(x) for x in input_ids)
    pad_id = tokenizer.pad_id
    input_ids = [x + [pad_id] * (max_len - len(x)) for x in input_ids]
    labels_list = [x + [-100] * (max_len - len(x)) for x in labels_list]
    return torch.tensor(input_ids), torch.tensor(labels_list)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BBPETokenizer.load(args.tokenizer_path or DEFAULT_TOKENIZER_PATH)

    dataset = SFTDataset(args.data, max_seq_len=args.seq_len, tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.seq_len),
    )

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = extract_checkpoint_config(ckpt)
    model = GleamLMModel(
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
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=False)
    model.train()

    # attention + FFN 全覆盖: FFN 是知识载体 (ADR-0003), 仅挂 Q/K/V/O 时
    # 知识型 SFT 数据学不动, loss 快速平台化 (~3.0, 全参 SFT 可至 ~2.34)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"],
    )
    apply_lora_to_model(model, lora_cfg)
    # LoRA 语义: base 全冻结, 仅训练 adapter。apply_lora_to_model 只冻结被替换的
    # 投影层 (Q/K/V/O + FFN), 其余层 (embedding/norm) 仍是 requires_grad=True —— 不显式再冻
    # 会让 optimizer 拿多余参数, 退化成"部分全参微调"(早期 QKVO: 66.4M vs 纯 adapter ~0.5M,
    # loss 趋势平且震荡大)。
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    lora_count = sum(p.numel() for p in lora_params)

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    scaler = create_scaler()

    print(
        f"LoRA — base: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, trainable: {lora_count / 1e3:.1f}K, r={args.lora_r}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # x 轴步数跨 epoch 单调递增 (面板按 step 去重, 归零会使后段点全部被丢);
    # 进度行与 sft.py tqdm postfix 同构 (N/M [... loss=.., lr=..]), WebUI 解析器可识别;
    # flush=True 保证管道/重定向下实时到达 (不 flush 会块缓冲延迟, 面板实时曲线缺失)。
    # step = 优化器步: 梯度累积后 global_step 才 +1 (同 sft.py 语义)。
    total_steps = math.ceil(len(loader) / args.accumulate_grad) * args.epochs
    # LR 衰减视野与实际训练步数解耦 (lr_decay_steps 独立于 total_steps):
    # None 时与 total_steps 一致 (行为不变); 设值时 lr 调度按该视野算,
    # 面板进度/指标 total 仍用实际 total_steps
    decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else total_steps
    if decay_steps != total_steps:
        print(f"Steps: {total_steps} (lr_decay_steps: {decay_steps}) — lr 调度按 decay 视野走")
    global_step = 0
    skipped_batches = 0
    # 记录窗口: 自上次指标行以来的微批原始 loss 累计与批数。
    # 记窗口均值而非瞬时单批值 —— 单批采样噪声大, 曲线会锯齿化
    win_loss_sum = 0.0
    win_batches = 0
    for _ in range(args.epochs):
        for batch_idx, (input_ids, labels) in enumerate(loader):
            input_ids, labels = input_ids.to(device), labels.to(device)

            # 防线 (同 core SFT): 全 mask 批跳过, 防 CE nan 污染权重
            if int((labels != -100).sum()) == 0:
                skipped_batches += 1
                continue
            # AMP 同 SFT 轨 (bf16 autocast): matmul 走 tensor core，kernel 时长短、
            # 显存减半。Windows WDDM 下纯 fp32 长 kernel 易致驱动挂起甚至蓝屏。
            with safe_autocast():
                logits, _, aux_loss, _ = model(input_ids)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                loss = loss + aux_loss * 0.01

            is_accum = (batch_idx + 1) % args.accumulate_grad == 0 or (batch_idx + 1) == len(loader)
            # 残差批 (末尾不足 accumulate) 按实际批数除, 避免归一化过头 (同 sft.py)
            denom = (
                ((batch_idx % args.accumulate_grad) + 1)
                if (batch_idx + 1) == len(loader)
                else args.accumulate_grad
            )
            loss = loss / denom
            scaler.scale(loss).backward()
            win_loss_sum += loss.item() * denom  # 还原缩放, 微批原始 loss
            win_batches += 1
            if is_accum:
                # lr 调度在 step 前更新 (与 sft.py 同构): warmup → cosine/wsd 衰减,
                # 替代原恒定 lr (无衰减后期难收敛)
                if args.lr_scheduler == "wsd":
                    lr_mult = get_lr_wsd(
                        global_step,
                        decay_steps,
                        args.warmup_ratio,
                        args.stable_ratio,
                        args.min_lr_ratio,
                    )
                else:
                    lr_mult = get_lr_cosine(
                        global_step, decay_steps, args.warmup_ratio, args.min_lr_ratio
                    )
                cur_lr = args.lr * lr_mult
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
                last_grad_norm = optimizer_step(
                    optimizer, scaler, parameters=lora_params, clip_grad=args.clip
                )
                global_step += 1

                if global_step == 1 or global_step % args.log_interval == 0:
                    window_loss = win_loss_sum / win_batches
                    print(
                        f"{global_step}/{total_steps} [loss={window_loss:.4f}, lr={cur_lr:.2e}]",
                        flush=True,
                    )
                    # 哨兵指标行（契约见 gleamlm/utils/metrics.py）: 面板优先消费。
                    # loss 记窗口均值（自上次记录至今的全部微批平均）
                    emit_metric(
                        split="train",
                        step=global_step,
                        total=total_steps,
                        loss=window_loss,
                        lr=cur_lr,
                        grad_norm=last_grad_norm,
                    )
                    win_loss_sum = 0.0
                    win_batches = 0

    if skipped_batches:
        print(f"跳过 {skipped_batches} 个全 mask 批 (无监督 token)")

    save_path = os.path.join(args.output_dir, "lora.pt")
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(
        {
            "lora": lora_state,
            "_config": cfg,
            "lora_config": {"r": args.lora_r, "alpha": args.lora_alpha},
        },
        save_path,
    )
    print(f"LoRA weights saved: {save_path}")

    if args.merge:
        merge_lora_weights(model)
        full_path = os.path.join(args.output_dir, "merged.pt")
        torch.save({"model_state_dict": model.state_dict(), "_config": cfg}, full_path)
        print(f"Merged model saved: {full_path}")


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM LoRA SFT")
    p.add_argument(
        "--variant",
        type=str,
        required=True,
        help="配置模板名 (读 {config_dir}/{variant}.yaml 的 lora 段默认值)",
    )
    p.add_argument(
        "--config_dir",
        type=str,
        default=os.path.join(_ROOT_DIR, "manual", "configs"),
        help="YAML 配置目录 (manual 轨专用)",
    )
    p.add_argument("--model", type=str, required=True, help="预训练基座 checkpoint")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="LoRA adapter 保存目录 (未传取 data.checkpoint_dir/lora, 落变体目录内)",
    )
    # ── 实验/方案级超参: 默认权威在 YAML lora 段 (default=None + 裁决, 无第二权威) ──
    p.add_argument(
        "--epochs", type=int, default=None, help="覆写训练轮数 (默认取 YAML lora.epochs)"
    )
    p.add_argument("--batch_size", type=int, default=None, help="覆写 batch size")
    p.add_argument(
        "--accumulate_grad",
        type=int,
        default=None,
        help="覆写梯度累积步数 (默认取 YAML lora.accumulate_grad)",
    )
    p.add_argument(
        "--seq_len", type=int, default=None, help="覆写序列长度 (默认取 YAML lora.max_seq_len)"
    )
    p.add_argument("--lr", type=float, default=None, help="覆写学习率 (默认取 YAML lora.lr)")
    p.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["cosine", "wsd"],
        default=None,
        help="覆写学习率调度器 (默认取 YAML lora.lr_scheduler)",
    )
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="覆写 warmup 比例 (默认取 YAML lora.warmup_ratio)",
    )
    p.add_argument(
        "--stable_ratio",
        type=float,
        default=None,
        help="覆写 WSD stable 比例 (cosine 时忽略, 默认取 YAML)",
    )
    p.add_argument(
        "--min_lr_ratio",
        type=float,
        default=None,
        help="覆写 lr 终点比例 (默认取 YAML lora.min_lr_ratio)",
    )
    p.add_argument(
        "--lr_decay_steps",
        type=int,
        default=None,
        help="LR 衰减视野步数 (默认取 YAML lora.lr_decay_steps; 未设 = 跟随实际总步数)",
    )
    p.add_argument(
        "--clip", type=float, default=None, help="覆写梯度裁剪 (默认取 YAML lora.clip_grad)"
    )
    p.add_argument("--lora_r", type=int, default=None, help="覆写 LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=None, help="覆写 LoRA alpha")
    p.add_argument("--log_interval", type=int, default=None, help="覆写日志间隔")
    p.add_argument(
        "--data", type=str, default=None, help="JSONL SFT 数据 (未传回落 YAML lora.data_path)"
    )
    p.add_argument("--tokenizer_path", type=str, default="")
    p.add_argument("--merge", action="store_true", help="训练后合并 LoRA 权重到 base")
    args = p.parse_args()

    # ── 单轨裁决: YAML lora 段为默认权威; CLI 显式传才覆写 ──
    config_path = os.path.join(args.config_dir, f"{args.variant}.yaml")
    if not os.path.isfile(config_path):
        raise SystemExit(
            f"配置不存在: {config_path} (--variant 指定配置模板名, --config_dir 指定目录)"
        )
    cfg = load_config(config_path, _ROOT_DIR, scope="lora")
    # 保存目录对齐 sft/dpo: 默认落变体根目录内 (checkpoints/<variant>/lora)
    args.output_dir = args.output_dir or os.path.join(cfg.data.checkpoint_dir, "lora")
    for _cli, _key in (("data", "data_path"), ("seq_len", "max_seq_len"), ("clip", "clip_grad")):
        if getattr(args, _cli) is None:
            setattr(args, _cli, getattr(cfg.lora, _key))
    for _key in (
        "epochs",
        "batch_size",
        "accumulate_grad",
        "lr",
        "lr_scheduler",
        "lr_decay_steps",
        "warmup_ratio",
        "stable_ratio",
        "min_lr_ratio",
        "lora_r",
        "lora_alpha",
        "log_interval",
    ):
        if getattr(args, _key) is None:
            setattr(args, _key, getattr(cfg.lora, _key))
    if not args.data:
        p.error("缺少 LoRA 数据: 传 --data 或在 YAML 配置 lora.data_path")
    return args


if __name__ == "__main__":
    from gleamlm.utils.logging_utils import setup_cli_logging

    setup_cli_logging()
    args = parse_args()
    train(args)

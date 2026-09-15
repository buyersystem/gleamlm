"""难度过滤 — 剔除"已频繁解出"的 prompt, 把采样预算留给有学习信号的题。

用 SFT checkpoint 对每题采样 K 次 (temperature 与 GRPO 训练一致), 按规则奖励计算
pass_rate; pass_rate 过高 (默认 > 0.9) 的题在 GRPO 中几乎必然全对 →
组内零方差 → 无优势梯度, 直接剔除以省推理预算。

采样逻辑与 GRPO rollout 共用 gleamlm/trainer/rl_trainer.sample_responses,
避免两套采样代码漂移; 产物旁写 manifest (data_tools/shared/audit.py)。

用法:
  python data_tools/rl/filter_by_difficulty.py \
      --model checkpoints/nano/sft/sft_best.pt \
      --data data/rlhf.jsonl --output data/rlhf_filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_sys_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

import torch

from data_tools.shared.audit import write_manifest
from gleamlm.data.rl_data import RLHFDataset, tokenize_prompts
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.trainer.rl_trainer import compute_reward, sample_responses
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config
from gleamlm.utils.torch_utils import clean_state_dict


def main() -> None:
    p = argparse.ArgumentParser(description="按难度过滤 RLHF prompt 池 (pass_rate 过滤)")
    p.add_argument("--model", required=True, help="SFT checkpoint (与 GRPO 起点一致)")
    p.add_argument("--data", required=True, help="prompt 池 jsonl (RLHFDataset 格式)")
    p.add_argument(
        "--output",
        default="",
        help="输出 jsonl (缺省 <data 去扩展名>_filtered.jsonl)",
    )
    p.add_argument("--samples", type=int, default=8, help="每题采样次数 K")
    p.add_argument("--threshold", type=float, default=0.9, help="pass_rate 超过该值即剔除 (已解出)")
    p.add_argument("--temperature", type=float, default=0.8, help="与 GRPO rollout 一致")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--tokenizer_path", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BBPETokenizer.load(args.tokenizer_path or DEFAULT_TOKENIZER_PATH)

    dataset = RLHFDataset(args.data, max_seq_len=args.seq_len)
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
    model.eval()

    print(
        f"filter_by_difficulty — model: {args.model}, 题数: {len(dataset)}, "
        f"K={args.samples}, threshold={args.threshold}"
    )

    kept: list[dict] = []
    dropped = 0
    no_gt = 0
    pass_rates: list[float] = []
    for i in range(len(dataset)):
        item = dataset[i]
        gt = item.get("ground_truth")
        if gt is None:
            # 无 ground_truth 无法规则判定对错 → 保留 (不参与难度过滤)
            kept.append({"prompt": item["prompt"], "ground_truth": None})
            no_gt += 1
            print(f"[{i + 1}/{len(dataset)}] 无 gt, 保留 (kept={len(kept)}, dropped={dropped})")
            continue

        prompt_ids = tokenize_prompts([item["prompt"]], tokenizer, args.seq_len).to(device)
        prompt_len = prompt_ids.size(1)
        with torch.no_grad():
            gen_seqs, truncated = sample_responses(
                model,
                tokenizer,
                prompt_ids,
                group_size=args.samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seq_len=args.seq_len,
            )
        n_pass = 0
        for g_idx in range(args.samples):
            if truncated[g_idx][0].item():
                # 1a 同款口径: 截断未出 eos 的回答在 GRPO 中不给正奖励,
                # 此处也不计入 pass (否则 pass_rate 虚高、误剔题目)
                continue
            text = tokenizer.decode(gen_seqs[g_idx][0, prompt_len:].tolist(), skip_special=True)
            if compute_reward(text, gt) > 0:
                n_pass += 1
        rate = n_pass / args.samples
        pass_rates.append(rate)
        if rate > args.threshold:
            dropped += 1
        else:
            kept.append({"prompt": item["prompt"], "ground_truth": gt})
        print(f"[{i + 1}/{len(dataset)}] pass_rate={rate:.2f} kept={len(kept)} dropped={dropped}")

    out = args.output or (os.path.splitext(args.data)[0] + "_filtered.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if pass_rates:
        mean = sum(pass_rates) / len(pass_rates)
        print(
            f"pass_rate: mean={mean:.2f} min={min(pass_rates):.2f} max={max(pass_rates):.2f} "
            f"(有 gt 题 {len(pass_rates)} 道)"
        )
    print(f"完成: 保留 {len(kept)} (无 gt {no_gt}) / 剔除 {dropped} / 共 {len(dataset)}")

    write_manifest(
        out,
        tool="filter_by_difficulty",
        inputs=[args.data],
        dropped={"already_solved": dropped},
        params={
            "model": args.model,
            "samples": args.samples,
            "threshold": args.threshold,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
        seed=None,
    )


if __name__ == "__main__":
    main()

"""SFT/DPO 数据切分：单 JSONL → train + val（固定 seed 可复现）。

背景：后训练 SFT/DPO 原先单文件训练，面板只有 train loss —— 无法判别过拟合。
K8 给训练脚本补 held-out val 评估（emit_metric split="val"），数据侧需要切分：
本脚本把单文件切为 {stem}_train.jsonl + {stem}_val.jsonl，**原文件保留不动**
（可回溯；mix_sft / DPO 生成链重跑后，对再生成的单文件重跑本脚本即可）。

切分策略:
  - seed 全量置换后取前 N 段为 val；train 恢复原行序（首行类型语义不变 ——
    SFTDataset 以首行决定单/双轮模式）
  - 行级切分（不解析 JSON），只剔除空行；train/val 直接写原始行文本（保真）
  - 防泄漏铁律：训练侧 data_path 须改指 {stem}_train.jsonl（YAML 已同步）

产物旁各写 manifest（沿用 audit.write_manifest 惯例）。

用法:
  python data_tools/sft/split_val.py --input data/lite/sft/sft_mix.jsonl --ratio 0.05
  python data_tools/sft/split_val.py --input data/lite/dpo/dpo_data.jsonl --ratio 0.05
"""

import argparse
import os
import random
import sys

_sys_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

from data_tools.shared.audit import write_manifest


def main():
    p = argparse.ArgumentParser(description="单 JSONL → train + val（固定 seed）")
    p.add_argument("--input", required=True, help="源 JSONL（保留不动）")
    p.add_argument("--ratio", type=float, default=0.05, help="val 占比 (默认 0.05)")
    p.add_argument("--seed", type=int, default=42, help="置换种子 (默认 42)")
    args = p.parse_args()

    if not 0 < args.ratio < 1:
        raise SystemExit(f"--ratio 须在 (0,1) 内: {args.ratio}")

    with open(args.input, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    n = len(lines)
    if n == 0:
        raise SystemExit(f"空文件: {args.input}")

    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_val = max(1, round(n * args.ratio))
    val_idx = set(idx[:n_val])

    stem = args.input[: -len(".jsonl")] if args.input.endswith(".jsonl") else args.input
    out_train, out_val = f"{stem}_train.jsonl", f"{stem}_val.jsonl"

    # newline="\n" 显式锁 LF：数据文件统一行尾，避免 Windows 下写出 CRLF
    n_train = 0
    with (
        open(out_train, "w", encoding="utf-8", newline="\n") as ft,
        open(out_val, "w", encoding="utf-8", newline="\n") as fv,
    ):
        for i, line in enumerate(lines):
            if not line.endswith("\n"):
                line += "\n"
            if i in val_idx:
                fv.write(line)
            else:
                ft.write(line)
                n_train += 1

    # 守恒自检（无交集由构造保证；断言防未来回归）
    assert n_train + n_val == n, "行数不守恒"
    print(f"输入 {n} 条 → train {n_train} + val {n_val} (ratio={args.ratio}, seed={args.seed})")
    print(f"  train: {out_train}")
    print(f"  val:   {out_val}")

    for out, role in ((out_train, "train"), (out_val, "val")):
        write_manifest(
            out,
            tool="split_val",
            inputs=[args.input],
            params={"ratio": args.ratio, "role": role},
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

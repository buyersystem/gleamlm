"""SFT 数据健康度 lint — 训练前跑一次, 明确数据健康度（方案 2 precheck 的可视化包装）。

调用训练侧 SFTDataset 的加载期 precheck（单一实现, 不复制截断/掩码逻辑）:
实例化即全量扫描, 输出 token 长度分布 + 病理计数（prompt_lost / all_masked /
truncated）, 并写 {data}.lint.json（含输入文件 sha256 指纹）。

病理定义（见 gleamlm/data/sft_data.py _precheck）:
  - prompt_lost: 超长样本保尾截断后 prompt 归零 → 残段上下文参与 loss, 训练污染
  - all_masked:  labels 全 -100 → CE 无有效 token, nan 静默白训

用法:
  python data_tools/sft/lint_sft.py --data data/nano/sft/sft_mix.jsonl
  python data_tools/sft/lint_sft.py --data data/lite/sft/sft_mix.jsonl --max-seq-len 2048
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

_sys_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

from data_tools.shared.audit import fingerprint
from gleamlm.data.sft_data import SFTDataset
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="SFT 数据健康度 lint (precheck 包装)")
    p.add_argument("--data", required=True, help="待检查的 SFT JSONL")
    p.add_argument("--max-seq-len", type=int, default=512, help="训练用序列长度口径")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_PATH, help="BBPE tokenizer 路径")
    args = p.parse_args()

    tokenizer = BBPETokenizer.load(args.tokenizer)
    ds = SFTDataset(args.data, tokenizer, max_seq_len=args.max_seq_len)
    stats = ds.precheck_stats

    records, sha = fingerprint(args.data)
    report = {
        "tool": "lint_sft",
        "input": {"path": args.data.replace("\\", "/"), "records": records, "sha256": sha},
        "max_seq_len": args.max_seq_len,
        "tokenizer": args.tokenizer.replace("\\", "/"),
        "precheck": stats,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = f"{args.data}.lint.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    bad = stats["prompt_lost"] + stats["all_masked"]
    lengths = stats["lengths"]
    print("=" * 56)
    print(f"data: {args.data}")
    print(f"max_seq_len={args.max_seq_len} | total={stats['total']} | ok={stats['ok']}")
    print(
        f"病理: prompt_lost={stats['prompt_lost']} all_masked={stats['all_masked']}"
        f" | 超长截断(truncated)={stats['truncated']}"
    )
    print(
        f"token 长度: p50={lengths['p50']} p90={lengths['p90']} "
        f"p99={lengths['p99']} max={lengths['max']}"
    )
    if bad:
        print(f"注意: 训练将剔除 {bad} 条病理样本 (上方 precheck 警告已列出样例索引)")
    else:
        print("PASS: 未发现病理样本")
    print(f"Report: {out}")
    print("=" * 56)


if __name__ == "__main__":
    main()

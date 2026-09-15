"""SFT 数据混合工具：基础集 + QA 集 → 单一训练文件。

背景：QA→SFT 提取（qa_to_sft.py）与基础集（sft_data.jsonl，含 chat_extra 闲聊
与多轮 messages）分开维护，训练入口 SFTDataset 只认单文件 → 此处合并。

混合策略（2026-09 决策）:
  - 基础集全量并入（3,415 条：单轮知识 + 761 多轮 + 169 闲聊），QA 不稀释闲聊信号
  - QA 按 --qa-keep 确定性抽样（qa_sft.jsonl 行序本身是 seed42 随机化产物，
    取前 N 行即随机 N 条）
  - 防重复：QA 的 instruction 若已存在于基础集则跳过（零重复原则，避免权重放大）
  - 顺序：基础集在前（首行必须非 messages，SFTDataset 以首行决定单轮/多轮模式）
  - DataLoader shuffle=True → 文件顺序只影响首行类型，不影响训练分布

产物旁写 {output}.manifest.json（输入指纹 / 剔除计数 / 参数，见 data_tools/shared/audit.py）。

用法:
  python data_tools/sft/mix_sft.py --qa-keep 10000 \
      --output data/nano/sft/sft_mix.jsonl
"""

import argparse
import json
import os
import sys

_sys_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

from data_tools.shared.audit import write_manifest

_QA_SRC = "data/nano/sft/qa_sft.jsonl"


def _load(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description="基础集 + QA → 单训练文件")
    p.add_argument("--base", default="data/nano/sft/sft_data.jsonl", help="基础集（含闲聊/多轮）")
    p.add_argument("--qa", default=_QA_SRC, help="QA→SFT 文件")
    p.add_argument("--qa-keep", type=int, default=10000, help="QA 抽样条数（0=全量）")
    p.add_argument("--output", default="data/nano/sft/sft_mix.jsonl")
    args = p.parse_args()

    base = _load(args.base)
    qa = _load(args.qa)
    if args.qa_keep:
        qa = qa[: args.qa_keep]

    # 防重复（两层）：① QA instruction 已存在于基础集 → 跳过；
    # ② output 全文重复（搬运答案：不同问题同答案）→ 保留第一条
    base_ins = set()
    seen_outs = set()
    for r in base:
        if "instruction" in r:
            base_ins.add(r["instruction"].strip())
            seen_outs.add(r.get("output", "").strip())
        elif "messages" in r:
            for m in r["messages"]:
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    base_ins.add(m["content"].strip())
    dup_ins = 0
    dup_out = 0
    kept = []
    for r in qa:
        ins = r.get("instruction", "").strip()
        out = r.get("output", "").strip()
        if ins in base_ins:
            dup_ins += 1
            continue
        if out in seen_outs:
            dup_out += 1
            continue
        base_ins.add(ins)
        seen_outs.add(out)
        kept.append(r)

    total = len(base) + len(kept)
    first_is_messages = "messages" in base[0]
    print(
        f"Base: {len(base)} | QA sampled: {len(qa)} | dup: instruction={dup_ins}, "
        f"output={dup_out} | kept QA: {len(kept)} | Total: {total}"
    )
    print(f"首行含 messages: {first_is_messages}（须为 False，SFTDataset 以首行定模式）")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in base + kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written: {args.output}")

    write_manifest(
        args.output,
        tool="mix_sft",
        inputs=[args.base, args.qa],
        dropped={"duplicate_instruction": dup_ins, "duplicate_output": dup_out},
        params={"qa_keep": args.qa_keep},
        seed=42,  # qa_sft.jsonl 行序为 seed42 随机化产物, 抽样依赖其可复现性
    )


if __name__ == "__main__":
    main()

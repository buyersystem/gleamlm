"""构建 DPO chosen 池（sft_mix 单源版）。

数据源：data/<variant>/sft/sft_mix.jsonl（与训练轨 sft.data_path 同源），
chosen 取其中的 SFT 高质答案：
  - 单轮 {instruction, output}    确定性抽 --single-n 条（默认 1000，output≤600 字防超长截断）
  - 多轮 {messages}               全量（须 6 轮结构且尾轮为 assistant 答案）

历史：v2 的 chosen 池曾混入 chat_extra 闲聊与 sft_data 基础池——这两类中间
产物已从数据目录删除，chosen 池现只从 sft_mix 抽取。

产出两个中间文件（喂 generate_rejected.py 生成 rejected，merge 后即可删）：
  - data/<variant>/dpo/dpo_chosen_single.jsonl  {instruction, output}
  - data/<variant>/dpo/dpo_chosen_multi.jsonl   {messages}（尾轮为 assistant 答案，脚本内部剥离）

用法:
  python data_tools/dpo/build_dpo_chosen.py --variant nano
  python data_tools/dpo/build_dpo_chosen.py --variant lite --single-n 1000 --seed 42
"""

import argparse
import json
import os
import random
import sys

# 单轮 output 字符上限：防 encode 后超过 DPO max_seq_len（1024 token）导致训练截断损坏
MAX_OUT_CHARS = 600
# 多轮对话轮数固定为 6 条消息（3 轮 user/assistant）
MULTI_TURNS = 6


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build DPO chosen pool from sft_mix")
    parser.add_argument(
        "--variant",
        type=str,
        default="nano",
        help="数据变体: 从 data/<variant>/sft/sft_mix.jsonl 抽取, 产物落 data/<variant>/dpo/",
    )
    parser.add_argument("--single-n", type=int, default=1000, help="单轮抽取数")
    parser.add_argument("--seed", type=int, default=42, help="确定性抽样 seed")
    args = parser.parse_args()

    # ---- 单源加载（sft_mix: 单轮 + 多轮混合，按键分流）----
    mix_path = os.path.join("data", args.variant, "sft", "sft_mix.jsonl")
    if not os.path.isfile(mix_path):
        sys.exit(f"Error: 源数据不存在: {mix_path}（chosen 池数据源）")
    rows = load_jsonl(mix_path)
    singles = [r for r in rows if "instruction" in r and "output" in r]
    # instruction 去重（保第一条）
    seen: set[str] = set()
    unique = []
    for r in singles:
        ins = r["instruction"]
        if ins not in seen:
            seen.add(ins)
            unique.append(r)
    print(f"sft_mix 单轮: {len(unique)}/{len(singles)} 条（去重后）")

    # 优先 output <= MAX_OUT_CHARS（防训练截断），不足时放宽
    short = [r for r in unique if len(r["output"]) <= MAX_OUT_CHARS]
    long = [r for r in unique if len(r["output"]) > MAX_OUT_CHARS]
    print(f"  其中 output<=600 字: {len(short)} 条, >600 字: {len(long)} 条")
    rng = random.Random(args.seed)
    if len(short) >= args.single_n:
        picked = rng.sample(short, args.single_n)
    else:
        picked = short + rng.sample(long, args.single_n - len(short))
        print(f"  !! 短答案池不足，从长答案补抽 {args.single_n - len(short)} 条（有截断风险）")
    print(f"单轮抽取: {len(picked)} 条")

    # ---- 多轮源：messages 全量（尾轮须为 assistant 答案）----
    multis = [r for r in rows if "messages" in r]
    valid_multi = []
    for r in multis:
        msgs = r["messages"]
        if len(msgs) != MULTI_TURNS or not msgs or msgs[-1].get("role") != "assistant":
            continue
        valid_multi.append(r)
    print(f"多轮: {len(valid_multi)}/{len(multis)} 条（6 轮结构）")

    # ---- 写中间文件（产物目录按 variant 推导）----
    out_dir = os.path.join("data", args.variant, "dpo")
    os.makedirs(out_dir, exist_ok=True)
    single_out = os.path.join(out_dir, "dpo_chosen_single.jsonl")
    multi_out = os.path.join(out_dir, "dpo_chosen_multi.jsonl")
    with open(single_out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(
                json.dumps(
                    {"instruction": r["instruction"], "output": r["output"]}, ensure_ascii=False
                )
                + "\n"
            )
    with open(multi_out, "w", encoding="utf-8") as f:
        for r in valid_multi:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    total = len(picked) + len(valid_multi)
    print(
        f"Done: single={len(picked)}, multi={len(valid_multi)}, 合计 {total} 对 "
        f"-> {single_out} / {multi_out}"
    )


if __name__ == "__main__":
    main()

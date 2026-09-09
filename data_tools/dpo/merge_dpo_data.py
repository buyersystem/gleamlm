"""合并 generate_rejected.py 分片产物 → dpo_data.jsonl，含基础清洗。

- single 分片: dpo_rejected_single[.N].jsonl → {instruction, chosen, rejected}
- multi 分片:  dpo_rejected_multi[.N].jsonl  → {messages(不含尾轮答案), chosen, rejected}

清洗规则（剔除训练毒样本）：
  1. rejected 为空 / < 8 字 / 与 chosen 完全相同 / chosen 长度<4
  2. 输出文件按 chosen 顺序保持与 chosen 池一致（供人工复核）

产物目录按 variant 推导（data/<variant>/dpo/dpo_data.jsonl，与训练轨
dpo.data_path 指向一致，重建后即生效）。

用法:
  python data_tools/dpo/merge_dpo_data.py --variant nano
  python data_tools/dpo/merge_dpo_data.py --variant lite --output data/lite/dpo/dpo_data_new.jsonl
"""

import argparse
import glob
import json
import os


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Merge rejected shards -> dpo_data.jsonl")
    parser.add_argument(
        "--variant",
        type=str,
        default="nano",
        help="数据变体: 分片 glob 与产物默认落在 data/<variant>/dpo/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件（缺省 data/<variant>/dpo/dpo_data.jsonl，即 dpo.data_path 指向）",
    )
    args = parser.parse_args()

    dpo_dir = os.path.join("data", args.variant, "dpo")
    single_glob = os.path.join(dpo_dir, "dpo_rejected_single*.jsonl")
    multi_glob = os.path.join(dpo_dir, "dpo_rejected_multi*.jsonl")
    out = args.output or os.path.join(dpo_dir, "dpo_data.jsonl")

    # 分片文件名：shard 0 无后缀（dpo_rejected_XXX.jsonl），shard N 为 .N.jsonl，按数字排回原序
    def shard_key(prefix: str):
        def key(p: str) -> int:
            base = os.path.basename(p)
            if base.endswith(".jsonl") and base != prefix + ".jsonl":
                return int(base.replace(prefix + ".", "").replace(".jsonl", ""))
            return 0

        return key

    single_files = sorted(glob.glob(single_glob), key=shard_key("dpo_rejected_single"))
    multi_files = sorted(glob.glob(multi_glob), key=shard_key("dpo_rejected_multi"))

    singles, multis = [], []
    for p in single_files:
        singles.extend(load(p))
    for p in multi_files:
        multis.extend(load(p))
    print(
        f"single: {len(singles)} (来自 {len(single_files)} 分片), multi: {len(multis)} (来自 {len(multi_files)} 分片)"
    )

    # ---- 清洗 ----
    dropped = {"empty": 0, "short": 0, "same": 0, "bad_chosen": 0}
    clean = []
    for r in singles:
        rej = (r.get("rejected") or "").strip()
        cho = (r.get("chosen") or "").strip()
        if len(cho) < 4:
            dropped["bad_chosen"] += 1
            continue
        if not rej:
            dropped["empty"] += 1
            continue
        if len(rej) < 8:
            dropped["short"] += 1
            continue
        if rej == cho:
            dropped["same"] += 1
            continue
        clean.append({"instruction": r["instruction"], "chosen": cho, "rejected": rej})
    for r in multis:
        rej = (r.get("rejected") or "").strip()
        cho = (r.get("chosen") or "").strip()
        msgs = r.get("messages")
        if not msgs or len(cho) < 4:
            dropped["bad_chosen"] += 1
            continue
        if not rej:
            dropped["empty"] += 1
            continue
        if len(rej) < 8:
            dropped["short"] += 1
            continue
        if rej == cho:
            dropped["same"] += 1
            continue
        clean.append({"messages": msgs, "chosen": cho, "rejected": rej})

    print(f"清洗剔除: {dropped} → 保留 {len(clean)} 对")
    with open(out, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_single = sum(1 for r in clean if "instruction" in r)
    print(f"Done -> {out} (single {n_single} + multi {len(clean) - n_single})")


if __name__ == "__main__":
    main()

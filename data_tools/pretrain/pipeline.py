"""数据预处理管道（与变体无关，所有变体共用）。

流程:
  step 1: 粗精确去重 (MD5)  — 先剔除完全重复，减少后续清洗/去重计算量
  step 2: 基础清洗           — 去乱码、繁转简、过滤广告/低质（全源 min_zh_ratio=0.15）
  step 3: SimHash 逐源去重   — 各源独立全局查重
  step 4: SimHash 跨源去重   — 所有源合并指纹，跨源剔除重复

输出 → data/raw/{name}_dedup.txt（供 build.py 混合切分使用）

用法:
    python data_tools/pretrain/pipeline.py
    python data_tools/pretrain/pipeline.py --skip_simhash
"""

import argparse
import os

from gleamlm.preprocessing.clean_text import clean_file
from gleamlm.preprocessing.dedup_text import dedup_file
from gleamlm.preprocessing.filter_qa import filter_qa

SOURCES = [
    {"name": "edu", "type": "text"},
    {"name": "news", "type": "news"},
    {"name": "wiki", "type": "text"},
    {"name": "baike", "type": "text"},
    {"name": "qa", "type": "qa"},
]

MIN_Zh_RATIO = 0.15


def _raw_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw.txt")


def _raw_dedup_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw_dedup.txt")


def _clean_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_clean.txt")


def _final_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_dedup.txt")


def main():
    parser = argparse.ArgumentParser(
        description="数据预处理管道（去重→清洗→SimHash 逐源+跨源去重）"
    )
    parser.add_argument("--input", default="data/raw", help="原始数据目录")
    parser.add_argument("--skip_exact_dedup", action="store_true")
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--skip_simhash", action="store_true")
    parser.add_argument("--exact_mode", default="exact", choices=["exact", "prefix"])
    parser.add_argument("--prefix_len", type=int, default=100)
    parser.add_argument("--simhash_threshold", type=int, default=3)
    args = parser.parse_args()

    raw_dir = args.input
    threshold = args.simhash_threshold
    names = [s["name"] for s in SOURCES]
    print(f"Sources: {names}")

    # ──── step 1: 粗精确去重 ────
    if args.skip_exact_dedup:
        print("\n[1/4] 跳过精确去重（--skip_exact_dedup）")
    else:
        print("\n[1/4] 粗精确去重（MD5 全文去重）")
        for s in SOURCES:
            raw = _raw_path(raw_dir, s["name"])
            deduped = _raw_dedup_path(raw_dir, s["name"])
            if not os.path.exists(raw):
                print(f"  Skip {s['name']}: {raw} not found")
                continue
            if os.path.exists(deduped) and os.path.getsize(deduped) > 0:
                print(f"  Skip {s['name']}: {deduped} exists")
                continue
            mode = "prefix" if s["type"] == "news" else args.exact_mode
            print(f"  去重: {s['name']} (mode={mode})")
            dedup_file(raw, deduped, mode=mode, prefix_len=args.prefix_len)

    # ──── step 2: 清洗 ────
    if args.skip_clean:
        print("\n[2/4] 跳过清洗（--skip_clean）")
    else:
        print(f"\n[2/4] 基础清洗（min_zh_ratio={MIN_Zh_RATIO}, 去乱码、繁转简、过滤低质）")
        for s in SOURCES:
            src = _raw_dedup_path(raw_dir, s["name"])
            if not os.path.exists(src):
                src = _raw_path(raw_dir, s["name"])
            clean = _clean_path(raw_dir, s["name"])
            if not os.path.exists(src):
                print(f"  Skip {s['name']}: no source found")
                continue
            if os.path.exists(clean) and os.path.getsize(clean) > 0:
                print(f"  Skip {s['name']}: {clean} exists")
                continue
            print(f"  Cleaning: {s['name']}")
            clean_file(
                src,
                clean,
                min_len=30,
                max_len=3000,
                convert_zh=True,
                min_zh_ratio=MIN_Zh_RATIO,
                filter_ads=s["name"] == "news",
                filter_wiki_junk=s["name"] == "wiki",
            )

    # ──── step 3: SimHash 逐源去重 + QA过滤 ────
    all_fingerprints: set[int] = set()
    if args.skip_simhash:
        print("\n[3/4] 跳过 SimHash 去重（--skip_simhash）")
    else:
        print("\n[3/4] SimHash 逐源去重 / QA过滤")
        for s in SOURCES:
            src = _clean_path(raw_dir, s["name"])
            if not os.path.exists(src):
                src = _final_path(raw_dir, s["name"])
            final = _final_path(raw_dir, s["name"])
            if not os.path.exists(src):
                print(f"  Skip {s['name']}: {src} not found")
                continue
            if os.path.exists(final) and os.path.getsize(final) > 0:
                print(f"  Skip {s['name']}: {final} exists")
                continue

            if s["type"] == "qa":
                print(f"  QA过滤: {s['name']}")
                filter_qa(src, final)
            else:
                print(f"  SimHash: {s['name']} (threshold={threshold})")
                fps = dedup_file(
                    src,
                    final,
                    mode="simhash",
                    simhash_threshold=threshold,
                )
                all_fingerprints.update(fps)

        print(
            f"\n  Collected fingerprints: {len(all_fingerprints):,} across {len(SOURCES)} sources"
        )

    # ──── step 4: 跨源 SimHash 全局去重 ────
    if args.skip_simhash:
        print("\n[4/4] 跳过跨源去重（--skip_simhash）")
    else:
        print("\n[4/4] 跨源 SimHash 全局去重")
        for s in SOURCES:
            final = _final_path(raw_dir, s["name"])
            if not os.path.exists(final) or s["type"] == "qa":
                continue
            tmp = final + ".tmp"
            print(f"  Cross-dedup: {s['name']} (against {len(all_fingerprints):,} fingerprints)")
            returned = dedup_file(
                final,
                tmp,
                mode="simhash",
                simhash_threshold=threshold,
                existing_fingerprints=all_fingerprints,
            )
            os.replace(tmp, final)
            all_fingerprints.update(returned)
            print(f"  Updated fingerprints: {len(all_fingerprints):,}")

    print("  完成")


if __name__ == "__main__":
    main()

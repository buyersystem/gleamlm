"""通用文本去重。支持精确去重和近邻去重（按前 N 字符哈希）。
适用于新闻、百科、维基等非 QA 数据。
"""

from __future__ import annotations

import argparse
import hashlib


def normalize(text: str, strip_whitespace: bool = True) -> str:
    if strip_whitespace:
        text = " ".join(text.split())
    return text


def dedup_file(
    input_path: str, output_path: str, mode: str = "exact", prefix_len: int = 100
) -> None:
    total = 0
    kept = 0
    deduped = 0
    seen: set[str] = set()

    print(f"Dedup: {input_path}")
    print(f"  mode={mode}, prefix_len={prefix_len}")

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total += 1
            text = normalize(line.strip())
            if not text:
                continue

            if mode == "exact":
                key = hashlib.md5(text.encode("utf-8")).hexdigest()
            else:  # prefix
                key = hashlib.md5(text[:prefix_len].encode("utf-8")).hexdigest()

            if key in seen:
                deduped += 1
                continue

            seen.add(key)
            fout.write(text + "\n")
            kept += 1

            if total % 500000 == 0:
                print(
                    f"  Processed {total:,} lines, kept {kept:,}, "
                    f"dedup {deduped:,} ({100 * deduped / total:.1f}%)"
                )

    pct = 100 * kept / max(1, total)
    dedup_pct = 100 * deduped / max(1, total)
    print(f"\nDone: {total:,} lines → {kept:,} kept ({pct:.1f}%)")
    print(f"  Deduplicated: {deduped:,} ({dedup_pct:.1f}%)")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="通用文本去重工具")
    parser.add_argument("--input", type=str, required=True, help="输入文件")
    parser.add_argument("--output", type=str, required=True, help="输出文件")
    parser.add_argument(
        "--mode",
        type=str,
        default="exact",
        choices=["exact", "prefix"],
        help="exact=全文精确去重, prefix=前N字符去重（默认exact）",
    )
    parser.add_argument(
        "--prefix_len", type=int, default=100, help="prefix 模式下的字符数（默认100）"
    )
    args = parser.parse_args()

    dedup_file(args.input, args.output, args.mode, args.prefix_len)


if __name__ == "__main__":
    main()

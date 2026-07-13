"""清洗 sft_api_new.jsonl：去除 markdown 标记、修复格式"""

import json
import re
import sys

INPUT = "sft_api_new.jsonl"
OUTPUT = "sft_api_clean.jsonl"

stats = {
    "total": 0,
    "removed_bold": 0,
    "removed_italic": 0,
    "removed_heading": 0,
    "removed_code": 0,
    "removed_list_marker": 0,
    "trimmed_trailing_newlines": 0,
    "invalid": 0,
}


def clean_text(text: str) -> str:
    """清洗单个 output/instruction 文本"""
    original = text

    # 1. 去除 **bold** → bold
    new = re.sub(r'\*\*(.+?)\*\*', r'\1', original)
    if new != original:
        stats["removed_bold"] += 1
    text = new

    # 2. 去除 *italic* / _underline_
    new = re.sub(r'(?<!\*)\*(.+?)\*(?!\*)', r'\1', text)
    if new != original:
        stats["removed_italic"] += 1
    text = new

    # 3. 去除行首 # 标题标记
    new = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    if new != text:
        stats["removed_heading"] += 1
    text = new

    # 4. 去除行首 `1.` / `- ` / `* ` 列表标记
    new = re.sub(r'^(\d+[.)]\s*|[-*]\s+)', '', text, flags=re.MULTILINE)
    if new != text:
        stats["removed_list_marker"] += 1
    text = new

    # 5. 去除行尾多余空行（保留最多1个）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 6. 去除首尾空白
    text = text.strip()

    return text


def main():
    with open(INPUT, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    cleaned = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        stats["total"] += 1

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            stats["invalid"] += 1
            if stats["invalid"] <= 3:
                print(f"  Warning: invalid JSON at line {i+1}: {e}")
            # 尝试修复常见断行重连
            continue

        instruction = obj.get("instruction", "")
        output = obj.get("output", "")

        # 不清洗 instruction，只清洗 output
        clean_output = clean_text(output)
        obj["output"] = clean_output

        cleaned.append(obj)

    with open(OUTPUT, "w", encoding="utf-8") as f_out:
        for obj in cleaned:
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Total lines: {stats['total']}")
    print(f"Clean lines: {len(cleaned)}")
    print(f"Invalid (skipped): {stats['invalid']}")
    print(f"Samples with bold `**` removed: {stats['removed_bold']}")
    print(f"Samples with italic removed: {stats['removed_italic']}")
    print(f"Samples with heading removed: {stats['removed_heading']}")
    print(f"Samples with list markers removed: {stats['removed_list_marker']}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()

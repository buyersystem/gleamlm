"""数据管线审计 manifest — 产物脚本收尾调用 write_manifest（方案 4 试点）。

设计（2026-09）:
  - manifest 与产物同目录（f"{output}.manifest.json"），记录 tool / 输入输出
    指纹 / 剔除计数 / 关键参数 / 时间戳 → "这批数据从哪来"可追溯
  - records 与 sha256 由文件本身流式数出（单一事实来源：调用方不必重复维护
    计数；流式读取不爆内存）
  - 路径统一以 / 分隔写入，跨平台可读

用法:
  from data_tools.shared.audit import write_manifest
  write_manifest(out, tool="mix_sft", inputs=[a, b],
                 dropped={"duplicate": 3}, params={"qa_keep": 10000}, seed=42)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime


def fingerprint(path: str) -> tuple[int, str]:
    """流式计算 JSONL 的 (非空行数, sha256)。"""
    h = hashlib.sha256()
    records = 0
    with open(path, "rb") as f:
        for line in f:
            h.update(line)
            if line.strip():
                records += 1
    return records, h.hexdigest()


def write_manifest(
    output: str,
    *,
    tool: str,
    inputs: list[str] | None = None,
    dropped: dict[str, int] | None = None,
    params: dict | None = None,
    seed: int | None = None,
) -> str:
    """写 {output}.manifest.json 并返回 manifest 路径。

    inputs 传处理前的源文件路径（原地重写类工具的输入已不可寻，可传备份文件）；
    input / output 的 records 与 sha256 均以文件当前内容为准自动数出。
    """
    in_records = []
    for p in inputs or []:
        records, sha = fingerprint(p)
        in_records.append({"path": p.replace("\\", "/"), "records": records, "sha256": sha})

    records, sha = fingerprint(output)
    manifest = {
        "tool": tool,
        "inputs": in_records,
        "output": {"path": output.replace("\\", "/"), "records": records, "sha256": sha},
        "dropped": dropped or {},
        "params": params or {},
        "seed": seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = f"{output}.manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Manifest: {path}")
    return path

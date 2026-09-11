"""训练指标通道 — 哨兵行 emit/parse 契约（训练脚本与 WebUI 解析侧共用）。

训练脚本在人类可读日志之外，额外打印一行带哨兵前缀的结构化 JSON:

    @@GLEAM_METRIC {"split":"train","step":100,"total":34054,"loss":3.19,"lr":0.00004}

字段（除 split 外均可选，缺失即不产出对应曲线）:
  split     "train" | "val"（val 独有 ppl）
  step      int — 训练步（面板 x 轴；跨 epoch 单调递增）
  total     int — 总步数（面板进度用）
  loss      float — 训练/验证损失（val 为裸 CE，可直接取 exp 当 PPL）
  lr        float — 当前学习率
  tok_per_s float — 吞吐（tok/s）
  gpu_mem   float — 进程显存占用（GiB）

WebUI 解析侧（webui/routers/training.py::_parse_metric_lines）优先消费哨兵行
（结构化、零正则），未命中时回退旧格式正则 —— 哨兵行之前的老日志重放不受影响。
从此人类可读行的格式不再承担机器接口职责，改格式不会静默断掉面板曲线。
"""

import json
from typing import Any

SENTINEL = "@@GLEAM_METRIC "


def format_metric_line(**fields: Any) -> str:
    """构造一行哨兵指标（含前缀，不含换行符）。"""
    return SENTINEL + json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def emit_metric(**fields: Any) -> None:
    """打印一行哨兵指标到 stdout（训练脚本侧统一出口；flush 保证面板实时）。"""
    print(format_metric_line(**fields), flush=True)


def parse_metric_line(line: str) -> dict[str, Any] | None:
    """解析哨兵指标行 → dict；非哨兵行或畸形 JSON 返回 None（调用方走正则回退）。"""
    stripped = line.strip()
    if not stripped.startswith(SENTINEL):
        return None
    try:
        rec = json.loads(stripped[len(SENTINEL) :])
    except json.JSONDecodeError:
        return None
    return rec if isinstance(rec, dict) else None

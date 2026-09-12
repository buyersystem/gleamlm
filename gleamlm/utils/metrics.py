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
  margin    float — DPO 隐式奖励间隔 β·mean((logπc−logπref_c)−(logπr−logπref_r))
  acc       float — 上式的排序正确率（term>0 的配对占比）
  reward    float — GRPO/PPO 的平均奖励（GRPO 已产出；PPO 尚未）
  kl        float — GRPO/PPO 的 KL 散度（尚未有脚本产出）
  len       float — 平均响应长度（尚未有脚本产出）

**键名即契约**：解析侧的许可名单在 `webui/routers/training.py::_EXTRA_KEYS`，
两条通道（哨兵 / tqdm 正则回退）共用同一份。未知键会被静默忽略 ——
写错键名不会报错、只会不显示，所以新指标要同时改这里与 _EXTRA_KEYS。

WebUI 解析侧（webui/routers/training.py::_parse_metric_lines）优先消费哨兵行
（结构化、零正则），未命中时回退旧格式正则 —— 哨兵行之前的老日志重放不受影响。
哨兵行常粘连在 tqdm 帧尾（帧以 \r 分帧、无换行），解析按行内标记定位而非行首。
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
    """解析哨兵指标行 → dict；无标记或畸形 JSON 返回 None（调用方走正则回退）。

    标记按「行内任意位置」定位而非行首：tqdm 帧以 \r 分帧（无换行），哨兵 print
    会直接粘在帧尾 —— 行首匹配会让全部哨兵解析失败（DPO 实测 18/18 粘连），
    派生 H19 门控失效、曲线退化为帧回退（坐标错位 + 阶梯形态）。
    """
    stripped = line.strip()
    idx = stripped.find(SENTINEL)
    if idx < 0:
        return None
    try:
        rec = json.loads(stripped[idx + len(SENTINEL) :])
    except json.JSONDecodeError:
        return None
    return rec if isinstance(rec, dict) else None

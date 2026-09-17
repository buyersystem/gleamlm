"""训练监控累积器 —— 7 个训练脚本共用的一份实现（吞吐 / 显存 / 窗口量）。

**为什么单独成文件**：`gleamlm/utils/metrics.py` 是哨兵行的 emit/parse 契约，被 WebUI
解析侧直接 import（`webui/routers/training.py`），必须保持**零 torch 依赖**；本模块要查
显存，另起文件才能保住那条边界。

**为什么 `window_max` 落在这里**：`gleamlm/trainer/base_trainer.py` 会拉进 models /
tokenizer / inference 整条栈，而 `utils` 是叶子层，依赖方向只允许 `trainer → utils`。
窗口量的累加本就是本模块的职责。

**为什么需要 `MetricWindow`**：`tok_per_s` / `gpu_mem` / `grad_norm` 的口径必须全场一致
（面板上同名 key 只表示一个量），但 7 个训练脚本各有自己的循环 —— `gleamlm/trainer/`
里没有公共 train loop 可挂钩子。此前这份逻辑在每个脚本里各内联一份，口径只能靠人工对齐；
收成一个类后，各阶段唯一真正的差异（「一个 step 算多少 token」）留在调用点。
"""

from __future__ import annotations

import time
from typing import Any

import torch

from gleamlm.utils.gpu import gpu_stats


def window_max(current: float | None, value: float | None) -> float | None:
    """窗口内最大值累加（None 安全）—— grad_norm 的记录口径。

    指标窗口 = 「自上次哨兵行以来」，而一个 log_interval 窗口里会做多次 optimizer step。
    grad_norm 记**窗口内 max 而非最后一个 step**：它的用途是预警发散，靠的是**尖峰**，
    只留末值会把中间 log_interval-1 个 step 的尖峰无声丢掉（loss 记窗口均值是对的 ——
    它关心趋势；两者口径刻意不同，见 gleamlm/utils/metrics.py 契约）。

    `value=None`（未启用裁剪 / 该步未做 optimizer step）时保持 `current` 不变；
    若整个窗口都没记录到，返回 None → 哨兵行记 null → 解析侧跳过、不产曲线点。
    """
    if value is None:
        return current
    return value if current is None else max(current, value)


class MetricWindow:
    """一个 `log_interval` 窗口的监控累积器。

    一轮生命周期：`start()` → 干活 → `stop(tokens)` →（多步）`add_loss()` /
    `add_grad_norm()` → 到上报点 `refresh_gpu()` + `emit_kwargs()` → `reset()`。

    **口径（关键，勿随手改）**：
      - `tok_per_s` = 最近一次 `stop(tokens)` 的 tokens ÷ 那段 `start()→stop()` 的耗时。
        **一次 `stop()` 覆盖什么、算多少 token 由调用方决定**：
          · 训练型（pretrain / sft / sft_lora）：一个微批的前后向，
            `tokens = batch × seq`（DPO 是 `2 × batch × seq`，一个微批含两次前向）
          · 生成型（grpo / opd / ppo）：一个训练迭代（**含 rollout**），
            `tokens = 本迭代新生成的 token 数`
        跨族不可直接比较 —— 生成型没有固定的 token/step。
      - `loss` 记**窗口均值**（关心趋势）；`grad_norm` 记**窗口内 max**（关心尖峰）。
        两者刻意不同，见 `gleamlm/utils/metrics.py` 的「窗口口径」一节。
      - `gpu_mem` 是**设备已用量**（`nvidia-smi memory.used` 口径），且**必须显式调
        `refresh_gpu()` 才更新** —— 查询较重（NVML 缺失时要起 nvidia-smi 子进程），
        不能每步都查。`stop()` 刻意不刷显存，正是为了保住调用方的刷新节奏。
    """

    def __init__(self, device: torch.device, local_rank: int = 0) -> None:
        self._device = device
        self._local_rank = local_rank
        self._t0: float | None = None
        # 瞬时量：reset() 不清（它们是「最近一次」的快照，不是窗口累计）
        self.last_dt = 0.0
        self.tok_per_s = 0.0
        self.gpu_util = 0.0
        self.gpu_mem = 0.0
        self.gpu_mem_peak = 0.0
        self.gpu_mem_total = 0.0
        # 窗口量：reset() 清空
        self._loss_sum = 0.0
        self._batches = 0
        self._grad_norm_max: float | None = None

    def start(self) -> None:
        """标记一段计时的起点（一个微批 / 一个训练迭代）。"""
        self._t0 = time.perf_counter()

    def stop(self, tokens: float) -> float:
        """结束计时并记 `tokens`，更新 `tok_per_s` / `last_dt`；返回本次 tok/s。

        与 `start()` 成对使用。刻意**不刷新显存** —— 见类 docstring 的刷新节奏说明。
        """
        if self._t0 is None:
            raise RuntimeError("MetricWindow.stop() 之前必须先 start()")
        self.last_dt = time.perf_counter() - self._t0
        self._t0 = None
        self.tok_per_s = tokens / self.last_dt if self.last_dt > 0 else 0.0
        return self.tok_per_s

    def refresh_gpu(self) -> None:
        """刷新显存 / 利用率快照。查询较重，按上报节奏调用（不是每步）。"""
        self.gpu_util, self.gpu_mem, self.gpu_mem_peak, self.gpu_mem_total = gpu_stats(
            self._device, self._local_rank
        )

    def add_loss(self, value: float) -> None:
        """累加一个窗口内的 loss（调用一次记一个点；`batches` 即调用次数）。"""
        self._loss_sum += value
        self._batches += 1

    def add_grad_norm(self, value: float | None) -> None:
        """累加窗口内 grad_norm 的**最大值**（None 安全，见 `window_max`）。"""
        self._grad_norm_max = window_max(self._grad_norm_max, value)

    @property
    def loss(self) -> float | None:
        """窗口 loss 均值；窗口内一个点都没有时为 None。"""
        return self._loss_sum / self._batches if self._batches else None

    @property
    def grad_norm(self) -> float | None:
        """窗口内 grad_norm 最大值；整窗未记录时为 None（未启用裁剪）。"""
        return self._grad_norm_max

    @property
    def batches(self) -> int:
        """窗口内已累加的点数（loss 的分母；GRPO 的 reward 均值也用它）。"""
        return self._batches

    def emit_kwargs(self) -> dict[str, Any]:
        """喂给 `emit_metric(**win.emit_kwargs(), split=..., step=..., lr=...)`。

        只含「监控健康度」这四个键；split / step / total / lr 与任务专属键
        （reward / margin / acc）由调用方给 —— 本类不认识任务语义。
        """
        return {
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "tok_per_s": self.tok_per_s,
            "gpu_mem": self.gpu_mem,
        }

    def reset(self) -> None:
        """清窗口量。瞬时量（`tok_per_s` / 显存）保留 —— 它们是快照，不是累计。"""
        self._loss_sum = 0.0
        self._batches = 0
        self._grad_norm_max = None

"""GPU 占用查询 —— 训练脚本共同的数据源（哨兵行 `gpu_mem` / wandb / 状态行）。

为什么不用纯 torch API 报显存：要报的是**设备已用量**（对齐 `nvidia-smi` 的
`memory.used`，含其它进程与驱动占用），而不是 PyTorch 缓存分配器的 `allocated`
（只含活跃张量，比真实占用小得多 —— 拿它当监控会把"已占满"读成"很宽裕"）。

Windows 上 `torch.cuda.utilization` 依赖 pynvml，缺失时**抛异常**；直接回退 0 会
误导监控（GPU 满载却显示 0%），所以回退到 `nvidia-smi` 子进程查询。调用点都在
每 log_interval 才执行一次的位置，毫秒级开销可忽略。
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

import torch

_SMI_ARGS = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


def gpu_stats(device: torch.device, local_rank: int = 0) -> tuple[float, float, float, float]:
    """(利用率%, 设备已用显存 GiB, 进程峰值 GiB, 总显存 GiB)；非 CUDA 全部 0。

    显存统一 GiB（bytes/2**30 或 MiB/1024），与 `nvidia-smi` 读数同量纲。
    `local_rank` 只在 nvidia-smi 回退路径用于多卡选取对应 GPU 行。
    任何探测失败都返回全 0（监控降级为"无数据"，不抛异常打断训练）。
    """
    if device.type != "cuda":
        return 0.0, 0.0, 0.0, 0.0
    try:
        # 主路径：mem_get_info 给 (free, total)，used = total − free 即设备已用量
        # （与 nvidia-smi 同口径）；utilization 需要 pynvml，缺失时抛异常 → 走回退。
        free, total = torch.cuda.mem_get_info(device)
        util = float(torch.cuda.utilization(device))
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        return util, (total - free) / 2**30, peak, total / 2**30
    except Exception:
        pass
    try:
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            # Ctrl+C 隔离：子进程默认与主进程同 console 前台组，用户 Ctrl+C 会
            # 同时中断 nvidia-smi（其挂起不退出会拖住下方等待，表现为退出卡住）；
            # 新进程组使其不受 SIGINT 影响，正常跑完即退。
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        out = subprocess.check_output(_SMI_ARGS, text=True, timeout=3, **kwargs)
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        u, mem_mib, total_mib = lines[min(local_rank, len(lines) - 1)].split(",")
        # 回退路径拿不到进程峰值 → 0（调用方只把它当 wandb 参考量）
        return float(u), float(mem_mib) / 1024, 0.0, float(total_mib) / 1024
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

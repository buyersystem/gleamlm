"""
GleamLM WebUI — 训练 router（进程遥控器 + 指标持久化 + 配置管理）。

挂载于 main.py（prefix=/api），端点:
  /api/configs /api/config /api/config/copy   配置读取 / 校验写回 / 另存为
  /api/train/tasks /start /stop /status /stream /runs /metrics
  /api/models                              checkpoint 树扫描（推理 tab 模型菜单）

三条铁律（docs/前端面板设计.md §2）：
  1. 不介入训练循环 —— subprocess 拉起现有脚本，面板是遥控器
  2. 不持有参数默认值 —— 前端表单空值不传 CLI（None → 脚本默认 / YAML 裁决）
  3. 校验复用 gleamlm.utils.config（Pydantic 同套 schema，前端不写校验规则）

进程模型：训练子进程 stdout/stderr 直写日志文件（stdout=文件句柄），
解析线程与 SSE 线程各自独立读文件增量 —— 面板进程挂了训练不挂、
训练崩了面板仍可重放日志；断线重连按行号 seq 回放（无 EventSource
Last-Event-ID，fetch 方案自管）。
"""

import asyncio
import contextlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from gleamlm.types import ConfigValidationError
from gleamlm.utils.config import load_config
from gleamlm.utils.metrics import parse_metric_line
from tools.tracker import ExperimentTracker

_WEBUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(_WEBUI_DIR, "logs")
DB_PATH = os.path.join(LOGS_DIR, "experiments.db")
os.makedirs(LOGS_DIR, exist_ok=True)  # 训练日志 + tracker SQLite 落盘目录
CONFIG_DIR = os.path.join(ROOT_DIR, "manual", "configs")
MY_CFG_DIR = os.path.join(ROOT_DIR, "manual", "my_configs")

# 变体 = 配置模板名：两配置目录（内置 + 用户副本）下全部 *.yaml 的 stem 即
# 变体清单（目录即白名单：IDE 直接新建的实验配置也能被面板读到/启动；同名时
# manual/my_configs 副本覆盖内置；写权限仍只开放 manual/my_configs/，见 _resolve_config_path）
_LAUNCHERS = ("python", "torchrun", "deepspeed")


def _variants() -> list[str]:
    """变体清单 = 两配置目录下全部 *.yaml 的 stem（去重、排序）。"""
    names: set[str] = set()
    for d in (CONFIG_DIR, MY_CFG_DIR):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".yaml"):
                    names.add(fn[: -len(".yaml")])
    return sorted(names)


def _config_path(variant: str) -> str:
    """变体名 → 配置绝对路径（manual/my_configs 同名副本优先）；不存在返回空串。"""
    if variant and variant in _variants():
        for d in (MY_CFG_DIR, CONFIG_DIR):
            p = os.path.join(d, f"{variant}.yaml")
            if os.path.isfile(p):
                return p
    return ""


router = APIRouter()

# ── 任务注册表: 面板的字段清单与脚本 argparse 对齐, 空值不传 CLI ─────────
# fields 元素: {name: flag 名(去 --), type: str/path/int/float/bool/choice,
#              label: 表单标签, help: 说明, required?: 必填(仅真必填),
#              choices?: choice 候选项, flag?: 脚本旗标拼写覆写（默认 = name）}
# upstream_stage?: 上游阶段（模型下拉默认聚焦 checkpoints/<variant>/<stage>/；
#                  "" = 变体根）。与 _task_defaults 的回落链同源、逐条对齐
#                  manual/*.py 的 CLI 缺省裁决
_TASKS: dict[str, dict[str, Any]] = {
    "pretrain": {
        "script": "manual/pretrain.py",
        "label": "预训练",
        "short": "预训练",
        "variant_flag": False,
        # 面板自动附加: --no-pbar 日志式每 log_interval 步一行(管道下 tqdm 帧不可靠);
        # --tensorboard 已有 flag, 零侵入, tfevents 落 output_dir/runs/ 供指标增强
        "auto": ["--no-pbar", "--tensorboard"],
        "fields": [
            {
                "name": "model",
                "type": "path",
                "label": "配置模板",
                "suggest": "configs",  # 候选 = 配置清单（manual/configs + manual/my_configs），选项只显文件名
                "required": True,
            },
            {
                "name": "data",
                "type": "path",
                "label": "数据目录",
            },
            {
                "name": "output_dir",
                "type": "path",
                "label": "保存目录",
            },
            {
                "name": "resume",
                "type": "path",
                "label": "续训模型",
            },
            {"name": "epochs", "type": "int", "label": "epochs"},
            {"name": "batch_size", "type": "int", "label": "batch_size"},
            {"name": "accumulate", "type": "int", "label": "accumulate"},
            {"name": "lr", "type": "float", "label": "lr"},
            {"name": "seed", "type": "int", "label": "seed"},
            {
                "name": "val_data",
                "type": "path",
                "label": "验证数据",
            },
        ],
    },
    "sft": {
        "script": "manual/sft.py",
        "label": "SFT 指令微调",
        "short": "SFT",
        "variant_flag": True,
        "config_dir_flag": True,
        "upstream_stage": "",
        "auto": [],
        "fields": [
            {
                "name": "model_path",
                "type": "path",
                "label": "基座模型",
            },
            {
                "name": "save_dir",
                "type": "path",
                "label": "保存目录",
            },
            {
                "name": "data_path",
                "type": "path",
                "label": "数据",
            },
            {"name": "epochs", "type": "int", "label": "epochs"},
            {"name": "lr", "type": "float", "label": "lr"},
            {"name": "batch_size", "type": "int", "label": "batch_size"},
            {"name": "accumulate_grad", "type": "int", "label": "accumulate_grad"},
            {"name": "max_seq_len", "type": "int", "label": "max_seq_len"},
            {
                "name": "lr_scheduler",
                "type": "choice",
                "label": "lr_scheduler",
                "choices": ["cosine", "wsd"],
            },
            {"name": "warmup_ratio", "type": "float", "label": "warmup_ratio"},
            {"name": "stable_ratio", "type": "float", "label": "stable_ratio"},
            {"name": "min_lr_ratio", "type": "float", "label": "min_lr_ratio"},
            {"name": "weight_decay", "type": "float", "label": "weight_decay"},
            {"name": "seed", "type": "int", "label": "seed"},
        ],
    },
    "dpo": {
        "script": "manual/dpo.py",
        "label": "DPO 偏好对齐",
        "short": "DPO",
        "variant_flag": True,
        "config_dir_flag": True,
        "upstream_stage": "sft",
        "auto": [],
        "fields": [
            {
                "name": "model_path",
                "type": "path",
                "label": "基座模型",
            },
            {
                "name": "output_dir",
                "type": "path",
                "label": "保存目录",
            },
            {
                "name": "data_path",
                "type": "path",
                "label": "偏好数据",
            },
            {"name": "epochs", "type": "int", "label": "epochs"},
            {"name": "lr", "type": "float", "label": "lr"},
            {"name": "beta", "type": "float", "label": "beta"},
            {"name": "batch_size", "type": "int", "label": "batch_size"},
            {"name": "accumulate_grad", "type": "int", "label": "accumulate_grad"},
            {"name": "max_seq_len", "type": "int", "label": "max_seq_len"},
            {
                "name": "lr_scheduler",
                "type": "choice",
                "label": "lr_scheduler",
                "choices": ["cosine", "wsd"],
            },
            {"name": "warmup_ratio", "type": "float", "label": "warmup_ratio"},
            {"name": "stable_ratio", "type": "float", "label": "stable_ratio"},
            {"name": "min_lr_ratio", "type": "float", "label": "min_lr_ratio"},
            {"name": "weight_decay", "type": "float", "label": "weight_decay"},
        ],
    },
    "opd": {
        "script": "manual/opd.py",
        "label": "OPD 在线策略蒸馏",
        "short": "OPD",
        "variant_flag": True,
        "config_dir_flag": True,
        "upstream_stage": "dpo",
        "auto": [],
        "fields": [
            {
                "name": "model",
                "type": "path",
                "label": "学生模型",
                "required": True,
            },
            {
                "name": "data",
                "type": "path",
                "label": "prompt 数据",
            },
            {
                "name": "teacher_model_path",
                "type": "path",
                "label": "教师模型",
            },
            {"name": "output_dir", "type": "path", "label": "保存目录"},
            {"name": "epochs", "type": "int", "label": "epochs"},
            {"name": "batch_size", "type": "int", "label": "batch_size"},
            {"name": "n_samples", "type": "int", "label": "n_samples"},
            {"name": "seq_len", "type": "int", "label": "seq_len"},
            {"name": "max_new_tokens", "type": "int", "label": "max_new_tokens"},
            {"name": "temperature", "type": "float", "label": "temperature"},
            {"name": "lr", "type": "float", "label": "lr"},
            {"name": "weight_decay", "type": "float", "label": "weight_decay"},
            {"name": "clip", "type": "float", "label": "clip"},
            {"name": "entropy_coeff", "type": "float", "label": "entropy_coeff"},
            {"name": "log_interval", "type": "int", "label": "log_interval"},
        ],
    },
    "sft_lora": {
        "script": "manual/sft_lora.py",
        "label": "LoRA 微调（可选旁路）",
        "short": "LoRA",
        "variant_flag": True,
        "config_dir_flag": True,
        "upstream_stage": "",
        "auto": [],
        "fields": [
            {
                "name": "model",
                "type": "path",
                "label": "基座模型",
                "required": True,
            },
            {
                "name": "data",
                "type": "path",
                "label": "数据",
            },
            {"name": "output_dir", "type": "path", "label": "保存目录"},
            {"name": "epochs", "type": "int", "label": "epochs"},
            {"name": "batch_size", "type": "int", "label": "batch_size"},
            {"name": "seq_len", "type": "int", "label": "seq_len"},
            {"name": "lr", "type": "float", "label": "lr"},
            {"name": "clip", "type": "float", "label": "clip"},
            {"name": "lora_r", "type": "int", "label": "lora_r"},
            {"name": "lora_alpha", "type": "int", "label": "lora_alpha"},
            {"name": "log_interval", "type": "int", "label": "log_interval"},
            {"name": "merge", "type": "bool", "label": "训练后合并 LoRA 权重"},
        ],
    },
    "grpo": {
        "script": "manual/grpo.py",
        "label": "GRPO 强化对齐（与 DPO 并列）",
        "short": "GRPO",
        "variant_flag": True,
        # 配置模板仅用于面板推导与 run 归属，不落 CLI（grpo.py 无 --variant）
        "variant_cli": False,
        "upstream_stage": "sft",
        "auto": [],
        "fields": [
            {
                "name": "model",
                "type": "path",
                "label": "基座模型",
                "required": True,
            },
            {
                "name": "data",
                "type": "path",
                "label": "prompt 数据",
                "required": True,
            },
            {"name": "output_dir", "type": "path", "label": "保存目录"},
            {"name": "epochs", "type": "int", "label": "epochs"},
            {"name": "batch_size", "type": "int", "label": "batch_size"},
            {"name": "seq_len", "type": "int", "label": "seq_len"},
            {"name": "max_new_tokens", "type": "int", "label": "max_new_tokens"},
            {"name": "group_size", "type": "int", "label": "group_size"},
            {"name": "temperature", "type": "float", "label": "temperature"},
            {"name": "beta", "type": "float", "label": "beta"},
            {"name": "lr", "type": "float", "label": "lr"},
            {"name": "weight_decay", "type": "float", "label": "weight_decay"},
            {"name": "clip", "type": "float", "label": "clip"},
            {"name": "log_interval", "type": "int", "label": "log_interval"},
        ],
    },
    "ppo": {
        "script": "manual/ppo.py",
        "label": "PPO 强化对齐（与 DPO 并列）",
        "short": "PPO",
        "variant_flag": True,
        # 配置模板仅用于面板推导与 run 归属，不落 CLI（ppo.py 无 --variant）
        "variant_cli": False,
        "upstream_stage": "sft",
        "auto": [],
        "fields": [
            {
                "name": "model",
                "type": "path",
                "label": "基座模型",
                "required": True,
            },
            {
                "name": "data",
                "type": "path",
                "label": "prompt 数据",
                "required": True,
            },
            {"name": "output_dir", "type": "path", "label": "保存目录"},
            {"name": "epochs", "type": "int", "label": "epochs"},
            {"name": "batch_size", "type": "int", "label": "batch_size"},
            {"name": "seq_len", "type": "int", "label": "seq_len"},
            {"name": "max_new_tokens", "type": "int", "label": "max_new_tokens"},
            {"name": "epsilon", "type": "float", "label": "epsilon"},
            {"name": "lr", "type": "float", "label": "lr"},
            {"name": "weight_decay", "type": "float", "label": "weight_decay"},
            {"name": "clip", "type": "float", "label": "clip"},
            {"name": "log_interval", "type": "int", "label": "log_interval"},
        ],
    },
    # DPO 数据生成：data_tools/dpo/run_generate.py 编排 chosen→rejected→merge。
    # 非训练任务，无 loss 曲线；launcher 仅 python（脚本内部自行分片并行）。
    # model_path 留空自动探测 checkpoints/<variant>/sft/sft_best.pt。
    "dpo_data": {
        "script": "data_tools/dpo/run_generate.py",
        "label": "DPO 数据生成",
        "short": "DPO数据",
        "variant_flag": True,
        "upstream_stage": "sft",
        "launchers": ["python"],
        "auto": [],
        "fields": [
            {
                "name": "model_path",
                "type": "path",
                "label": "SFT 模型",
                # 面板字段名 model_path → 脚本旗标 --model-path（run_generate 连字符拼写）
                "flag": "model-path",
                "help": "留空自动探测 checkpoints/<variant>/sft/sft_best.pt（需先完成 SFT）",
            },
            {
                "name": "shards",
                "type": "int",
                "label": "并行分片",
                "help": "rejected 生成并行进程数（默认 4；小模型单进程 GPU 用不满，12GB+ 显存可开 8）",
            },
        ],
    },
}

# path 字段候选池: 显式 suggest 优先；缺省按字段名归类——模型/续训类只建议
# checkpoint 文件，其余 path 字段（目录/数据文件/教师目录等）不挂候选，
# 避免数据/目录框误列 checkpoints 下的模型文件
_CKPT_FIELD_NAMES = {"model", "model_path", "resume"}
for _t in _TASKS.values():
    for _f in _t["fields"]:
        if _f.get("suggest") is None and _f["type"] == "path" and _f["name"] in _CKPT_FIELD_NAMES:
            _f["suggest"] = "ckpt"


def _read_yaml_summary(rel: str) -> dict:
    """读模型 YAML 浅层摘要（供 lr 图 WSD 阶段线等展示）。

    只取已知展示键，不展开 extends（子配置若无对应段则返回空，前端自动降级
    为不画阶段线 —— 阶段线只是装饰，不阻塞任何功能）。
    """
    try:
        with open(_abs(rel), encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception:
        return {}
    out: dict[str, Any] = {}
    lr = doc.get("lr")
    if isinstance(lr, dict):
        out["lr"] = {
            k: lr[k]
            for k in ("type", "lr", "warmup_ratio", "stable_ratio", "min_lr_ratio")
            if k in lr
        }
    return out


def _rel(path: str) -> str:
    """相对 ROOT 的正斜杠路径（展示/接口用）。"""
    return os.path.relpath(path, ROOT_DIR).replace("\\", "/")


def _abs(rel: str) -> str:
    p = os.path.abspath(os.path.join(ROOT_DIR, rel))
    return p


# ── 配置文件权限分级（§4.1: 内置只读 / manual/my_configs 可写）──────
# user_model 模板已移除 (2026-09): base 即模板, 新建配置 = 复制内置另存为 manual/my_configs/
def _in_builtin_configs(rel: str) -> bool:
    # manual/configs/ 下所有 yaml 均为内置只读配置（读取/启动可见，写回 403）
    return rel.startswith("manual/configs/") and rel.endswith(".yaml")


def _config_entries() -> list[dict]:
    """manual/configs/ 全部 yaml + manual/my_configs/ 用户副本清单（读全部、写仅用户侧）。"""
    entries = []
    if os.path.isdir(CONFIG_DIR):
        for fn in sorted(os.listdir(CONFIG_DIR)):
            if fn.endswith(".yaml"):
                rel = f"manual/configs/{fn}"
                entries.append({"path": rel, "name": fn, "builtin": True, "writable": False})
    my_dir = MY_CFG_DIR
    if os.path.isdir(my_dir):
        for fn in sorted(os.listdir(my_dir)):
            if fn.endswith(".yaml"):
                rel = f"manual/my_configs/{fn}"
                entries.append({"path": rel, "name": fn, "builtin": False, "writable": True})
    return entries


def _resolve_config_path(rel: str) -> tuple[str, bool]:
    """解析 + 白名单校验（防任意文件读写）。返回 (绝对路径, 可写)。"""
    if not rel or ".." in rel.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail=f"非法配置路径: {rel!r}")
    p = _abs(rel)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {rel}")
    if _in_builtin_configs(rel):
        return p, False
    if rel.startswith("manual/my_configs/") and rel.endswith(".yaml"):
        return p, True
    raise HTTPException(status_code=403, detail=f"路径不在配置白名单: {rel}")


_config_lock = threading.Lock()


def _validate_config_text(content: str) -> list[dict]:
    """复用 gleamlm load_config（extends + scope 必读 + Pydantic）校验 YAML 文本。

    返回错误列表 [{loc, msg}]；空列表 = 通过。0 占位（d_model=0 等）由 Pydantic
    在此报出 —— 模板未填真实架构就保存/启动必失败（“忘改必报错”护栏）。
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        load_config(tmp, ROOT_DIR, scope="full")
        return []
    except ConfigValidationError as e:
        return [{"loc": "config", "msg": str(e)}]
    except ValidationError as e:
        return [
            {"loc": ".".join(str(x) for x in err["loc"]), "msg": str(err["msg"])}
            for err in e.errors()
        ]
    except yaml.YAMLError as e:
        return [{"loc": "yaml", "msg": f"YAML 语法错误: {e}"}]
    finally:
        os.unlink(tmp)


@router.get("/configs")
def list_configs() -> list[dict]:
    return _config_entries()


@router.get("/config")
def get_config(path: str) -> dict:
    p, writable = _resolve_config_path(path)
    with open(p, encoding="utf-8") as f:
        content = f.read()
    return {"path": path, "writable": writable, "content": content}


class ConfigWriteRequest(BaseModel):
    path: str
    content: str


@router.post("/config")
def save_config(req: ConfigWriteRequest) -> dict:
    """Pydantic 校验通过后原子写回（临时文件 + os.replace，防半写状态）。"""
    p, writable = _resolve_config_path(req.path)
    if not writable:
        raise HTTPException(status_code=403, detail="内置配置只读 — 用「另存为我的配置」复制后再改")
    errors = _validate_config_text(req.content)
    if errors:
        raise HTTPException(status_code=400, detail={"ok": False, "errors": errors})
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with _config_lock:
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(req.content)
        os.replace(tmp, p)
    return {"ok": True, "path": req.path}


class ConfigCopyRequest(BaseModel):
    source: str
    dest_name: str = ""


@router.post("/config/copy")
def copy_config(req: ConfigCopyRequest) -> dict:
    """内置配置「另存为我的配置」→ manual/my_configs/（git 忽略的用户副本目录）。"""
    p, _ = _resolve_config_path(req.source)
    name = req.dest_name.strip() or os.path.basename(req.source)
    if not name.endswith(".yaml"):
        name += ".yaml"
    os.makedirs(MY_CFG_DIR, exist_ok=True)
    dest = os.path.join(MY_CFG_DIR, name)
    if os.path.exists(dest):
        raise HTTPException(status_code=409, detail=f"已存在同名配置: manual/my_configs/{name}")
    with open(p, encoding="utf-8") as f:
        content = f.read()
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return {"ok": True, "path": _rel(dest), "writable": True}


# ── GPU 显存（面板侧统一探测; 对齐 pretrain.py _gpu_stats 的 nvidia-smi 回退）──
def gpu_info() -> list[dict]:
    """全部 GPU 显存占用（供 header 徽章与训练/推理互斥提示）。无 NVML 时回退 nvidia-smi。"""
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append(
                {
                    "index": i,
                    "used_gb": round(mem.used / 2**30, 1),
                    "total_gb": round(mem.total / 2**30, 1),
                }
            )
        pynvml.nvmlShutdown()
        return out
    except Exception:
        pass
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        out = []
        for line in r.stdout.strip().splitlines():
            idx, used, total = [x.strip() for x in line.split(",")]
            out.append(
                {
                    "index": int(idx),
                    "used_gb": round(int(used) / 1024, 1),
                    "total_gb": round(int(total) / 1024, 1),
                }
            )
        return out
    except Exception:
        return []


# ── checkpoint 树扫描（推理 tab 模型菜单 + tab 联动）──────────────────
def _scan_checkpoints() -> list[dict]:
    """扫描 checkpoints/ 下所有 *.pt，按 variant/stage 组织。

    checkpoints/{variant}/{stage}/*.pt → stage 为子目录名；variant 根下直放的
    checkpoint（final.pt 等）stage 为空串。新训练产物自然出现（tab 联动）。
    """
    base = os.path.join(ROOT_DIR, "checkpoints")
    files: list[dict] = []
    if not os.path.isdir(base):
        return files
    for variant in sorted(os.listdir(base)):
        vdir = os.path.join(base, variant)
        if not os.path.isdir(vdir):
            continue
        for dirpath, _dirs, names in os.walk(vdir):
            for fn in sorted(names):
                if not fn.endswith(".pt"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = _rel(full)
                stage = os.path.relpath(dirpath, vdir)
                if stage == ".":
                    stage = ""
                size_mb = round(os.path.getsize(full) / 2**20, 1)
                mtime = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y-%m-%d %H:%M")
                files.append(
                    {
                        "name": fn,
                        "path": rel,
                        "variant": variant,
                        "stage": stage,
                        "size_mb": size_mb,
                        "mtime": mtime,
                    }
                )
    return files


@router.get("/models")
def list_models() -> dict:
    files = _scan_checkpoints()
    groups: dict[str, list[dict]] = {}
    for f in files:
        v = f["variant"]
        if v not in groups:
            groups[v] = []
        groups[v].append(f)
    return {"files": files, "groups": groups}


# ── 训练进程管理 ─────────────────────────────────────────────────────
# 命令形态与 README 对齐（cwd=ROOT_DIR, sys.executable 保证同环境）；
# stdout+stderr 合并直写日志文件（tqdm 进度条写 stderr，两路都要收）。


class TrainStartRequest(BaseModel):
    task: str
    launcher: str = "python"
    nproc: int = 1
    variant: str = ""
    fields: dict[str, str] = {}
    run_name: str = ""


class TrainRun:
    def __init__(self, req: TrainStartRequest, cmd: list[str], run_id: str):
        self.run_id = run_id
        self.task = req.task
        self.variant = req.variant
        self.launcher = req.launcher
        self.cmd = cmd
        self.fields = {k: v for k, v in req.fields.items() if v not in (None, "")}
        self.log_path = os.path.join(LOGS_DIR, f"run_{run_id}.log")
        self.proc: subprocess.Popen | None = None
        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status = "starting"
        self.exit_code: int | None = None
        self.last_metric: dict[str, float] = {}
        self._stop_requested = False
        self._last_step = 0  # tqdm 行缺 step 时的帧计数器
        self._sentinel_seen = False  # 已解析到哨兵行 → 关闭 tqdm 帧回退（H19）
        self.total_steps: int | None = None  # 日志式行 step N/M 的 M（lr 图 WSD 阶段线坐标）

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "variant": self.variant,
            "launcher": self.launcher,
            "cmd": " ".join(shlex.quote(c) for c in self.cmd),
            "fields": self.fields,
            "log_path": _rel(self.log_path),
            "started_at": self.started_at,
            "status": self.status,
            "exit_code": self.exit_code,
            "last_metric": self.last_metric,
            "total_steps": self.total_steps,
        }


def _build_command(req: TrainStartRequest) -> tuple[list[str], dict[str, Any]]:
    """按任务注册表拼装命令（空字段不传 → 脚本默认 / YAML 裁决，铁律 2）。"""
    spec = _TASKS.get(req.task)
    if spec is None:
        raise HTTPException(
            status_code=400, detail=f"未知任务类型: {req.task} (可选: {sorted(_TASKS)})"
        )
    allowed = spec.get("launchers") or _LAUNCHERS
    if req.launcher not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"任务 {req.task} 不支持 launcher: {req.launcher} (可选: {allowed})",
        )
    if spec["variant_flag"] and not _config_path(req.variant):
        raise HTTPException(
            status_code=400,
            detail=f"配置模板不存在: {req.variant!r} (可选: {_variants()})",
        )

    if req.launcher == "python":
        cmd = [sys.executable]
    elif req.launcher == "torchrun":
        cmd = ["torchrun", f"--nproc_per_node={req.nproc or 1}"]
    else:
        cmd = ["deepspeed"]
    cmd.append(spec["script"])
    cmd.extend(spec.get("auto", []))
    if spec["variant_flag"]:
        # 面板侧模板（grpo/ppo）脚本无 --variant 参数，不落 CLI
        if spec.get("variant_cli", True):
            cmd += ["--variant", req.variant]
        if spec.get("config_dir_flag"):
            # 显式传 --config_dir：同名副本在 manual/my_configs 时脚本缺省目录会读错文件
            cmd += ["--config_dir", _rel(os.path.dirname(_config_path(req.variant)))]

    if req.launcher == "python":
        script_abs = os.path.join(ROOT_DIR, spec["script"])
        if not os.path.isfile(script_abs):
            raise HTTPException(status_code=404, detail=f"训练脚本不存在: {spec['script']}")
    # 提交字段（白名单 + 类型转换 + 必填检查）
    for field in spec["fields"]:
        raw = req.fields.get(field["name"])
        if raw in (None, ""):
            if field.get("required"):
                raise HTTPException(
                    status_code=400,
                    detail=f"缺少必填参数: {field['name']} — {field.get('help', '')}",
                )
            continue
        ftype = field["type"]
        fname = field.get("flag", field["name"])  # 脚本旗标拼写（默认 = 字段名）
        try:
            if ftype == "bool":
                if str(raw).strip().lower() in ("true", "1", "yes", "on"):
                    cmd.append(f"--{fname}")
                continue
            if ftype in ("int", "float"):
                conv: Any = int if ftype == "int" else float
                conv(str(raw))
            if ftype == "choice" and raw not in field.get("choices", []):
                raise ValueError(f"需为 {field.get('choices')} 之一")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"参数 {field['name']} 非法: {e}") from None
        cmd += [f"--{fname}", str(raw)]

    # 前置文件检查（数据目录等由脚本校验，只查明确存在的入口文件）
    if req.task in ("pretrain",):
        model_rel = req.fields.get("model", "")
        if not os.path.isfile(_abs(model_rel)):
            raise HTTPException(status_code=400, detail=f"配置 YAML 不存在: {model_rel}")
    return cmd, {"spec": spec["label"]}


# 指标解析: 面板是 tracker SQLite 唯一写入方（复用 tools/tracker.py schema）
#  - 哨兵行（首选, 结构化零正则）: @@GLEAM_METRIC {"split":"train","step":N,...}
#    emit/parse 契约见 gleamlm/utils/metrics.py —— 训练脚本与解析侧共用同一契约,
#    人类可读行的格式变化不再静默断曲线
#    注意: tqdm 帧以 \r 分帧（无换行）, 哨兵行常粘连在帧尾 —— parse_metric_line
#    按行内标记定位; 行首匹配曾使全部哨兵解析失败（H19 随之失效, DPO 曲线退化
#    为帧回退: x=分片位置 + postfix 值重复采样 = 阶梯形态）。
#  - 预训练 --no-pbar 日志式:  step N/M (pct%)  loss=.. lr=.. Xk tok/s  GPU:u/tG
#  - SFT/DPO tqdm 帧(\r 行):   N/M [..it/s, loss=.., lr=..]   （管道下每帧完整一行）
#    上述两条为回退路径: 哨兵行之前的老日志重放不受影响。
#  H19: 同一 run 见过哨兵后, tqdm 帧回退关闭 —— 帧 step 取 N/M 的 N（dataloader
#  位置）, accumulate>1 时比哨兵 global_step 大 accumulate 倍且先到, 会以「每 key
#  已写最大 step」把哨兵点滤掉（曲线/ETA 跟着 dataloader 位置走）; --no-pbar
#  日志式与哨兵同为真步坐标, 不受此限。
_PT_RE = re.compile(
    r"step (\d+)/(\d+) \([\d.]+%\)  loss=([\d.eE+-]+)  lr=([\d.eE+-]+)  "
    r"([\d.]+)k tok/s(?:  GPU:([\d.]+)/[\d.]+G)?"
)
# 训练内嵌周期验证行（manual/pretrain.py evaluate 打印，裸 CE 口径）:
#   Val step 64000: loss=3.20  ppl=24.56
_VAL_RE = re.compile(r"Val step (\d+): loss=([\d.eE+-]+)  ppl=([\d.eE+-]+)")
# 后训练 tqdm/手工帧: loss 后必跟逗号 + lr（sft/dpo 的 set_postfix 与
# opd/grpo/ppo/sft_lora 手工帧均为 [loss=.., lr=..] 同构）。loss 后要求逗号
# 是硬约束: pretrain 的 “Best model saved (val_loss=..) -> path” 行含 loss=..)
# 且后跟 ->, 宽松正则会把 lr 捕获成 '-' 导致 float 崩溃杀死解析线程。
# 前缀排除（负向后顾）: epoch 汇总行 “train_loss=.., lr=..”（sft）/“dpo_loss=..”
# 同样满足逗号 + lr 约束却不是帧 —— 曾以 fallback_step+1 作 step 产出假点
# （曲线 x 轴错位）, 带字词/下划线前缀的 loss 名一律不作帧。
_TQDM_RE = re.compile(r"(?<![A-Za-z0-9_])loss=([\d.eE+-]+),\s*lr=([\d.eE+-]+)")
_STEP_RE = re.compile(r"(\d+)/(\d+) \[")
# 后训练各阶段专属指标（设计文档 §7.3）: DPO 的 margin/acc、GRPO/PPO 的 reward/kl。
# **许可名单而非通配** —— 通配会把 tqdm 帧里将来任何 k=v 都当指标入库（含噪声）。
# 训练侧只要在 set_postfix 里多打印一个键，面板即自动可见（本文件之外零改动）。
# 注意: 值用严格浮点模式, 避免 "margin=-" 这类残帧把整行解析炸掉。
_EXTRA_KEYS = ("margin", "acc", "reward", "kl", "len")
_EXTRA_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(_EXTRA_KEYS) + r")="
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _parse_metric_lines(run: TrainRun, lines: list[str]) -> list[tuple[str, int, float]]:
    """行 → (key, step, value) 列表（sft/dpo 共用 tqdm loss/lr 正则）。"""
    out: list[tuple[str, int, float]] = []
    fallback_step = run._last_step
    for line in lines:
        try:
            rec = parse_metric_line(line)
            if rec is not None:
                # 哨兵行: 结构化直接入列, 不走正则（粘连帧尾形态由行内标记定位）
                run._sentinel_seen = True  # H19: 帧回退从此对该 run 关闭（见 _TQDM_RE 分支）
                step = int(rec["step"])
                if rec.get("split") == "val":
                    out += [
                        ("val_loss", step, float(rec["loss"])),
                        ("val_ppl", step, float(rec["ppl"])),
                    ]
                else:
                    out += [
                        ("loss", step, float(rec["loss"])),
                        ("lr", step, float(rec["lr"])),
                    ]
                    if rec.get("tok_per_s") is not None:
                        out.append(("tok_per_sec", step, float(rec["tok_per_s"])))
                    if rec.get("gpu_mem") is not None:
                        out.append(("gpu_mem", step, float(rec["gpu_mem"])))
                    # H4：后训练阶段专属指标（DPO 的 margin/acc、GRPO/PPO 的 reward/kl）。
                    # 键名与 _EXTRA_KEYS 同一份名单 —— 哨兵与正则回退两条路产出的键必须一致，
                    # 否则同一指标会因来源不同而落在不同的键上。
                    for _ek in _EXTRA_KEYS:
                        if rec.get(_ek) is not None:
                            out.append((_ek, step, float(rec[_ek])))
                total = int(rec.get("total") or 0)
                if total > 0:
                    run.total_steps = total
                fallback_step = step
                continue
            m = _PT_RE.search(line)
            if m:
                step, total, loss, lr, tok, gpu = m.groups()
                if int(total) > 0:
                    run.total_steps = int(total)
                out += [
                    ("loss", int(step), float(loss)),
                    ("lr", int(step), float(lr)),
                    ("tok_per_sec", int(step), float(tok) * 1e3),
                ]
                if gpu:
                    out.append(("gpu_mem", int(step), float(gpu)))
                fallback_step = int(step)
                continue
            m = _VAL_RE.search(line)
            if m:
                step, vloss, vppl = m.groups()
                out += [
                    ("val_loss", int(step), float(vloss)),
                    ("val_ppl", int(step), float(vppl)),
                ]
                fallback_step = int(step)
                continue
            m = _TQDM_RE.search(line)
            if m:
                # H19: 帧 step 是 dataloader 位置, 与哨兵 global_step 不同源 ——
                # 见过哨兵后帧退场（无哨兵的老日志回退照常, 见文件头格式说明）
                if run._sentinel_seen:
                    continue
                loss, lr = m.groups()
                s = _STEP_RE.search(line)
                step = int(s.group(1)) if s else fallback_step + 1
                fallback_step = step
                out += [("loss", step, float(loss)), ("lr", step, float(lr))]
                # 专属指标: 只在日志里真的出现时才产出（缺失 → 前端静默回落）
                for ekey, eval_ in _EXTRA_RE.findall(line):
                    out.append((ekey, step, float(eval_)))
        except (ValueError, TypeError, KeyError):
            # 单行解析失败（脏行/畸形帧/哨兵行缺字段）只丢弃该行, 不杀解析线程 ——
            # 否则面板指标管线会在长跑中途整体死掉。
            continue
    run._last_step = fallback_step
    return out


def _dedup_points(
    pending: list[tuple[str, int, float]], flushed: dict[str, int]
) -> list[tuple[str, int, float]]:
    """批内同 (key, step) 只留最后一条, 并滤掉已写过的 step。

    双写过渡期同一 step 先出旧格式行（低精度）、后出哨兵行（全精度）——
    两者几乎同时到达同一批, 留后者才能保证入库值来自哨兵；跨批重复
    （tqdm 帧/重放行）由 flushed（每 key 已写最大 step）承担。
    """
    latest: dict[tuple[str, int], float] = {}
    for k, s, v in pending:
        if s > flushed.get(k, -1):
            latest[(k, s)] = v
    return [(k, s, v) for (k, s), v in latest.items()]


def _parse_loop(run: TrainRun) -> None:
    """后台线程: 轮询日志文件增量 → 解析指标 → 写 tracker SQLite。

    同 step 去重（tqdm 帧/重放行可能重复）: 每 key 记录已写最大 step，只写增量。
    全量读 + 完整行号推进（不用字节 seek：读会撞上子进程写入中的半行，被截断的
    行会永久丢失）；文件尾部无 \n 的半行等写完整后再解析。进程退出后补读一次
    收尾再落定 exit_code（SSE 的 exit 事件等的是它，保证 total_steps/last_metric 已齐）。
    """
    tracker = ExperimentTracker("webui", DB_PATH)
    flushed: dict[str, int] = {}
    pending: list[tuple[str, int, float]] = []
    seen_lines = 0  # 已解析的完整行数（文件全量读，按行号增量推进）

    def pump() -> bool:
        """全量读文件 → 把新出现的完整行并入待写指标；返回是否有新行。"""
        nonlocal seen_lines
        try:
            with open(run.log_path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except FileNotFoundError:
            return False
        if not text:
            return False
        lines = text.replace("\r", "\n").split("\n")
        # 恒丢末段: 以 \n 结尾时末段是 split 产生的空串, 否则是写入中的半行 ——
        # 两者都不可解析。旧实现把尾空串计入 complete, seen_lines 恒超前 1,
        # 逐行新增的日志会与空串错位而被滤掉（长跑指标全丢的根因）。
        complete = lines[:-1]
        if not complete:
            return False
        if len(complete) <= seen_lines:
            return False
        new_lines = [ln for ln in complete[seen_lines:] if ln.strip()]
        seen_lines = len(complete)
        pending.extend(_parse_metric_lines(run, new_lines))
        return True

    def flush() -> None:
        """同 step 去重后写 tracker（pending 空则空转，无 sqlite 调用）。"""
        nonlocal pending
        fresh = _dedup_points(pending, flushed)
        pending = []
        if not fresh:
            return
        for k, s, v in fresh:
            tracker.log_metric(run.run_id, k, v, s)
            flushed[k] = s
            # val 指标要记住它对应的训练步: last_metric["step"] 会被后续训练行
            # 不断覆盖成最新训练步, 而 val ppl 是 eval_interval 前测的 —— 状态行
            # 显示 ppl 时必须用 val_step 对齐, 否则"多少步的 ppl"会错位。
            if k in ("val_loss", "val_ppl"):
                run.last_metric["val_step"] = s
        run.last_metric.update({"step": fresh[-1][1]})
        run.last_metric.update({k: v for k, s, v in fresh})

    while True:
        pump()
        flush()
        if run.proc is not None and run.proc.poll() is not None:
            pump()  # 收尾补读：进程已死文件不再增长，吞掉竞态窗口内的最后写入
            flush()
            run.exit_code = run.proc.returncode
            run.status = (
                "stopped"
                if run._stop_requested
                else ("finished" if run.proc.returncode == 0 else "failed")
            )
            tracker.finish_run(run.run_id, note=f"exit={run.proc.returncode}")
            tracker.close()
            break
        time.sleep(0.5)


class TrainManager:
    """单训练进程遥控器（预训练/后训练共用同一套机制）。"""

    def __init__(self) -> None:
        self._run: TrainRun | None = None
        self._lock = threading.Lock()

    def current(self) -> TrainRun | None:
        return self._run

    def start(self, req: TrainStartRequest) -> TrainRun:
        with self._lock:
            cur = self._run
            if cur is not None and cur.proc is not None and cur.proc.poll() is None:
                raise HTTPException(
                    status_code=409,
                    detail=(f"已有任务在运行: {cur.task}/{cur.run_id} — 先停止再启动新任务"),
                )
            cmd, _meta = _build_command(req)
            os.makedirs(LOGS_DIR, exist_ok=True)
            variant = req.variant or ""
            if variant and not _TASKS[req.task]["variant_flag"]:
                # 非 variant 任务 (pretrain): 忽略误传变体, run 名不带变体段
                variant = ""
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 变体名来自配置文件名（可含中文/空格等）→ run_id 段仅保留标识符
            # 字符（对齐删除路由 _RUN_ID_RE 契约）；原变体名仍进展示/DB config
            slug = re.sub(r"[^A-Za-z0-9_.-]", "-", variant)
            run_id = req.run_name or (f"{req.task}_{slug}_{ts}" if slug else f"{req.task}_{ts}")
            run = TrainRun(req, cmd, run_id)
            # 孤儿清扫: 新 run 开跑前, 把 DB 里仍标记 running 的旧条目归档为
            # interrupted（内存态随服务重启丢失, 服务重启后旧 run 不可能再被
            # parse_loop 收尾; 面板重启但训练进程仍活着时, 新 run 本就不该开）
            _mark_orphan_runs_interrupted()
            env = dict(os.environ)
            env.update(
                {
                    # 面板 + TensorBoard 接管监控（§10.4）; 用户手动 CLI 训练不受影响
                    "WANDB_MODE": "disabled",
                    # Windows 管道输出按 utf-8 收（否则 gbk 报错/乱码）
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            with open(run.log_path, "wb") as logf:
                run.proc = subprocess.Popen(
                    cmd,
                    cwd=ROOT_DIR,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=env,
                    creationflags=creationflags,
                )
            self._run = run
            model_rel = req.fields.get("model", "") or req.fields.get("model_path", "")
            # val_data 表单留空时回落 model YAML data.val_data（与 manual/pretrain.py
            # 裁决一致: --val_data None → YAML 值）。注入 run.fields 使 summary/状态行/
            # DB 快照三处同源 —— 都与真实进程参数一致（状态行 hasValData 判断依赖它）
            if req.task == "pretrain" and not run.fields.get("val_data") and model_rel:
                try:
                    yaml_cfg = load_config(_abs(model_rel), scope="training")
                except Exception:
                    yaml_cfg = None  # YAML 不可读: 静默（进程侧会暴露真实错误）
                if yaml_cfg is not None and yaml_cfg.data.val_data:
                    run.fields["val_data"] = yaml_cfg.data.val_data
            cfg: dict[str, Any] = {
                "task": req.task,
                "variant": req.variant,
                "launcher": req.launcher,
                "fields": dict(run.fields),
                "cmd": list(cmd),
            }
            if model_rel:
                cfg["yaml_summary"] = _read_yaml_summary(model_rel)
            tracker = ExperimentTracker("webui", DB_PATH)
            # create_run(config 为第一参数): 不能用位置传 run_id，run_name 已含
            tracker.create_run(
                config=cfg,
                tags=[req.task, req.variant] if req.variant else [req.task],
                run_name=run_id,
            )
            tracker.close()
            threading.Thread(target=_parse_loop, args=(run,), daemon=True).start()
            return run

    def stop(self) -> TrainRun:
        with self._lock:
            run = self._run
            if run is None or run.proc is None:
                raise HTTPException(status_code=409, detail="没有正在运行的任务")
            if run.proc.poll() is not None:
                raise HTTPException(status_code=409, detail="任务已结束")
            run._stop_requested = True
            run.status = "stopping"
            if os.name == "nt":
                # 进程树终止（torchrun 会拉起 N 个 worker，不能只杀 launcher）
                subprocess.run(
                    ["taskkill", "/pid", str(run.proc.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(run.proc.pid), 15)  # SIGTERM
            return run


manager = TrainManager()


def _mark_orphan_runs_interrupted() -> None:
    """DB 孤儿清扫: 所有 status='running' 的 run → 'interrupted'（服务重启后
    旧 parse_loop 已死, 无人会收尾; 保留日志文件供重放, note 记录孤儿标记）。"""
    try:
        con = sqlite3.connect(DB_PATH)
        try:
            con.execute(
                "UPDATE runs SET status='interrupted', note=note || '; orphaned by restart' "
                "WHERE status='running'"
            )
            con.commit()
        finally:
            con.close()
    except sqlite3.Error:
        pass  # DB 不存在/损坏时静默（首次运行等）


@router.get("/train/tasks")
def train_tasks() -> dict:
    """任务注册表元数据（前端动态渲染启动表单，字段/选项以本文件注册表为准）。"""
    return {
        "tasks": {
            k: {
                "script": v["script"],
                "label": v["label"],
                "short": v.get("short", v["label"]),
                "variant_flag": v["variant_flag"],
                "upstream_stage": v.get("upstream_stage", ""),
                "fields": v["fields"],
                "launchers": v.get("launchers"),
            }
            for k, v in _TASKS.items()
        },
        "variants": _variants(),
        "launchers": list(_LAUNCHERS),
        "configs": _config_entries(),
    }


# ── path 字段的留空回落值（启动表单旁显/预填）────────────────────────
# 面板把"留空会用什么"显式摆在字段旁, 消除隐性默认 —— 与各脚本的 CLI 缺省
# 裁决逐条对齐（manual/sft.py / dpo.py / opd.py / sft_lora.py、
# data_tools/dpo/run_generate.py、grpo/ppo 的 argparse default）。
_DEFAULT_SPEC_SCOPE = {"sft": "sft", "dpo": "dpo", "opd": "opd", "sft_lora": "lora"}


def _cands_abs(abs_dir: str, kind: str) -> list[dict]:
    """目录内与目标同类的条目：file=全部常规文件 / dir=全部子目录（点开头跳过）。

    按修改时间倒序（并列按名字），展示路径统一正斜杠；目录不可读返回空列表。
    """
    try:
        names = os.listdir(abs_dir)
    except OSError:
        return []
    rows: list[tuple[float, str]] = []
    for name in names:
        if name.startswith("."):
            continue
        p = os.path.join(abs_dir, name)
        if not (os.path.isfile(p) if kind == "file" else os.path.isdir(p)):
            continue
        try:
            rows.append((os.path.getmtime(p), name))
        except OSError:
            rows.append((0.0, name))
    rows.sort(key=lambda r: (-r[0], r[1]))
    return [{"path": _rel(os.path.join(abs_dir, name)), "name": name} for _, name in rows]


def _entry(path: str, kind: str = "", cand_dir: str = "") -> dict:
    """回落项：kind=file/dir 附存在性；cand_dir 附「推导目录 + 目录内同类条目候选」
    （推导字段 = 目录随模板固定，目录内文件全部列出可选）。保存目录不检测。"""
    item: dict[str, Any] = {"path": _rel(path)}
    if kind == "file":
        item["exists"] = os.path.isfile(path)
    elif kind == "dir":
        item["exists"] = os.path.isdir(path)
    if cand_dir:
        item["dir"] = _rel(cand_dir)
        item["cands"] = _cands_abs(cand_dir, kind or "file")
    return item


def _pretrain_prod(abs_ck: str) -> str:
    """预训练产物回落链：final.pt → best_model.pt → 兜底 final.pt（sft.py 同链）。"""
    for name in ("final.pt", "best_model.pt"):
        cand = os.path.join(abs_ck, name)
        if os.path.exists(cand):
            return cand
    return os.path.join(abs_ck, "final.pt")


def _task_defaults(task: str, variant: str) -> tuple[dict[str, dict], str]:
    """任务 × 变体 → (path 字段留空回落值, 变体配置的 checkpoint_dir 相对路径)。

    第二项供前端按配置目录前缀过滤 ckpt 候选 —— 变体 = 配置模板名, 产物归属
    以模板的 checkpoint_dir 为准; 非 YAML 链任务/不可解析时为空串。
    """
    spec = _TASKS.get(task)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"未知任务类型: {task}")
    if spec["variant_flag"] and not _config_path(variant):
        raise HTTPException(
            status_code=400,
            detail=f"配置模板不存在: {variant!r} (可选: {_variants()})",
        )
    out: dict[str, dict] = {}
    if task in ("grpo", "ppo"):
        # 模型 = 上游 SFT 产物（与 DPO 同链），保存目录随模板；不可解析时静默降级
        try:
            cfg = load_config(_config_path(variant), ROOT_DIR, scope="full")
        except Exception:
            return {}, ""
        ck = cfg.data.checkpoint_dir
        sft_dir = os.path.join(ck, "sft")
        out["model"] = _entry(os.path.join(sft_dir, "sft_best.pt"), "file", sft_dir)
        out["output_dir"] = _entry(os.path.join(ck, task))
        return out, (_rel(ck) if ck else "")
    if task == "dpo_data":
        # run_generate.py 硬拼 checkpoints/<variant>/sft/sft_best.pt（不读 YAML）；
        # 候选 = sft 阶段目录内全部文件
        sft_dir = os.path.join(ROOT_DIR, "checkpoints", variant, "sft")
        out["model_path"] = _entry(os.path.join(sft_dir, "sft_best.pt"), "file", sft_dir)
        return out, ""
    if task == "pretrain":
        # 配置模板（表单「配置模板」下拉，路径经前端转模板名传入）：数据目录（YAML
        # data_dir 前缀）与保存目录（checkpoint_dir）随模板推导预填；未选模板不预填
        if not variant:
            return out, ""
        try:
            cfg = load_config(_config_path(variant), ROOT_DIR, scope="full")
        except Exception:
            return {}, ""
        ck = cfg.data.checkpoint_dir
        if cfg.data.data_dir:
            out["data"] = _entry(str(cfg.data.data_dir))
        out["output_dir"] = _entry(str(ck))
        return out, (_rel(ck) if ck else "")
    scope = _DEFAULT_SPEC_SCOPE.get(task, "")
    if not scope:
        return out, ""  # 无 YAML 回落链的任务（含测试探针等扩展任务）
    try:
        cfg = load_config(_config_path(variant), ROOT_DIR, scope=scope)
    except Exception:
        return {}, ""  # YAML 不可读/缺键: 静默降级（启动时进程侧暴露真实错误）
    ck = cfg.data.checkpoint_dir
    if task == "sft":
        out["model_path"] = _entry(_pretrain_prod(ck), "file", ck)  # 候选 = ckpt 根目录内全部文件
        if cfg.sft.data_path:
            data_file = str(cfg.sft.data_path)
            out["data_path"] = _entry(data_file, "file", os.path.dirname(data_file))
        out["save_dir"] = _entry(os.path.join(ck, "sft"))
    elif task == "dpo":
        sft_dir = os.path.join(ck, "sft")
        out["model_path"] = _entry(os.path.join(sft_dir, "sft_best.pt"), "file", sft_dir)
        if cfg.dpo.data_path:
            data_file = str(cfg.dpo.data_path)
            out["data_path"] = _entry(data_file, "file", os.path.dirname(data_file))
        out["output_dir"] = _entry(os.path.join(ck, "dpo"))
    elif task == "opd":
        # 学生模型 = 上游 DPO 产物（dpo.py 落盘 dpo_best.pt）
        dpo_dir = os.path.join(ck, "dpo")
        out["model"] = _entry(os.path.join(dpo_dir, "dpo_best.pt"), "file", dpo_dir)
        if cfg.opd.data_path:
            data_file = str(cfg.opd.data_path)
            out["data"] = _entry(data_file, "file", os.path.dirname(data_file))
        if cfg.opd.teacher_model_path:
            teacher = str(cfg.opd.teacher_model_path).rstrip("\\/")
            # 候选 = 教师目录的父目录内全部子目录（同级目录均可选）
            out["teacher_model_path"] = _entry(teacher, "dir", os.path.dirname(teacher))
        out["output_dir"] = _entry(os.path.join(ck, "opd"))
    elif task == "sft_lora":
        # 基座模型 = 预训练产物（与 sft 同链：final.pt → best_model.pt）
        out["model"] = _entry(_pretrain_prod(ck), "file", ck)
        if cfg.lora.data_path:
            data_file = str(cfg.lora.data_path)
            out["data"] = _entry(data_file, "file", os.path.dirname(data_file))
        out["output_dir"] = _entry(os.path.join(ck, "lora"))
    return out, (_rel(ck) if ck else "")


@router.get("/train/defaults")
def train_defaults(task: str, variant: str = "") -> dict:
    """path 字段留空时的回落值（前端弹窗旁显 / 保存目录预填）。"""
    fields, ck_dir = _task_defaults(task, variant)
    return {"task": task, "variant": variant, "fields": fields, "checkpoint_dir": ck_dir}


class TrainStopRequest(BaseModel):
    pass


@router.post("/train/start")
def train_start(req: TrainStartRequest) -> dict:
    run = manager.start(req)
    return {
        "ok": True,
        "run_id": run.run_id,
        "log_path": _rel(run.log_path),
        "cmd": run.summary()["cmd"],
    }


@router.post("/train/stop")
def train_stop() -> dict:
    run = manager.stop()
    return {"ok": True, "run_id": run.run_id, "status": run.status}


@router.get("/train/status")
def train_status() -> dict:
    run = manager.current()
    if run is None:
        return {"running": False}
    return {"running": run.proc is not None and run.proc.poll() is None, **run.summary()}


def _iter_log_lines(log_path: str) -> tuple[list[str], int]:
    """读日志文件全部完整行（\r → \n 归一化，tqdm 帧逐行）。

    只返回以 \n 收尾的完整行：写入中的尾半行等下次拼全再发（seq 行号保持
    连续，杜绝残行与重复行）。返回 (行, 字节数)。
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except FileNotFoundError:
        return [], 0
    lines = text.replace("\r", "\n").split("\n")
    if lines:
        lines.pop()  # 尾项：以 \n 结尾时空串，否则为写入中的半行 —— 均不发
    return lines, len(text.encode("utf-8"))


@router.get("/train/stream")
async def train_stream(run_id: str, seq: int = 0):
    """SSE 日志流: 行号 seq 起回放 + 实时尾随；断线重连传 seq 续传。

    事件: data: {"type":"log","seq":n,"text":..} / {"type":"exit","code":..} /
    心跳注释行 : ping。结束条件: 文件 EOF 且进程已退出（或历史 run 空闲 2s）。
    """
    log_path = os.path.join(LOGS_DIR, f"run_{run_id}.log")
    run = manager.current()
    live = run is not None and run.run_id == run_id

    async def gen():
        last_pos = 0
        emitted = seq
        idle = 0.0
        sent_ping = 0.0
        while True:
            lines, size = await asyncio.to_thread(_iter_log_lines, log_path)
            if lines and last_pos == 0 and emitted < len(lines):
                # 首次/重连: 先回放全部已有行（读文件全量，按文件大小判增量不必要）
                last_pos = 1  # 标记已回放
            if lines:
                for i in range(emitted, len(lines)):
                    payload = {"type": "log", "seq": i, "text": lines[i]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    emitted = i + 1
            now = time.monotonic()
            if now - sent_ping > 15:
                yield ": ping\n\n"
                sent_ping = now
            proc_done = False
            if live and run is not None and run.proc is not None and run.proc.poll() is not None:
                proc_done = True
            if not live and idle > 2 and emitted >= len(lines):
                # 历史 run（live=False）: 回放完毕静默 2s 即结束（文件不再增长，
                # 不再要求文件为空 —— 否则有内容的文件会永远挂住不出 exit 事件）
                proc_done = True
            if proc_done:
                if lines and emitted < len(lines):
                    continue
                # 收尾补读: 进程退出瞬间的最后写入可能未被上轮读到（文件已不再增长）
                tail, _ = await asyncio.to_thread(_iter_log_lines, log_path)
                for i in range(emitted, len(tail)):
                    payload = {"type": "log", "seq": i, "text": tail[i]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    emitted = i + 1
                if live and run is not None:
                    # 等 _parse_loop 落定 exit_code/status（进程退出后 ≤0.5s 内完成）
                    for _ in range(6):
                        if run.exit_code is not None:
                            break
                        await asyncio.sleep(0.3)
                    payload = {"type": "exit", "code": run.exit_code, "status": run.status}
                else:
                    payload = {"type": "exit", "code": 0, "status": "idle"}
                yield f"data: {json.dumps(payload)}\n\n"
                return
            await asyncio.sleep(0.3)
            idle += 0.3

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/train/runs")
def train_runs() -> list[dict]:
    tracker = ExperimentTracker("webui", DB_PATH)
    try:
        return tracker.get_runs()
    finally:
        tracker.close()


# run_id 由 task_variant_ts 拼装（run_name 可自定义），仅允许普通标识符字符
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@router.delete("/train/runs/{run_id}")
def train_delete_run(run_id: str) -> dict:
    """删除历史 run：DB 记录 + 指标 + 日志文件（不可恢复）。

    仅拒绝 manager 正在运行中的 run；DB 残留 running 但进程已亡的孤儿条目
    （服务重启/进程被杀后的常态）可删。日志文件被存活进程持有句柄删不掉时
    保留文件并如实上报（Windows 文件锁语义）。
    """
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="非法 run_id")
    cur = manager.current()
    if (
        cur is not None
        and cur.run_id == run_id
        and cur.proc is not None
        and cur.proc.poll() is None
    ):
        raise HTTPException(status_code=409, detail=f"run 运行中不可删除: {run_id} — 请先停止")
    tracker = ExperimentTracker("webui", DB_PATH)
    try:
        removed = tracker.delete_run(run_id)
    finally:
        tracker.close()
    if not removed:
        raise HTTPException(status_code=404, detail=f"run 不存在: {run_id}")
    log_removed = False
    try:
        os.remove(os.path.join(LOGS_DIR, f"run_{run_id}.log"))
        log_removed = True
    except FileNotFoundError:
        pass
    except OSError:
        pass  # 进程仍持有句柄 → 保留文件，如实上报
    return {"ok": True, "run_id": run_id, "log_removed": log_removed}


@router.get("/train/metrics")
def train_metrics(run_id: str) -> dict:
    """单 run 全量指标（画曲线; 数据量小直接全拉, 前端自管 x 轴）。"""
    tracker = ExperimentTracker("webui", DB_PATH)
    try:
        return {"run_id": run_id, "series": tracker.get_metrics(run_id)}
    finally:
        tracker.close()

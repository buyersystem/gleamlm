"""webui 门户功能测试 — 后端 API 端到端（docs/前端面板设计.md）。

覆盖面：
  入口/静态资源 → config 分级与写回校验 → checkpoint 树/任务注册表 →
  训练全生命周期（start/SSE/metrics/runs/stop/冲突 409）→ 推理（空态 503、
  真实加载 + 非流式/流式 chat）。

不启动真实训练：注入「探针任务」probe，其脚本行为由环境变量 WEBUI_PROBE_MODE
控制（ok=秒退成功 / fail=exit 3 / sleep=挂起 20s），在 HTTP 层走完整进程链路；
tracker 数据库与日志目录重定向到 tmp_path，不污染 webui/logs/experiments.db。
依赖 fastapi/httpx（serve extra），缺失时整模块跳过。
"""

import json
import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import webui.routers.training as T  # noqa: E402
from webui.main import app  # noqa: E402

# ── 探针任务与脚本 ────────────────────────────────────────────────────
_PROBE_SCRIPT_REL = "webui/logs/_probe.py"
_PROBE_SRC = """\
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gleamlm.utils.metrics import emit_metric
mode = os.environ.get("WEBUI_PROBE_MODE", "ok")
print(f"probe start mode={mode}", flush=True)
if mode == "ok":
    # 回退路径: 手抄旧格式行, 验证正则兜底仍工作（哨兵行之前的老日志重放）
    # step N/M (pct%)  loss=.4f  两空格  lr=.6f  两空格  X.Xk tok/s  GPU:u/tG
    print("step 1/10 (10.0%)  loss=1.5000  lr=0.000100  12.3k tok/s  GPU:1.2/24.0G", flush=True)
    print("step 2/10 (20.0%)  loss=1.4100  lr=0.000095  12.5k tok/s  GPU:1.3/24.0G", flush=True)
    # 哨兵通道: 经真实 emit_metric 发射（非手抄）, 验证结构化解析路径
    emit_metric(split="train", step=3, total=10, loss=1.30, lr=0.000090, tok_per_s=12600.0, gpu_mem=1.35)
    emit_metric(split="val", step=3, loss=1.20, ppl=3.32)
    emit_metric(split="train", step=4, total=10, loss=1.29, lr=0.000085)
    # 同 step 双写过渡期: 旧格式行(低精度) 在前、哨兵行(全精度) 在后 ——
    # flush 批内去重后 DB 只留一条（值刻意一致, 消除同批/跨批时序差异）
    print("step 5/10 (50.0%)  loss=1.2800  lr=0.000080  12.8k tok/s  GPU:1.4/24.0G", flush=True)
    emit_metric(split="train", step=5, total=10, loss=1.28, lr=0.00008)
    print("probe done", flush=True)
elif mode == "fail":
    print("probe boom", flush=True)
    sys.exit(3)
elif mode == "sleep":
    print("probe sleep", flush=True)
    time.sleep(20)
print("exit(0)", flush=True)
"""

_MODEL_PT = "checkpoints/nano/sft/sft_best.pt"  # 推理加载样本（缺失则跳过）


@pytest.fixture(scope="module", autouse=True)
def probe_installed():
    """模块级一次：探针脚本落盘 + 注册 probe 任务；结束后清理。"""
    script_abs = os.path.join(T.ROOT_DIR, _PROBE_SCRIPT_REL)
    with open(script_abs, "w", encoding="utf-8") as f:
        f.write(_PROBE_SRC)
    T._TASKS["probe"] = {
        "script": _PROBE_SCRIPT_REL,
        "label": "Probe 探针（功能测试）",
        "variant_flag": False,
        "auto": [],
        "fields": [
            {
                "name": "model",
                "type": "path",
                "label": "配置 YAML",
                "help": "供 yaml_summary 注入断言（manual/configs/nano.yaml）",
            },
        ],
    }
    yield
    T._TASKS.pop("probe", None)
    os.remove(script_abs)


@pytest.fixture()
def api(monkeypatch):
    """每测试独立 tracker 库/日志目录。

    重定向到 webui/logs 同盘子目录（_rel() 相对 ROOT 需同盘；tmp_path 在 C:
    会触发 ntpath 跨盘 ValueError），测后整体删除，不碰真实 experiments.db。
    """
    import shutil
    import tempfile

    base = os.path.join(T.ROOT_DIR, "webui", "logs")
    os.makedirs(base, exist_ok=True)
    logs = tempfile.mkdtemp(prefix="_ft_logs_", dir=base)
    monkeypatch.setattr(T, "LOGS_DIR", logs)
    monkeypatch.setattr(T, "DB_PATH", os.path.join(logs, "experiments.db"))
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
    # daemon 解析线程/句柄延迟释放 → 重试删除，避免残留 _ft_logs_* 目录
    for _ in range(5):
        if not os.path.exists(logs):
            break
        shutil.rmtree(logs, ignore_errors=True)
        time.sleep(0.3)


def _wait_status(api, timeout: float = 15.0, want_running: bool = False) -> dict:
    """轮询 /api/train/status 直到满足 running 期望，返回响应体。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = api.get("/api/train/status").json()
        if bool(st.get("running")) == want_running:
            return st
        time.sleep(0.4)
    raise AssertionError(f"{timeout}s 内 status 未达 running={want_running}: {st}")


def _sse(api, method: str, url: str, **kw) -> list[dict]:
    """读 SSE 流到 exit/[DONE] 事件，返回事件列表。"""
    events: list[dict] = []
    with api.stream(method, url, **kw) as resp:
        assert resp.status_code == 200, resp.text
        for line in resp.iter_lines():
            if not line or line.startswith(":"):
                continue
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":  # 纯文本终止帧，非 JSON
                    events.append(payload)
                    break
                events.append(json.loads(payload))
                if events[-1].get("type") == "exit":
                    break
    return events


# ── 入口 / 静态资源 ──────────────────────────────────────────────────
def test_root_serves_index(api):
    r = api.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "GleamLM" in r.text


def test_static_assets_no_store(api):
    for path in [
        "/static/index.html",
        "/static/theme.css",
        "/static/util.js",
        "/static/sse.js",
        "/static/chart.js",
        "/static/trainer.js",
        "/static/chat.js",
        "/static/pretrain.js",
        "/static/posttrain.js",
        "/images/luna_night2.png",
    ]:
        r = api.get(path)
        assert r.status_code == 200, path
        assert r.headers.get("cache-control") == "no-store", path


def test_health_and_info(api):
    assert api.get("/health").json() == {"status": "ok", "app": "webui"}
    info = api.get("/api/info").json()
    assert info["tabs"] == ["pretrain", "posttrain", "inference"]
    assert info["train"] is True
    assert info["inference"]["loaded"] is False
    assert isinstance(info["gpu"], list)


# ── 配置管理（分级 / 白名单 / 校验 / 另存为）──────────────────────────
def test_config_listing_and_permissions(api):
    entries = api.get("/api/configs").json()
    by_path = {e["path"]: e for e in entries}
    nano = by_path["manual/configs/nano.yaml"]
    assert nano["builtin"] is True and nano["writable"] is False
    # user_model 模板已移除: base 即模板, 用户经「另存为」在 manual/my_configs/ 建配置
    assert "manual/configs/user_model.yaml" not in by_path
    assert all(e["builtin"] is False or e["writable"] is False for e in entries)


def test_config_read_guards(api):
    # 内置只读返回
    r = api.get("/api/config", params={"path": "manual/configs/nano.yaml"})
    assert r.status_code == 200
    body = r.json()
    assert body["writable"] is False and "model:" in body["content"]
    # 穿越 / 不存在 / 白名单外（存在但非 config 区 → 403；不存在优先 404）
    assert api.get("/api/config", params={"path": "../pyproject.toml"}).status_code == 400
    assert api.get("/api/config", params={"path": "manual/configs/nope.yaml"}).status_code == 404
    assert api.get("/api/config", params={"path": "pyproject.toml"}).status_code == 403


def test_builtin_write_forbidden(api):
    r = api.post(
        "/api/config",
        json={"path": "manual/configs/nano.yaml", "content": "model: {d_model: 0}"},
    )
    assert r.status_code == 403


def test_config_copy_save_roundtrip(api):
    dest = "manual/my_configs/_ft_probe_nano.yaml"
    # 预清理：本测试首步 copy 依赖 dest「不存在」。
    # 若上一次运行被中断（Ctrl-C / 进程被杀 / 沙箱拦截），finally 不会执行，
    # 遗留的 dest 会让首步误判重名返回 409，表现为与本改动无关的假失败。
    # 只在 finally 里清理不够——必须在开始前也清一次，测试才是幂等的。
    leftover = os.path.join(T.ROOT_DIR, dest)
    if os.path.exists(leftover):
        os.remove(leftover)
    try:
        r = api.post(
            "/api/config/copy",
            json={"source": "manual/configs/nano.yaml", "dest_name": "_ft_probe_nano.yaml"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["path"] == dest and r.json()["writable"] is True
        # 重名 → 409
        assert (
            api.post(
                "/api/config/copy",
                json={"source": "manual/configs/nano.yaml", "dest_name": "_ft_probe_nano.yaml"},
            ).status_code
            == 409
        )
        # 副本可读且可写
        body = api.get("/api/config", params={"path": dest}).json()
        assert body["writable"] is True and "model:" in body["content"]
        # 坏 YAML → 400 + Pydantic/YAML 错误明细
        r = api.post("/api/config", json={"path": dest, "content": "model: [1, 2"})
        assert r.status_code == 400
        assert r.json()["detail"]["ok"] is False and r.json()["detail"]["errors"]
        # 合法修改（lr 0.0004 → 0.0005）→ 原子写回并读回
        content = body["content"].replace("lr: 0.0004", "lr: 0.0005")
        r = api.post("/api/config", json={"path": dest, "content": content})
        assert r.status_code == 200, r.text
        assert "lr: 0.0005" in api.get("/api/config", params={"path": dest}).json()["content"]
    finally:
        os.remove(os.path.join(T.ROOT_DIR, dest))


# ── checkpoint 树 / 任务注册表 ───────────────────────────────────────
def test_models_tree(api):
    # 依赖真实训练产物：checkpoints/ 被 .gitignore 忽略，CI 全新 checkout 为空目录。
    # 与推理用例同模式（见 _MODEL_PT 用法）——产物缺失时跳过，而不是让门禁变红。
    if not os.path.isfile(os.path.join(T.ROOT_DIR, _MODEL_PT)):
        pytest.skip(f"缺少训练产物 {_MODEL_PT}（全新 checkout 不含 checkpoints/）")
    body = api.get("/api/models").json()
    assert body["files"], "checkpoints/ 下应有 .pt 产物"
    assert "nano" in body["groups"]
    for f in body["files"]:
        assert f["path"].startswith("checkpoints/") and "\\" not in f["path"]
        assert os.path.isfile(os.path.join(T.ROOT_DIR, f["path"]))
    # 多变体并存: lite/pro 的 sft_best.pt 与 nano 同名, 不能无脑取第一个匹配
    sft_files = [f for f in body["files"] if f["name"] == "sft_best.pt"]
    sft = next((f for f in sft_files if f["variant"] == "nano"), None)
    assert sft is not None, "models 树应含 nano 的 sft_best.pt"
    assert sft["stage"] == "sft" and sft["size_mb"] > 0


def test_task_registry(api):
    body = api.get("/api/train/tasks").json()
    builtin = {"pretrain", "sft", "dpo", "opd", "sft_lora", "grpo", "ppo"}
    assert builtin.issubset(body["tasks"].keys())
    assert body["tasks"]["probe"]["script"] == _PROBE_SCRIPT_REL  # 探针注入可见
    pre = body["tasks"]["pretrain"]
    assert pre["short"] == "预训练" and pre["fields"][0]["label"] == "配置模板"
    assert pre["fields"][0]["name"] == "model" and pre["fields"][0]["required"] is True
    assert pre["fields"][3]["label"] == "续训模型"  # resume（核心区第 4 项）
    # sft_lora：数据字段标签「数据」；grpo/ppo 挂配置模板（同 DPO 链），模型 = 上游
    # SFT 产物推导下拉，variant_cli=False 时模板不落 CLI
    assert body["tasks"]["sft_lora"]["fields"][1]["label"] == "数据"
    assert body["tasks"]["grpo"]["variant_flag"] is True
    assert body["tasks"]["ppo"]["variant_flag"] is True
    assert T._TASKS["grpo"].get("variant_cli") is False
    assert T._TASKS["ppo"].get("variant_cli") is False
    assert body["tasks"]["grpo"]["fields"][0]["suggest"] == "ckpt"
    assert body["tasks"]["sft"]["variant_flag"] is True
    # 变体 = 配置模板名：动态扫描 manual/configs + manual/my_configs（base.yaml 即模板）
    variants = body["variants"]
    assert {"base", "nano", "lite", "pro"} <= set(variants) and variants == sorted(variants)
    assert body["launchers"] == ["python", "torchrun", "deepspeed"]
    assert any(e["path"] == "manual/configs/nano.yaml" for e in body["configs"])
    # dpo_data 生成任务：仅 python launcher，字段全空即可启动（模型自动探测）
    dd = body["tasks"]["dpo_data"]
    assert dd["script"] == "data_tools/dpo/run_generate.py"
    assert dd["launchers"] == ["python"] and dd["variant_flag"] is True
    assert all(not f.get("required") for f in dd["fields"])
    # upstream_stage: 模型下拉默认聚焦的 checkpoints/<variant>/ 子目录
    assert body["tasks"]["sft"]["upstream_stage"] == ""
    assert body["tasks"]["dpo"]["upstream_stage"] == "sft"
    assert body["tasks"]["opd"]["upstream_stage"] == "dpo"


def test_train_defaults(api):
    """留空回落值端点：与脚本 CLI 缺省裁决对齐（模型/数据/保存目录/教师）。"""
    # 参数守卫：未知任务 / variant_flag 任务的非法变体
    assert api.get("/api/train/defaults", params={"task": "nope"}).status_code == 400
    assert (
        api.get("/api/train/defaults", params={"task": "sft", "variant": "zzz"}).status_code == 400
    )
    assert api.get("/api/train/defaults", params={"task": "sft"}).status_code == 400
    # sft: final.pt→best_model.pt 链 + 数据 + 保存目录（目录不带 exists 检测）
    body = api.get("/api/train/defaults", params={"task": "sft", "variant": "nano"}).json()
    assert body["checkpoint_dir"] == "checkpoints/nano"  # 模板 ckpt 目录下发（候选过滤）
    f = body["fields"]
    assert f["save_dir"]["path"] == "checkpoints/nano/sft" and "exists" not in f["save_dir"]
    assert f["model_path"]["path"].startswith("checkpoints/nano/")
    assert isinstance(f["model_path"]["exists"], bool)
    assert f["data_path"]["path"] == "data/nano/sft/sft_mix.jsonl"
    # 推导目录 + 目录内同类条目候选（前端渲染为「目录固定 + 文件可选」下拉）
    assert f["model_path"]["dir"] == "checkpoints/nano"
    assert f["data_path"]["dir"] == "data/nano/sft"
    assert isinstance(f["model_path"]["cands"], list)
    assert all("\\" not in d["path"] for d in f.values())  # 展示路径统一正斜杠
    # dpo: 硬拼 sft_best.pt（dpo.py 同链）+ 输出目录
    f = api.get("/api/train/defaults", params={"task": "dpo", "variant": "nano"}).json()["fields"]
    assert f["model_path"]["path"] == "checkpoints/nano/sft/sft_best.pt"
    assert f["model_path"]["dir"] == "checkpoints/nano/sft"
    assert f["output_dir"]["path"] == "checkpoints/nano/dpo"
    # opd: 学生模型 = 上游 DPO 产物 + 数据 + 教师目录 + 输出
    f = api.get("/api/train/defaults", params={"task": "opd", "variant": "nano"}).json()["fields"]
    assert f["model"]["path"] == "checkpoints/nano/dpo/dpo_best.pt"
    assert f["model"]["dir"] == "checkpoints/nano/dpo"
    assert f["teacher_model_path"]["path"] == "checkpoints/Qwen3-0.6B"
    assert f["teacher_model_path"]["dir"] == "checkpoints"  # 候选=同级目录
    assert f["output_dir"]["path"] == "checkpoints/nano/opd"
    # sft_lora: 基座模型 = 预训练产物链（与 sft 同源）+ 数据 + 输出目录
    f = api.get("/api/train/defaults", params={"task": "sft_lora", "variant": "nano"}).json()[
        "fields"
    ]
    assert f["model"]["path"].startswith("checkpoints/nano/")
    assert f["model"]["dir"] == "checkpoints/nano"
    # dpo_data: run_generate.py 硬拼的 SFT 产物
    f = api.get("/api/train/defaults", params={"task": "dpo_data", "variant": "nano"}).json()
    assert f["fields"]["model_path"]["path"] == "checkpoints/nano/sft/sft_best.pt"
    assert f["fields"]["model_path"]["dir"] == "checkpoints/nano/sft"
    # grpo/ppo: 模型 = 上游 SFT 产物（同 DPO 链）+ 保存目录随模板 + 无模板 400
    assert api.get("/api/train/defaults", params={"task": "grpo"}).status_code == 400
    body = api.get("/api/train/defaults", params={"task": "grpo", "variant": "nano"}).json()
    assert body["checkpoint_dir"] == "checkpoints/nano"
    f = body["fields"]
    assert f["model"]["path"] == "checkpoints/nano/sft/sft_best.pt"
    assert f["model"]["dir"] == "checkpoints/nano/sft"
    assert f["output_dir"]["path"] == "checkpoints/nano/grpo"
    f = api.get("/api/train/defaults", params={"task": "ppo", "variant": "nano"}).json()["fields"]
    assert f["model"]["path"] == "checkpoints/nano/sft/sft_best.pt"
    assert f["output_dir"]["path"] == "checkpoints/nano/ppo"
    # pretrain: 未选配置模板不预填；选定后数据/保存目录随模板推导 + ckpt 前缀下发
    assert api.get("/api/train/defaults", params={"task": "pretrain"}).json()["fields"] == {}
    body = api.get("/api/train/defaults", params={"task": "pretrain", "variant": "nano"}).json()
    assert body["checkpoint_dir"] == "checkpoints/nano"  # 续训模型候选过滤前缀
    assert body["fields"]["data"]["path"] == "data/nano/pretrain/train"
    assert body["fields"]["output_dir"]["path"] == "checkpoints/nano"
    # 自定义配置模板（变体=配置模板名）: manual/my_configs/ 副本可解析，回落链按
    # 模板 checkpoint_dir 计算（nano 副本改 checkpoint_dir 后路径整体跟随）
    tpl_path = os.path.join(T.MY_CFG_DIR, "_ft_probe_tpl.yaml")
    with open(os.path.join(T.CONFIG_DIR, "nano.yaml"), encoding="utf-8") as fh:
        tpl_src = fh.read().replace(
            "checkpoint_dir: checkpoints/nano", "checkpoint_dir: checkpoints/_ft_probe_ck"
        )
    with open(tpl_path, "w", encoding="utf-8") as fh:
        fh.write(tpl_src)
    try:
        body = api.get(
            "/api/train/defaults", params={"task": "sft", "variant": "_ft_probe_tpl"}
        ).json()
        assert body["checkpoint_dir"] == "checkpoints/_ft_probe_ck"
        assert body["fields"]["save_dir"]["path"] == "checkpoints/_ft_probe_ck/sft"
        assert body["fields"]["model_path"]["path"].startswith("checkpoints/_ft_probe_ck/")
    finally:
        os.remove(tpl_path)


# ── 训练生命周期 ─────────────────────────────────────────────────────
def test_train_start_validation(api):
    assert api.post("/api/train/start", json={"task": "nope"}).status_code == 400
    assert (
        api.post("/api/train/start", json={"task": "probe", "launcher": "bad"}).status_code == 400
    )
    assert api.post("/api/train/start", json={"task": "sft", "variant": "zzz"}).status_code == 400
    # grpo 挂配置模板后：缺模板 400
    assert api.post("/api/train/start", json={"task": "grpo", "fields": {}}).status_code == 400
    r = api.post("/api/train/start", json={"task": "pretrain", "fields": {}})
    assert r.status_code == 400 and "model" in r.json()["detail"]  # 必填缺失
    r = api.post(
        "/api/train/start",
        json={"task": "pretrain", "fields": {"model": "manual/configs/nope.yaml"}},
    )
    assert r.status_code == 400 and "不存在" in r.json()["detail"]


def test_train_start_custom_template_cmd(api):
    """变体=配置模板名：启动命令带 --variant + --config_dir（按副本所在目录）。"""
    tpl_path = os.path.join(T.MY_CFG_DIR, "_ft_probe_tpl.yaml")
    with open(os.path.join(T.CONFIG_DIR, "nano.yaml"), encoding="utf-8") as fh:
        src = fh.read()
    with open(tpl_path, "w", encoding="utf-8") as fh:
        fh.write(src)
    try:
        cmd, _meta = T._build_command(T.TrainStartRequest(task="sft", variant="_ft_probe_tpl"))
        assert cmd[-2:] == ["--config_dir", "manual/my_configs"] and "--variant" in cmd
        # 内置模板：--config_dir 指向 manual/configs（显式传，不靠脚本缺省）
        cmd, _meta = T._build_command(T.TrainStartRequest(task="sft", variant="nano"))
        assert cmd[-2:] == ["--config_dir", "manual/configs"]
        # dpo_data 脚本无 --config_dir 参数：不注入（仅 --variant，目录按名称约定）
        cmd, _meta = T._build_command(T.TrainStartRequest(task="dpo_data", variant="nano"))
        assert "--config_dir" not in cmd
        # 字段名→脚本旗标覆写：run_generate 用连字符 --model-path（默认拼写会是 --model_path）
        cmd, _meta = T._build_command(
            T.TrainStartRequest(
                task="dpo_data",
                variant="nano",
                fields={"model_path": "checkpoints/nano/sft/sft_best.pt"},
            )
        )
        assert "--model-path" in cmd and "--model_path" not in cmd
        # grpo/ppo：脚本无 --variant 参数，模板仅供推导，不落 CLI
        cmd, _meta = T._build_command(
            T.TrainStartRequest(
                task="grpo",
                variant="nano",
                fields={"model": "checkpoints/nano/sft/sft_best.pt", "data": "data/x.jsonl"},
            )
        )
        assert "--variant" not in cmd and "--config_dir" not in cmd
        assert "--model" in cmd and "--data" in cmd
    finally:
        os.remove(tpl_path)


def test_train_stop_when_idle(api):
    r = api.post("/api/train/stop")
    assert r.status_code == 409


def test_train_lifecycle_ok(api, monkeypatch):
    """成功 run：SSE 日志回放 + exit 事件 → status/metrics/runs 落库含 yaml_summary。"""
    monkeypatch.setenv("WEBUI_PROBE_MODE", "ok")
    r = api.post(
        "/api/train/start",
        json={
            "task": "probe",
            "run_name": "ft_ok",
            "fields": {"model": "manual/configs/nano.yaml"},
        },
    )
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert run_id == "ft_ok"
    assert "webui/logs/_probe.py" in r.json()["cmd"]

    events = _sse(api, "GET", f"/api/train/stream?run_id={run_id}&seq=0")
    logs = [e for e in events if e["type"] == "log"]
    assert logs and logs[0]["seq"] == 0  # 行号从 0 起
    assert [e["seq"] for e in logs] == list(range(len(logs)))  # seq 连续
    assert any("step 1/10 (10.0%)" in e["text"] for e in logs)
    exit_ev = events[-1]
    assert exit_ev["type"] == "exit" and exit_ev["code"] == 0
    assert exit_ev["status"] == "finished"

    st = api.get("/api/train/status").json()
    assert st["running"] is False and st["status"] == "finished"
    assert st["exit_code"] == 0 and st["total_steps"] == 10
    assert st["last_metric"]["loss"] == 1.28  # 尾步 = step5 双写（去重后单条）

    runs = {x["id"]: x for x in api.get("/api/train/runs").json()}
    run = runs[run_id]
    assert run["status"] == "finished" and run["note"] == "exit=0"
    assert run["config"]["task"] == "probe"
    assert run["config"]["yaml_summary"]["lr"]["type"] == "wsd"  # nano.yaml 注入
    assert any(c.endswith("_probe.py") for c in run["config"]["cmd"])

    series = api.get("/api/train/metrics", params={"run_id": run_id}).json()["series"]
    # 1-2 步走旧格式正则回退; 3-4 步走哨兵行（含 val 与缺省字段）;
    # 5 步同 step 双写（旧格式行 + 哨兵行）: 批内去重后每 step 仅一条
    assert [s for s, _ in series["loss"]] == [1, 2, 3, 4, 5]
    assert series["loss"][:4] == [[1, 1.5], [2, 1.41], [3, 1.3], [4, 1.29]]
    assert series["loss"][4] == [5, 1.28]
    assert series["lr"][:4] == [[1, 0.0001], [2, 0.000095], [3, 0.00009], [4, 0.000085]]
    assert series["lr"][4] == [5, 0.00008]
    assert series["tok_per_sec"] == [[1, 12300.0], [2, 12500.0], [3, 12600.0], [5, 12800.0]]
    assert series["gpu_mem"] == [[1, 1.2], [2, 1.3], [3, 1.35], [5, 1.4]]
    assert series["val_loss"] == [[3, 1.2]]
    assert series["val_ppl"] == [[3, 3.32]]


def test_train_fail_run(api, monkeypatch):
    """失败 run：exit code 透传 SSE，runs 落库 failed 供前端失败态展示。"""
    monkeypatch.setenv("WEBUI_PROBE_MODE", "fail")
    r = api.post("/api/train/start", json={"task": "probe", "run_name": "ft_fail"})
    assert r.status_code == 200
    events = _sse(api, "GET", "/api/train/stream?run_id=ft_fail&seq=0")
    assert events[-1]["type"] == "exit" and events[-1]["code"] == 3
    assert events[-1]["status"] == "failed"
    st = api.get("/api/train/status").json()
    assert st["status"] == "failed" and st["exit_code"] == 3
    runs = {x["id"]: x for x in api.get("/api/train/runs").json()}
    assert runs["ft_fail"]["note"] == "exit=3"


def test_train_stop_and_conflict(api, monkeypatch):
    """挂起 run：并发 start 409 → stop 杀进程树 → status stopped 落定。"""
    monkeypatch.setenv("WEBUI_PROBE_MODE", "sleep")
    assert (
        api.post("/api/train/start", json={"task": "probe", "run_name": "ft_sleep"}).status_code
        == 200
    )
    st = _wait_status(api, want_running=True)
    assert st["run_id"] == "ft_sleep" and st["status"] == "starting"
    r = api.post("/api/train/start", json={"task": "probe", "run_name": "ft_conflict"})
    assert r.status_code == 409 and "在运行" in r.json()["detail"]
    r = api.post("/api/train/stop")
    assert r.status_code == 200 and r.json()["status"] == "stopping"
    # Windows taskkill /T /F 应秒杀；失败时 20s 后自然退出, 两种结局都 status=stopped
    # （parse_loop 在进程死后 ≤0.5s 落定 status，须等到 stopped 而非仅 running False）
    deadline = time.monotonic() + 40.0
    while time.monotonic() < deadline:
        st = api.get("/api/train/status").json()
        if st["status"] == "stopped":
            break
        time.sleep(0.4)
    assert st["status"] == "stopped"
    runs = {x["id"]: x for x in api.get("/api/train/runs").json()}
    assert runs["ft_sleep"]["status"] == "finished"
    assert runs["ft_sleep"]["note"].startswith("exit=")
    assert api.get("/api/train/metrics", params={"run_id": "ft_sleep"}).status_code == 200


def test_train_orphan_adopt_and_stop(api):
    """遗留接管: DB running + 进程真存活 → startup_recovery 认领为可停止任务。

    模拟"面板被强杀"：训练进程仍在跑（真 Python 子进程），DB 里留下带 pid
    的 running 条目; 新面板启动时 adopt 成接管 run → status 可查、stop 可杀。
    """
    script_rel = "webui/logs/_ft_orphan.py"
    script_abs = os.path.join(T.ROOT_DIR, script_rel)
    with open(script_abs, "w", encoding="utf-8") as f:
        f.write("import time\ntime.sleep(120)\n")
    proc = subprocess.Popen([sys.executable, script_abs])
    log_path = os.path.join(T.LOGS_DIR, "run_ft_orphan.log")
    try:
        # 历史行（上一代面板已解析入库）—— 接管后不得重放（metrics 无唯一约束）
        with open(log_path, "w", encoding="utf-8") as f:
            f.write('@@GLEAM_METRIC {"split":"train","step":1,"loss":1.5}\n')
        tracker = T.ExperimentTracker("webui", T.DB_PATH)
        try:
            tracker.create_run(
                config={
                    "task": "probe",
                    "variant": "",
                    "launcher": "python",
                    "fields": {},
                    "cmd": [sys.executable, script_rel],
                    "pid": proc.pid,
                },
                tags=["probe"],
                run_name="ft_orphan",
            )
        finally:
            tracker.close()
        T.startup_recovery()
        run = T.manager.current()
        assert run is not None and run.run_id == "ft_orphan" and run.adopted is True
        st = api.get("/api/train/status").json()
        assert st["running"] is True and st["adopted"] is True
        assert st["run_id"] == "ft_orphan" and st["task"] == "probe"
        # 停止全链路复用: taskkill /T /F 杀接管进程 → parse_loop 收尾 stopped
        r = api.post("/api/train/stop")
        assert r.status_code == 200 and r.json()["status"] == "stopping"
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            st = api.get("/api/train/status").json()
            if st["status"] == "stopped":
                break
            time.sleep(0.4)
        assert st["status"] == "stopped" and st["exit_code"] == 1
        runs = {x["id"]: x for x in api.get("/api/train/runs").json()}
        assert runs["ft_orphan"]["note"] == "exit=1"
        # 历史行被跳过: 接管不清重放（metrics 里不应出现预写的 step 1 点）
        series = api.get("/api/train/metrics", params={"run_id": "ft_orphan"}).json()
        assert series["series"] == {}
    finally:
        if proc.poll() is None:
            proc.kill()
        T.manager._run = None  # 不把接管 run 泄漏给后续测试
        os.remove(script_abs)
        if os.path.exists(log_path):
            os.remove(log_path)


def test_train_orphan_dead_marked_interrupted(api):
    """遗留死进程: DB running 但 pid 已亡 → 不接管, 归档 interrupted + note。"""
    tracker = T.ExperimentTracker("webui", T.DB_PATH)
    try:
        tracker.create_run(
            config={
                "task": "probe",
                "variant": "",
                "launcher": "python",
                "fields": {},
                "cmd": [sys.executable, "manual/sft_lora.py"],
                "pid": 999999,
            },
            tags=["probe"],
            run_name="ft_orphan_dead",
        )
    finally:
        tracker.close()
    T.startup_recovery()
    assert T.manager.current() is None  # 未接管
    runs = {x["id"]: x for x in api.get("/api/train/runs").json()}
    dead = runs["ft_orphan_dead"]
    assert dead["status"] == "interrupted" and "orphaned" in dead["note"]


# ── run 删除 ──────────────────────────────────────────────────────────
def test_train_delete_run_after_finish(api, monkeypatch):
    """删除已完成 run：DB 记录 + 指标 + 日志文件全清；非法/不存在 id 被拒。"""
    monkeypatch.setenv("WEBUI_PROBE_MODE", "ok")
    r = api.post("/api/train/start", json={"task": "probe", "run_name": "ft_del"})
    assert r.status_code == 200
    _sse(api, "GET", "/api/train/stream?run_id=ft_del&seq=0")
    log_path = os.path.join(T.LOGS_DIR, "run_ft_del.log")
    assert os.path.isfile(log_path)
    # 非法 id：路径穿越形态在路由层即被拒（404，不命中删除路由）；非法字符 400
    assert api.delete("/api/train/runs/..%2Fx").status_code == 404
    assert api.delete("/api/train/runs/ft%20x").status_code == 400
    assert api.delete("/api/train/runs/never_ran").status_code == 404
    r = api.delete("/api/train/runs/ft_del")
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "run_id": "ft_del", "log_removed": True}
    ids = [x["id"] for x in api.get("/api/train/runs").json()]
    assert "ft_del" not in ids
    assert api.get("/api/train/metrics", params={"run_id": "ft_del"}).json()["series"] == {}
    assert not os.path.exists(log_path)


def test_train_delete_running_conflict(api, monkeypatch):
    """运行中的 run 拒绝删除（409）；停止落定后可删。"""
    monkeypatch.setenv("WEBUI_PROBE_MODE", "sleep")
    r = api.post("/api/train/start", json={"task": "probe", "run_name": "ft_del_run"})
    assert r.status_code == 200
    _wait_status(api, want_running=True)
    r = api.delete("/api/train/runs/ft_del_run")
    assert r.status_code == 409 and "运行中" in r.json()["detail"]
    assert api.post("/api/train/stop").status_code == 200
    deadline = time.monotonic() + 40.0
    while time.monotonic() < deadline:
        st = api.get("/api/train/status").json()
        if st["status"] == "stopped":
            break
        time.sleep(0.4)
    assert st["status"] == "stopped"
    r = api.delete("/api/train/runs/ft_del_run")
    assert r.status_code == 200 and r.json()["log_removed"] is True
    ids = [x["id"] for x in api.get("/api/train/runs").json()]
    assert "ft_del_run" not in ids
    assert not os.path.exists(os.path.join(T.LOGS_DIR, "run_ft_del_run.log"))


# ── 指标批内去重（同 step 双写过渡期契约）───────────────────────────
def test_dedup_points_same_batch_keeps_last():
    """同批同 (key, step): 留最后一条 —— 哨兵行在后, 全精度覆盖旧格式行低精度。"""
    pts = [("loss", 5, 1.28), ("loss", 5, 1.2786666), ("lr", 5, 0.00008)]
    assert T._dedup_points(pts, {}) == [("loss", 5, 1.2786666), ("lr", 5, 0.00008)]


def test_dedup_points_filters_flushed_steps():
    """跨批保护: 已写过的 step 一律丢弃（重复 tqdm 帧/重放行不再入库）。"""
    pts = [("loss", 3, 1.0), ("loss", 4, 1.1)]
    assert T._dedup_points(pts, {"loss": 3}) == [("loss", 4, 1.1)]


# ── H19: 哨兵出现后关闭 tqdm 帧回退 ──────────────────────────────────
def test_sentinel_closes_tqdm_fallback():
    """H19: 见过哨兵后 tqdm 帧不再产点。

    帧 step 是 dataloader 位置（N/M 的 N）—— accumulate_grad>1 时比哨兵
    global_step 大 accumulate 倍且先到, 会以「已写最大 step」把哨兵点滤掉;
    见哨兵即关回退, x 轴回到 global_step 单刻度。
    """
    run = T.TrainRun(T.TrainStartRequest(task="probe"), ["echo"], "t_h19")
    lines = [
        # 首个哨兵之前: 帧照常入列（哨兵之前的老日志兼容窗口）
        "4/100 [00:01<00:25, 3.9it/s, loss=1.5000, lr=5.00e-07]",
        '@@GLEAM_METRIC {"split":"train","step":1,"total":25,"loss":1.5,"lr":5e-07,"reward":0.25}',
        # 哨兵之后: 帧必须退场（否则 dataloader 位置的 8/12 会挤掉后续哨兵点）
        "8/100 [00:02<00:24, 3.9it/s, loss=1.4000, lr=5.00e-07]",
        "12/100 [00:03<00:23, 3.9it/s, loss=1.3000, lr=5.00e-07]",
    ]
    pts = T._parse_metric_lines(run, lines)
    assert {s for k, s, _ in pts if k == "loss"} == {4, 1}
    assert ("reward", 1, 0.25) in pts  # 契约字段直接入列（grpo 的 reward 走这里）


def test_glued_sentinel_line_parsed_and_gates_frames():
    """回归: 哨兵粘连在 tqdm 帧尾时仍须识别（行首匹配曾致哨兵全灭）。

    真实日志: tqdm 帧以 \r 分帧无换行, 哨兵 print 直接接帧尾 —— 未识别时
    H19 门控不生效, 曲线退化为帧回退（x=分片位置、postfix 值重复采样 = 阶梯）。
    """
    run = T.TrainRun(T.TrainStartRequest(task="probe"), ["echo"], "t_glue")
    lines = [
        "4/100 [00:01<00:25, 3.9it/s, loss=1.5000, lr=5.00e-07]",
        "4/100 [00:01<00:25, 3.9it/s, loss=1.5000, lr=5.00e-07]"
        '@@GLEAM_METRIC {"split":"train","step":1,"total":50,'
        '"loss":1.4999256,"lr":5e-07,"margin":0.1,"acc":1.0}',
        "8/100 [00:02<00:24, 3.9it/s, loss=1.4000, lr=5.00e-07]",
    ]
    pts = T._parse_metric_lines(run, lines)
    losses = [p for p in pts if p[0] == "loss"]
    assert ("loss", 4, 1.5) in losses  # 哨兵之前的帧照常（兼容窗口）
    assert ("loss", 1, 1.4999256) in losses  # 粘连哨兵: step=global_step, 全精度
    assert ("margin", 1, 0.1) in pts and ("acc", 1, 1.0) in pts
    assert not [p for p in pts if p[0] == "loss" and p[1] == 8]  # 哨兵后帧退场
    assert run.total_steps == 50


def test_tqdm_fallback_kept_for_legacy_logs():
    """无哨兵的老日志: 帧回退照常（H19 修复只对「见过哨兵」的 run 生效）。"""
    run = T.TrainRun(T.TrainStartRequest(task="probe"), ["echo"], "t_h19b")
    pts = T._parse_metric_lines(run, ["3/50 [00:01<00:20, 4.0it/s, loss=2.0000, lr=1.00e-04]"])
    assert ("loss", 3, 2.0) in pts and ("lr", 3, 1e-4) in pts


def test_epoch_summary_line_not_parsed_as_frame():
    """回归: sft/dpo 的 epoch 汇总行不得命中帧回退。

    “Epoch 0: train_loss=.., lr=..” 满足 loss=.., lr=.. 的逗号约束却不是帧——
    曾以 fallback_step+1 作 step 产出假点（曲线 x 轴错位, 如 3355 假点）。
    """
    run = T.TrainRun(T.TrainStartRequest(task="probe"), ["echo"], "t_ep_sum")
    lines = [
        "4/100 [00:01<00:25, 3.9it/s, loss=1.5000, lr=5.00e-07]",
        "Epoch 0: train_loss=2.8616, lr=7.80e-05",
        "Epoch 1: dpo_loss=2.5360, lr=2.96e-05",
    ]
    pts = T._parse_metric_lines(run, lines)
    assert [p for p in pts if p[0] == "loss"] == [("loss", 4, 1.5)]
    assert [p for p in pts if p[0] == "lr"] == [("lr", 4, 5e-07)]


# ── 推理 ─────────────────────────────────────────────────────────────
def test_inference_unloaded_guards(api):
    assert api.get("/v1/models/status").json()["loaded"] is False
    msg = {"messages": [{"role": "user", "content": "hi"}]}
    r = api.post("/v1/chat/completions", json=msg)
    assert r.status_code == 503 and "未加载" in r.json()["detail"]
    assert api.post("/v1/completions", json={"prompt": "hi"}).status_code == 503
    assert (
        api.post("/v1/models/load", json={"model_path": "checkpoints/nope.pt"}).status_code == 404
    )


def test_inference_load_and_chat(api):
    """真模型加载 + 非流式/流式一致性（temperature=0 确定性输出）。"""
    pt = os.path.join(T.ROOT_DIR, _MODEL_PT)
    if not os.path.isfile(pt):
        pytest.skip(f"缺少推理样本 {_MODEL_PT}")
    try:
        return _load_and_chat(api)
    finally:
        # 卸载恢复空态（server 模块级单例，跨测试残留会破坏 unloaded 用例）
        from webui.routers import inference as INF

        INF.server.unload()


def _load_and_chat(api):
    r = api.post("/v1/models/load", json={"model_path": _MODEL_PT})
    assert r.status_code == 200, r.text
    assert r.json()["cached"] is False
    assert (
        api.post("/v1/models/load", json={"model_path": _MODEL_PT}).json()["cached"] is True
    )  # 热切换幂等
    st = api.get("/v1/models/status").json()
    assert st["loaded"] is True and st["model_path"].endswith("sft_best.pt")
    assert st["params_m"] > 0

    body = {"messages": [{"role": "user", "content": "1+1=?"}], "temperature": 0, "max_tokens": 16}
    r = api.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    text = r.json()["choices"][0]["message"]["content"]
    assert isinstance(text, str)

    events = _sse(api, "POST", "/v1/chat/completions", json={**body, "stream": True})
    streamed = "".join(
        e["choices"][0]["delta"]["content"]
        for e in events
        if isinstance(e, dict) and e.get("choices") and e["choices"][0]["delta"].get("content")
    )
    assert events[-1] == "[DONE]"
    assert streamed == text  # 流式拼接 == 非流式（T=0 无采样随机）


def test_inference_unload_releases(api):
    """load → unload：身份校验（409 防误卸）+ 空态/503 复归 + 幂等。"""
    pt = os.path.join(T.ROOT_DIR, _MODEL_PT)
    if not os.path.isfile(pt):
        pytest.skip(f"缺少推理样本 {_MODEL_PT}")
    try:
        assert api.post("/v1/models/load", json={"model_path": _MODEL_PT}).status_code == 200
        # 身份不符 → 409，且不误卸当前模型
        r = api.post("/v1/models/unload", json={"model_path": "checkpoints/nope.pt"})
        assert r.status_code == 409 and "不一致" in r.json()["detail"]
        assert api.get("/v1/models/status").json()["loaded"] is True
        # 正确身份 → 卸载成功；再卸幂等
        r = api.post("/v1/models/unload", json={"model_path": _MODEL_PT})
        assert r.status_code == 200 and r.json()["unloaded"] is True
        assert api.get("/v1/models/status").json()["loaded"] is False
        assert (
            api.post("/v1/models/unload", json={"model_path": _MODEL_PT}).json()["unloaded"]
            is False
        )
        msg = {"messages": [{"role": "user", "content": "hi"}]}
        assert api.post("/v1/chat/completions", json=msg).status_code == 503
    finally:
        from webui.routers import inference as INF

        INF.server.unload()

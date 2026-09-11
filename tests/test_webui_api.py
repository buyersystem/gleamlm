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
mode = os.environ.get("WEBUI_PROBE_MODE", "ok")
print(f"probe start mode={mode}", flush=True)
if mode == "ok":
    # 日志式指标行, 与 manual/pretrain.py --no-pbar 逐字节同构 (_PT_RE):
    # step N/M (pct%)  loss=.4f  两空格  lr=.6f  两空格  X.Xk tok/s  GPU:u/tG
    print("step 1/10 (10.0%)  loss=1.5000  lr=0.000100  12.3k tok/s  GPU:1.2/24.0G", flush=True)
    print("step 2/10 (20.0%)  loss=1.4100  lr=0.000095  12.5k tok/s  GPU:1.3/24.0G", flush=True)
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
    # user_model 模板已移除: base 即模板, 用户经「另存为」在 my_configs/ 建配置
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
    dest = "my_configs/_ft_probe_nano.yaml"
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
    sft = next(f for f in body["files"] if f["name"] == "sft_best.pt")
    assert sft["variant"] == "nano" and sft["stage"] == "sft" and sft["size_mb"] > 0


def test_task_registry(api):
    body = api.get("/api/train/tasks").json()
    builtin = {"pretrain", "sft", "dpo", "opd", "sft_lora", "grpo", "ppo"}
    assert builtin.issubset(body["tasks"].keys())
    assert body["tasks"]["probe"]["script"] == _PROBE_SCRIPT_REL  # 探针注入可见
    pre = body["tasks"]["pretrain"]
    assert pre["fields"][0]["name"] == "model" and pre["fields"][0]["required"] is True
    assert body["tasks"]["sft"]["variant_flag"] is True
    assert body["variants"] == ["nano", "lite", "pro"]
    assert body["launchers"] == ["python", "torchrun", "deepspeed"]
    assert any(e["path"] == "manual/configs/nano.yaml" for e in body["configs"])
    # dpo_data 生成任务：仅 python launcher，字段全空即可启动（模型自动探测）
    dd = body["tasks"]["dpo_data"]
    assert dd["script"] == "data_tools/dpo/run_generate.py"
    assert dd["launchers"] == ["python"] and dd["variant_flag"] is True
    assert all(not f.get("required") for f in dd["fields"])


# ── 训练生命周期 ─────────────────────────────────────────────────────
def test_train_start_validation(api):
    assert api.post("/api/train/start", json={"task": "nope"}).status_code == 400
    assert (
        api.post("/api/train/start", json={"task": "probe", "launcher": "bad"}).status_code == 400
    )
    assert api.post("/api/train/start", json={"task": "sft", "variant": "zzz"}).status_code == 400
    r = api.post("/api/train/start", json={"task": "pretrain", "fields": {}})
    assert r.status_code == 400 and "model" in r.json()["detail"]  # 必填缺失
    r = api.post(
        "/api/train/start",
        json={"task": "pretrain", "fields": {"model": "manual/configs/nope.yaml"}},
    )
    assert r.status_code == 400 and "不存在" in r.json()["detail"]


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
    assert abs(st["last_metric"]["loss"] - 1.41) < 1e-6

    runs = {x["id"]: x for x in api.get("/api/train/runs").json()}
    run = runs[run_id]
    assert run["status"] == "finished" and run["note"] == "exit=0"
    assert run["config"]["task"] == "probe"
    assert run["config"]["yaml_summary"]["lr"]["type"] == "wsd"  # nano.yaml 注入
    assert any(c.endswith("_probe.py") for c in run["config"]["cmd"])

    series = api.get("/api/train/metrics", params={"run_id": run_id}).json()["series"]
    assert series["loss"] == [[1, 1.5], [2, 1.41]]
    assert series["lr"] == [[1, 0.0001], [2, 0.000095]]
    assert series["tok_per_sec"] == [[1, 12300.0], [2, 12500.0]]
    assert series["gpu_mem"] == [[1, 1.2], [2, 1.3]]


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

        INF.server.model = None


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

/* 训练共享层（单例）：后端 /train/tasks 元数据驱动的启动表单 + SSE 日志流 +
   status/metrics 轮询，向订阅者分发。pretrain.js / posttrain.js 各自注册钩子。
   铁律 2：表单值全部留空不传 —— 面板不持有任何参数默认值，脚本/YAML 裁决。
   （例外：仅记忆用户上次显式填过的 path 字段用于下次预填 —— 用户输入而非默认值） */
"use strict";

const trainer = {
  meta: null,       // /api/train/tasks 元数据（字段清单以脚本 argparse 为准）
  run: null,        // 当前 run（/api/train/status 轮询结果）
  files: [],        // path 字段补全候选（checkpoint .pt + 配置 .yaml）
  lines: [],        // 全局日志行（cap 5000，两 tab 日志面板共用）
  seq: 0,
  _runId: null,
  _poll: null,
  _sseTimer: null,
  _alive: false,
  _ended: true,
  _sseGen: 0, // SSE 连接代际: 换代(bindRun/replay)后旧连接的帧与收尾全部失效
  _sseCtl: null, // 当前连接的 AbortController（换代时主动掐断旧连接）
  subs: { status: [], metric: [], log: [], exit: [], ended: [], ready: [] },

  on(evt, fn) {
    this.subs[evt].push(fn);
  },
  emit(evt, payload) {
    this.subs[evt].forEach((fn) => fn(payload));
  },

  /* ── 初始化：元数据 + 当前 run 探测（候选清单由 refreshCandidates 拉取）── */
  async init() {
    const [meta, st] = await Promise.all([
      api("/api/train/tasks"),
      api("/api/train/status").catch(() => ({ running: false })),
    ]);
    this.meta = meta;
    await this.refreshCandidates();
    if (st && st.run_id) {
      this.run = st;
      this.bindRun(st.run_id, st.running === true, false);
    }
    // 日志面板重绑（后端重启后 run 丢了但日志文件在: status 返回 running=false
    // 且无 run_id 时，runs 里最后一条 unfinished 的可手动点开重放 — 见各 tab）
    this.emit("ready", this.meta);
  },

  /* ── 候选清单（配置 yaml 一个池 / 模型 ckpt 一个池，按语义拆分，
       避免“配置 YAML”输入框建议里混进 checkpoints 下的模型文件）──
       每次打开启动弹窗前实时重拉：新建/删除的文件立即可见，无需刷新页面 */
  async refreshCandidates() {
    const [models, configs] = await Promise.all([
      api("/api/models").catch(() => ({ files: [] })),
      api("/api/configs").catch(() => []),
    ]);
    this.files = (models.files || []).map((f) => f.path);
    this.cfgFiles = configs.map((c) => c.path);
  },

  tasks(filter) {
    if (!this.meta) return {};
    const ks = Object.keys(this.meta.tasks);
    return filter
      ? Object.fromEntries(ks.filter((k) => filter.includes(k)).map((k) => [k, this.meta.tasks[k]]))
      : this.meta.tasks;
  },

  fileCandidates() {
    return this.files;
  },

  /* ── 启动流程：表单 modal（taskFilter 由发起 tab 决定）── */
  async openStartModal(taskFilter) {
    if (!this.meta) await this.init();
    // 版本标记：改前端后 Ctrl+F5，console 出现此行 = 已加载新版（排查旧缓存用）
    console.info("[trainer] fe v6: configs exempt + candidates refresh");
    await this.refreshCandidates(); // 候选实时刷新（见 refreshCandidates）
    const ts = this.tasks(taskFilter);
    const keys = Object.keys(ts);
    const defaultTask = keys.includes("pretrain") ? "pretrain" : keys[0];
    const box = openModal(startFormHtml(ts, this.meta, defaultTask));
    refreshTaskUi(box, defaultTask, ts[defaultTask]);
    box.querySelector("#start-task").addEventListener("change", (e) => {
      refreshTaskUi(box, e.target.value, ts[e.target.value]);
    });
    box.querySelector("#start-launcher").addEventListener("change", (e) => {
      box.querySelector("#nproc-row").style.display = e.target.value === "torchrun" ? "" : "none";
    });
    box.querySelector("#start-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      await this._submitStart(box, ts);
    });
  },

  async _submitStart(box, ts) {
    const task = box.querySelector("#start-task").value;
    const launcher = box.querySelector("#start-launcher").value;
    const nproc = parseInt(box.querySelector("#start-nproc").value, 10) || 1;
    // variant_flag 关闭的任务（pretrain 等）无变体下拉：隐藏的 select 仍返回首项，
    // 必须置空，否则 run 命名 / DB tag 误带第一个变体名（如 pretrain_nano_*）
    const variant = ts[task].variant_flag ? box.querySelector("#start-variant").value : "";
    const fields = {};
    box.querySelectorAll("[data-fname]").forEach((el) => {
      if (el.type === "checkbox") {
        if (el.checked) fields[el.dataset.fname] = "true";
      } else if (el.value.trim() !== "") {
        fields[el.dataset.fname] = el.value.trim();
      }
    });
    // 必填前置检查（元数据已声明 required）
    for (const f of ts[task].fields) {
      if (f.required && !fields[f.name]) {
        alert(`缺少必填参数: ${f.name}`);
        return;
      }
    }
    const btn = box.querySelector("#start-submit");
    btn.disabled = true;
    btn.textContent = "启动中…";
    try {
      const res = await api("/api/train/start", {
        method: "POST",
        body: JSON.stringify({ task, launcher, nproc, variant, fields, run_name: "" }),
      });
      closeModal();
      savePrefill(task, fields, ts[task]); // 记住 path 字段（下次启动带出）
      this.clearLog();
      this.bindRun(res.run_id, true);
      this.emit("status", { ...(this.run || {}), run_id: res.run_id, task, variant, status: "starting", running: true });
      this.emit("ended", { run_id: res.run_id, fresh: true });
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "启动";
      alert("启动失败：" + err.message);
    }
  },

  /* ── 绑定 run：开 SSE 日志流 + 启动轮询 ── */
  bindRun(runId, live, resetSeq = true) {
    this._runId = runId;
    this._live = !!live;
    if (resetSeq) this.seq = 0;
    this._ended = !live;
    this._sseGen++; // 换代: 在途旧连接的回调/收尾全部失效
    if (this._sseCtl) this._sseCtl.abort();
    this._alive = false;
    if (this._poll) clearInterval(this._poll);
    if (this._sseTimer) clearTimeout(this._sseTimer);
    this._poll = setInterval(() => this.poll(), 2000);
    if (live) this.connectSse();
    this.emit("metric", null); // 通知各 tab 切换数据源
    this.pollMetrics();
  },

  connectSse() {
    if (!this._runId || this._ended || this._alive) return;
    const url = `/api/train/stream?run_id=${encodeURIComponent(this._runId)}&seq=${this.seq}`;
    this._alive = true;
    const gen = ++this._sseGen; // 本连接的代际号: 换代后本连接的回调全部失效
    const ctl = new AbortController();
    this._sseCtl = ctl;
    let last = Date.now(); // 连接局部存活时间戳 — 看门狗只由本流的帧喂
    // 看门狗: 服务端每 15s 发心跳(注释帧)喂狗，45s 无任何帧即判定假死并 abort。
    // TCP 半开/中间层静默丢弃时 read() 悬住既不返回也不报错，只等 onError
    // 永远等不到 —— 日志面板停更而曲线轮询照常的根因。
    const wd = setInterval(() => {
      if (Date.now() - last > 45000) ctl.abort();
    }, 5000);
    const stale = () => gen !== this._sseGen; // 已被换代（bindRun/replay）→ 静默丢弃
    sseFetch(url, {
      signal: ctl.signal,
      onHeartbeat: () => {
        if (!stale()) last = Date.now();
      },
      onData: (ev) => {
        if (stale()) return;
        last = Date.now();
        if (ev.type === "log") {
          this.appendLine(ev.text);
          this.seq = ev.seq + 1;
        }
      },
      onExit: (ev) => {
        if (stale()) return; // 旧连接的结束帧不得污染新连接状态
        this._ended = true;
        this.emit("exit", ev);
        this.pollMetrics();
      },
      onError: () => {}, // 统一由下方 finally 兜底（重置存活态 + 调度重连）
    }).finally(() => {
      clearInterval(wd);
      if (stale()) return; // 旧代际收尾: 不碰共享状态、不调度重连
      this._sseCtl = null;
      this._alive = false;
      if (!this._ended) {
        // 断线重连: seq 续传（fetch 无 Last-Event-ID，行号由本端记录）。
        // 覆盖三类非正常结束: 网络错误 / 看门狗 abort / 服务端无 exit 帧
        // 静默关闭 —— 此前静默关闭直接 resolve 无人重置 _alive, 永不重连。
        this._sseTimer = setTimeout(() => this.connectSse(), 2000);
      }
    });
  },

  /* 状态轮询（2s）：驱动徽章/日志结束判定；顺带同步最新指标（全量 series） */
  async poll() {
    try {
      const st = await api("/api/train/status");
      this.run = st;
      const running = !!st.running;
      // 绑定失效检测（服务重启后 manager 内存态丢失是常态）:
      //  - run_id 匹配且已不跑 → 正常结束
      //  - live 绑定但后端不认识/已换成别的 run → 标记 lost，UI 走结束路径
      if (!this._ended && this._runId && this._live) {
        if (st.run_id === this._runId) {
          if (!running) {
            this._ended = true;
            this.emit("exit", { type: "exit", code: st.exit_code, status: st.status });
          }
        } else {
          this._ended = true;
          this._live = false;
          this._runId = null;
          this.emit("exit", { type: "exit", code: null, status: "lost" });
        }
      }
      this.emit("status", st);
      if (running && !this._alive && !this._ended) {
        this._ended = false;
        this.connectSse();
      }
      if (running) this.pollMetrics();
    } catch (_) {
      /* 服务未就绪 */
    }
  },

  async pollMetrics(runId) {
    const id = runId || this._runId;
    if (!id) return;
    try {
      const data = await api(`/api/train/metrics?run_id=${encodeURIComponent(id)}`);
      this.emit("metric", { run_id: data.run_id, series: data.series });
    } catch (_) {
      /* run 不存在等，忽略 */
    }
  },

  /* 历史 run 重放（无 live run 时点击 run 列表）：SSE 按 seq 回放日志 + 拉指标。
     不触碰 /api/train/status 的当前 run 判定（后端重启后 manager 已无 run）。 */
  replay(runId) {
    if (!runId || this.run && this.run.running) return;
    this._runId = runId;
    this._live = false;
    this.seq = 0;
    this._ended = false;
    this._sseGen++; // 换代: 在途旧连接（含其 exit/finally）不得污染本次回放
    if (this._sseCtl) this._sseCtl.abort();
    this._alive = false;
    this.emit("metric", null); // 各 tab 切换主曲线数据源
    this.pollMetrics(runId);
    this.connectSse();
  },

  /* ── 日志 ── */
  appendLine(text) {
    this.lines.push(text);
    if (this.lines.length > 5000) this.lines.splice(0, this.lines.length - 5000);
    this.emit("log", text);
  },

  clearLog() {
    this.lines = [];
    this.emit("log", "__clear__");
  },

  async stop() {
    try {
      const r = await api("/api/train/stop", { method: "POST", body: "{}" });
      this.emit("status", { ...(this.run || {}), running: false, status: r.status });
    } catch (err) {
      alert("停止失败：" + err.message);
    }
  },
};

/* ── 启动表单 HTML（meta 驱动；字段全部空值 → 不传 CLI）── */
function startFormHtml(tasks, meta, defaultTask) {
  const t = tasks[defaultTask] || {};
  const taskOpts = Object.entries(tasks)
    .map(([k, v]) => `<option value="${k}">${esc(v.short || v.label)}</option>`)
    .join("");
  return `<form id="start-form">
    <div class="form-grid">
      <label>任务类型</label>
      <select id="start-task">${taskOpts}</select>
      <span id="variant-field" style="display:contents">
        <label>模型变体</label>
        <select id="start-variant">
          ${meta.variants.map((v) => `<option value="${v}">${v}</option>`).join("")}
        </select>
      </span>
      <label>启动方式</label>
      <select id="start-launcher">
        <option value="python">python</option>
        <option value="torchrun">torchrun</option>
        <option value="deepspeed">deepspeed</option>
      </select>
      <label>进程数 nproc</label>
      <span id="nproc-row" style="display:none">
        <input id="start-nproc" type="number" min="1" max="8" value="2" style="width:90px" />
      </span>
    </div>
    <div class="full" style="margin-top:12px;border-top:1px solid var(--border);padding-top:10px">
      <div id="req-area"></div>
      <details id="adv-area" style="margin-top:12px">
        <summary style="font-size:12px;color:var(--dim);cursor:pointer;user-select:none">
          ☰ 高级参数
        </summary>
        <div id="opt-area" style="margin-top:10px"></div>
      </details>
    </div>
    <div class="full" style="margin-top:12px;display:flex;justify-content:flex-end;gap:10px">
      <button type="button" class="btn ghost" data-close="modal-mask">取消</button>
      <button type="submit" class="btn" id="start-submit">启动训练</button>
    </div>
  </form>
  <datalist id="dl-files">
    ${trainer.fileCandidates().map((p) => `<option value="${p}"></option>`).join("")}
  </datalist>
  <datalist id="dl-configs">
    ${(trainer.cfgFiles || []).map((p) => `<option value="${p}"></option>`).join("")}
  </datalist>`;
}

function fieldRowsHtml(task, requiredOnly) {
  if (!task) return "";
  const fs = task.fields.filter((f) => (requiredOnly ? f.required : !f.required));
  const rows = [];
  for (const f of fs) {
    const id = "f-" + f.name;
    if (f.type === "bool") {
      rows.push(`<label for="${id}">${f.label}</label>
        <input id="${id}" data-fname="${f.name}" type="checkbox" style="width:auto" />`);
      continue;
    }
    const req = f.required ? ' <span style="color:var(--err)">*必填</span>' : "";
    const help = f.help ? `<div class="help">${esc(f.help)}</div>` : "";
    if (f.type === "choice") {
      rows.push(`<label for="${id}">${f.label}</label>
        <span><select id="${id}" data-fname="${f.name}">
          <option value="">默认</option>
          ${f.choices.map((c) => `<option value="${c}">${c}</option>`).join("")}
        </select>${help}</span>`);
      continue;
    }
    const isNum = f.type === "int" || f.type === "float";
    // path 候选按 suggest 归类: configs→配置清单 / ckpt→checkpoint 文件；
    // 未标注的 path 字段（目录、数据文件等）不挂候选，不检测任何文件池
    const listId = isNum
      ? ""
      : f.suggest === "configs"
        ? "dl-configs"
        : f.suggest === "ckpt"
          ? "dl-files"
          : "";
    const listAttr = listId ? ` list="${listId}"` : "";
    const attr = isNum
      ? `type="number" step="${f.type === "float" ? "any" : "1"}"`
      : `type="text"${listAttr}`;
    rows.push(`<label for="${id}" style="font-size:12px">${esc(f.label)}${req}</label>
      <span><input id="${id}" data-fname="${f.name}" ${attr} autocomplete="off" style="font-family:var(--mono);font-size:12px" />${help}</span>`);
  }
  return rows.join("");
}

/* ── 启动 path 记忆：只记用户上次显式填过的路径字段（非面板默认值）── */
const PREFILL_KEY = "webui.startPrefill";

function readPrefill() {
  try {
    return JSON.parse(localStorage.getItem(PREFILL_KEY) || "{}");
  } catch (_) {
    return {};
  }
}

function savePrefill(taskKey, fields, task) {
  const vals = {};
  for (const f of task.fields) {
    // 配置类字段不参与记忆：每次启动前人工选当前配置，旧值只会误导
    if (f.type === "path" && f.suggest !== "configs" && fields[f.name]) vals[f.name] = fields[f.name];
  }
  try {
    const saved = readPrefill();
    saved[taskKey] = vals;
    localStorage.setItem(PREFILL_KEY, JSON.stringify(saved));
  } catch (_) {
    /* 存储不可用（隐私模式等）时静默 */
  }
}

function applyPrefill(box, taskKey, task) {
  const saved = readPrefill();
  const entry = saved[taskKey] || {};
  // 配置类字段永不记忆（见 savePrefill）。此处按字段元数据判定而非 DOM 的
  // el.list 关联（动态注入的 input 若 datalist 关联失败会漏网回填），并顺带
  // 自愈：剥离历史版本遗留的 config 记忆，让 localStorage 不再躺陈旧配置路径
  const cfgNames = new Set(
    (task.fields || [])
      .filter((f) => f.type === "path" && f.suggest === "configs")
      .map((f) => f.name),
  );
  let dirty = false;
  for (const n of cfgNames) {
    if (entry[n]) {
      delete entry[n];
      dirty = true;
    }
  }
  if (dirty) {
    if (Object.keys(entry).length) saved[taskKey] = entry;
    else delete saved[taskKey];
    try {
      localStorage.setItem(PREFILL_KEY, JSON.stringify(saved));
    } catch (_) {
      /* 存储不可用时静默 */
    }
  }
  // path 记忆只在文件仍存在时预填：模型/ckpt 被删或改名后，旧值会误导启动
  const liveFiles = new Set(trainer.fileCandidates());
  box.querySelectorAll("[data-fname]").forEach((el) => {
    if (cfgNames.has(el.dataset.fname)) return;
    const v = entry[el.dataset.fname];
    if (!v) return;
    if (el.list && !liveFiles.has(v)) return; // 记忆值已不在候选池 → 跳过预填
    el.value = v;
  });
}

function refreshTaskUi(box, taskKey, task) {
  const reqRows = fieldRowsHtml(task, true);
  box.querySelector("#req-area").innerHTML = reqRows
    ? `<div class="form-grid">${reqRows}</div>`
    : '<div class="hint" style="font-size:12px">无必填参数，可直接启动</div>';
  box.querySelector("#opt-area").innerHTML = `<div class="form-grid">${fieldRowsHtml(task, false)}</div>`;
  box.querySelector("#variant-field").style.display = task.variant_flag ? "contents" : "none";
  // 启动方式按任务白名单过滤（如 dpo_data 只支持 python，内部自行分片）；
  // 无条件重建选项：从受限任务切回普通任务时恢复全量下拉
  const ls = box.querySelector("#start-launcher");
  const allL = (trainer.meta || {}).launchers || ["python", "torchrun", "deepspeed"];
  const allowL = task.launchers && task.launchers.length ? task.launchers : allL;
  const curL = ls.value;
  ls.innerHTML = allowL.map((l) => `<option value="${l}">${l}</option>`).join("");
  ls.value = allowL.includes(curL) ? curL : allowL[0];
  ls.disabled = allowL.length === 1;
  box.querySelector("#nproc-row").style.display = "none";
  applyPrefill(box, taskKey, task);
}

/* ── 日志面板：全局行同步到所有 .log 容器 ── */
/* 日志行批量写：逐行 append + scrollTop 会强制 reflow，回放上千行时卡
   主线程。攒 200 行冲刷一次；慢速流用 rAF 合帧（每帧至多一次）。 */
function initLogSync() {
  let buf = [];
  let raf = 0;
  const flush = () => {
    raf = 0;
    if (!buf.length) return;
    const lines = buf;
    buf = [];
    const frag = document.createDocumentFragment();
    for (const t of lines) {
      const d = document.createElement("div");
      d.textContent = t;
      frag.appendChild(d);
    }
    $$(".log").forEach((el) => {
      el.appendChild(frag.cloneNode(true));
      while (el.children.length > 2000) el.removeChild(el.firstChild);
      el.scrollTop = el.scrollHeight; // 每批只强制一次滚动定位
    });
  };
  trainer.on("log", (text) => {
    if (text === "__clear__") {
      buf = [];
      if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
      $$(".log").forEach((el) => (el.textContent = ""));
      return;
    }
    buf.push(text);
    if (buf.length >= 200) {
      // 大批量（回放/瞬间涌入）不等帧，立即按批冲刷
      if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
      flush();
      return;
    }
    if (!raf) raf = requestAnimationFrame(flush);
  });
  // 历史 run 重放点击 → 直接在此显示？由各 tab 的 run 列表绑定。
}

/* ── 停止按钮与 header 训练徽章 ── */
// 注: 本函数由 boot 的 DOMContentLoaded 回调调用; 旧实现内部再嵌套一层
// DOMContentLoaded 监听, 在事件派发中注册永不触发 → 徽章/停止按钮失效。
function initTrainHeader() {
  $$("[data-stop-btn]").forEach((b) => {
    b.addEventListener("click", () => trainer.stop());
  });
  trainer.on("status", (st) => {
    $$("[data-stop-btn]").forEach((b) => (b.disabled = !st.running));
    $$("[data-start-btn]").forEach((b) => (b.disabled = !!st.running)); // 运行中禁用启动（变暗）
    const badge = $("#train-text");
    const dot = $("#train-dot");
    if (!badge) return;
    const lm = st.last_metric || {};
    if (st.status === "stopping") {
      badge.textContent = "停止中…";
      dot.className = "dot run";
    } else if (st.running) {
      badge.textContent = `${st.task} ${lm.step != null ? "· " + lm.step + " step" : ""}`;
      dot.className = "dot run";
    } else if (st.status === "finished") {
      badge.textContent = `完成 ${st.task} ${lm.loss != null ? "· loss " + fmtNum(lm.loss) : ""}`;
      dot.className = "dot idle";
    } else if (st.status === "failed") {
      badge.textContent = `失败 ${st.task} (exit=${st.exit_code})`;
      dot.className = "dot err";
    } else {
      badge.textContent = "空闲";
      dot.className = "dot idle";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initLogSync();
  initTrainHeader();
  trainer.init();
});

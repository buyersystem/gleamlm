/* 训练共享层（单例）：后端 /train/tasks 元数据驱动的启动表单 + SSE 日志流 +
   status/metrics 轮询，向订阅者分发。pretrain.js / posttrain.js 各自注册钩子。
   铁律 2 演进：留空不传的语义保留，但「留空会用什么」由 /train/defaults 显式
   旁显（模型/数据/保存目录），模型类字段升级为按变体×阶段的下拉 ——
   填写↔执行严格对应：显式值原样执行、隐性值显式化、错配需确认。 */
"use strict";

const trainer = {
  meta: null,       // /api/train/tasks 元数据（字段清单以脚本 argparse 为准）
  run: null,        // 当前 run（/api/train/status 轮询结果）
  files: [],        // checkpoint .pt 候选（ckpt 字段补全 + 记忆存活校验）
  models: [],       // checkpoint 明细（name/path/variant/stage/size_mb/mtime）
  cfgs: [],         // 配置清单明细（path+name，供「配置模板」下拉）
  _defGen: 0,       // /train/defaults 请求代际号（防快速切变体的竞态）
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
      if (st.running === true) {
        this.bindRun(st.run_id, true, false);
      } else {
        this.replay(st.run_id); // 刚结束的 run：自动回放日志（只读），刷新即见
      }
    }
    // 日志面板重绑（后端重启后 run 丢了但日志文件在: status 返回 running=false
    // 且无 run_id 时，runs 里最后一条 unfinished 的可手动点开重放 — 见各 tab）
    this.emit("ready", this.meta);
  },

  /* ── 候选清单（配置 yaml 一个池 / 模型 ckpt 一个池，按语义拆分，
       避免“配置 YAML”输入框建议里混进 checkpoints 下的模型文件）──
       每次打开启动弹窗前实时重拉：新建/删除的文件立即可见，无需刷新页面 */
  /* 片一：改用 allSettled —— 原先两个 .catch() 无法区分「哪个失败」，
     失败时会静默变成空候选（用户以为没有配置文件）。
     策略：失败**不清空**旧候选（stale-but-usable，输入框仍可用旧值补全），
     只提示一句。弹窗里没有位置放整屏失败态，所以这里用 toast。 */
  async refreshCandidates() {
    const [mRes, cRes] = await Promise.allSettled([api("/api/models"), api("/api/configs")]);
    const okM = mRes.status === "fulfilled";
    const okC = cRes.status === "fulfilled";
    if (okM) {
      this.models = mRes.value.files || []; // 明细供 ckpt 下拉（变体×阶段分组）
      this.files = this.models.map((f) => f.path);
    }
    if (okC) this.cfgs = cRes.value; // 配置清单明细（path+name；「配置模板」下拉选项）
    if (okM && okC) return;
    const what = !okM && !okC
      ? "模型与配置文件列表"
      : okM ? "配置文件列表" : "模型文件列表";
    toast(`无法读取${what} —— 输入框仍可手填路径`, "err", 6000);
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
    console.info("[trainer] fe v14: grpo/ppo variant select (model derived like DPO)");
    prunePrefill(); // 记忆机制退役：仅 pretrain 保留 path 记忆，清历史遗留
    await this.refreshCandidates(); // 候选实时刷新（见 refreshCandidates）
    const ts = this.tasks(taskFilter);
    const keys = Object.keys(ts);
    const single = keys.length === 1; // 卡片入口：标题锁定任务名、隐藏任务下拉
    const defaultTask = keys.includes("pretrain") ? "pretrain" : keys[0];
    const box = openModal(startFormHtml(ts, this.meta, defaultTask, single));
    // 单任务入口 #start-task 隐藏（display:none 的焦点为 no-op）→ 落焦首个可见控件
    // （启动方式/进程数已收进高级参数：pretrain 入口无可见下拉时回落首个核心字段）
    requestAnimationFrame(() => {
      const cand = [
        box.querySelector("#start-variant"),
        box.querySelector("#start-launcher"),
        box.querySelector("#core-area [data-fname]"),
      ];
      const el = cand.find((x) => x && x.offsetParent !== null);
      if (el) el.focus();
    });
    await refreshTaskUi(box, defaultTask, ts[defaultTask]);
    box.querySelector("#start-task").addEventListener("change", async (e) => {
      await refreshTaskUi(box, e.target.value, ts[e.target.value]);
    });
    box.querySelector("#start-variant").addEventListener("change", () => {
      const k = box.querySelector("#start-task").value;
      applyContext(box, k, ts[k]);
    });
    // 字段区随任务重建 → 用事件委托承接 ckpt 下拉/自定义输入的变化
    box.addEventListener("change", (e) => {
      const k = box.querySelector("#start-task").value;
      if (e.target.tagName === "SELECT" && e.target.dataset.ckpt) {
        syncCkptCustom(box, e.target.dataset.ckpt);
        updateMismatch(box, k, ts[k]);
      } else if (e.target.tagName === "SELECT" && e.target.dataset.configs) {
        // 预训练「配置模板」切换 → 重推导（数据/保存目录预填 + 续训模型候选）
        applyContext(box, k, ts[k]);
      }
    });
    box.addEventListener("input", (e) => {
      if (e.target.classList && e.target.classList.contains("ckpt-custom")) {
        const k = box.querySelector("#start-task").value;
        updateMismatch(box, k, ts[k]);
      }
    });
    box.querySelector("#start-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      await this._submitStart(box, ts);
    });
  },

  /* ── 留空回落值（/api/train/defaults）：代际号保证最近一次上下文胜出 ──
     返回整个响应体（fields + checkpoint_dir）—— 变体=配置模板，候选归属
     按模板 checkpoint_dir 前缀过滤 */
  async loadDefaults(task, variant) {
    const gen = ++this._defGen;
    try {
      const res = await api(
        `/api/train/defaults?task=${encodeURIComponent(task)}&variant=${encodeURIComponent(variant)}`,
      );
      return gen === this._defGen ? res || {} : null; // 过期响应丢弃
    } catch (_) {
      return gen === this._defGen ? {} : null; // 端点不可用：静默降级（不预填不旁显）
    }
  },

  async _submitStart(box, ts) {
    const taskKey = box.querySelector("#start-task").value;
    const task = ts[taskKey];
    const launcher = box.querySelector("#start-launcher").value;
    const nproc = parseInt(box.querySelector("#start-nproc").value, 10) || 1;
    // variant_flag 关闭的任务（pretrain / grpo / ppo）无变体语义：置空，
    // 否则 run 命名 / DB tag 误带下拉里的第一个变体名（如 pretrain_nano_*）
    const variant = task.variant_flag ? box.querySelector("#start-variant").value : "";
    const fields = {};
    box.querySelectorAll("[data-fname]").forEach((el) => {
      if (el.type === "checkbox") {
        if (el.checked) fields[el.dataset.fname] = "true";
      } else if (el.tagName === "SELECT" && el.dataset.ckpt && el.value === "__custom__") {
        /* 自定义路径占位项不是值：实际值由 -custom 输入框提供（非空才收集） */
      } else if (el.value.trim() !== "") {
        fields[el.dataset.fname] = el.value.trim();
      }
    });
    // 必填前置检查（元数据已声明 required；推导字段的值来自下拉默认项，不参与缺失检查）
    for (const f of task.fields) {
      if (f.required && !fields[f.name] && !isFixedField(task, f)) {
        // E6：字段级反馈取代 alert —— 弹窗会给不出「是哪个字段」
        showFieldError(box, f.name, `缺少必填参数：${f.label || f.name}`);
        return;
      }
    }
    // 错配显式确认：跨变体/跨阶段合法，但必须先让用户看见并确认
    const mm = updateMismatch(box, taskKey, task);
    if (mm) {
      const ok = await confirmOverlay({
        title: "确认按该模型启动？",
        okText: "仍要启动",
        body: `<p style="margin:0 0 6px">${esc(mm)}</p>
          <p style="margin:0;color:var(--dim);font-size:13px">面板严格按你显式填写的路径启动，不做任何替换。</p>`,
      });
      if (!ok) return;
    }
    const btn = box.querySelector("#start-submit");
    btn.disabled = true;
    btn.textContent = "启动中…";
    try {
      const res = await api("/api/train/start", {
        method: "POST",
        body: JSON.stringify({ task: taskKey, launcher, nproc, variant, fields, run_name: "" }),
      });
      closeModal();
      // path 记忆仅保留 pretrain（后训练字段由 defaults 预填/旁显接管）
      if (taskKey === "pretrain") savePrefill(taskKey, fields, task);
      this.clearLog();
      this.bindRun(res.run_id, true);
      this.emit("status", { ...(this.run || {}), run_id: res.run_id, task: taskKey, variant, status: "starting", running: true });
      this.emit("ended", { run_id: res.run_id, fresh: true });
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "启动训练";
      toast("启动失败：" + err.message, "err");
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
    // F4：后台标签页停止轮询（切回前台由 visibilitychange 立即补一次）
    this._poll = setInterval(() => {
      if (!document.hidden) this.poll();
    }, 2000);
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
      notePoll(true); // F3：在线（清除顶部失联提示）
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
      notePoll(false); // F3：失联计数（3 次后顶部提示）
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
    this.clearLog(); // 重放前清空日志面板：跨 run 点击不堆叠、重复回放不翻倍
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

  /* E7：停止是不可逆的破坏性操作，必须先确认。
     原实现是单击即停 —— 而"删除一条已完成 run"反而有 confirm，
     保护强度与破坏性倒挂。这里顺带把当前进度显示出来，好判断值不值得停。
     （选样式化确认而非"按住 1 秒"：能展示 step/checkpoint 上下文，信息量更大。）*/
  async stop() {
    const st = this.run || {};
    const lm = st.last_metric || {};
    const stepLine =
      lm.step != null
        ? `step <b>${esc(lm.step)}</b>${st.total_steps ? ` / ${esc(st.total_steps)}` : ""}`
        : "尚未产生 step";
    const ok = await confirmDialog({
      title: "停止训练？",
      okText: "停止训练",
      body: `<p style="margin:0">当前任务　<b>${esc(st.task || "-")}</b>${esc(stepLine)}</p>
        <p style="margin:0;color:var(--dim);font-size:13px">训练不会继续；已落盘的 checkpoint 保留，
        未保存的优化器状态会丢失。</p>`,
    });
    if (!ok) return;
    try {
      const r = await api("/api/train/stop", { method: "POST", body: "{}" });
      this.emit("status", { ...(this.run || {}), running: false, status: r.status });
      toast("已请求停止，等待进程退出…");
    } catch (err) {
      toast("停止失败：" + err.message, "err");
    }
  },
};

/* ── 启动表单 HTML（meta 驱动；字段留空不传 CLI，留空回落值旁显）── */
function startFormHtml(tasks, meta, defaultTask, single) {
  const t = tasks[defaultTask] || {};
  const taskOpts = Object.entries(tasks)
    .map(([k, v]) => `<option value="${k}">${esc(v.short || v.label)}</option>`)
    .join("");
  // 单任务入口：标题直接锁定任务短名（SFT / DPO / …），任务下拉隐藏
  const title = single ? esc(t.short || t.label || defaultTask) : "启动训练";
  return `<form id="start-form">
    <div class="m-head"><b>${title}</b><span class="spacer"></span>
      <button type="button" class="btn ghost sm" data-close="modal-mask">✕</button></div>
    <div class="m-body">
      <div class="form-grid">
        <span id="task-field" style="display:${single ? "none" : "contents"}">
          <label>任务类型</label>
          <select id="start-task">${taskOpts}</select>
        </span>
        <span id="variant-field" style="display:contents">
          <label>配置模板</label>
          <select id="start-variant">
            ${/* 显示完整文件名（nano.yaml）；value 仍为模板名（API/CLI 契约） */ ""}
            ${meta.variants.map((v) => `<option value="${v}">${v}.yaml</option>`).join("")}
          </select>
        </span>
      </div>
      <div id="mismatch-bar" class="warnbar" style="display:none;margin-top:12px"></div>
      <div class="full" style="margin-top:12px;border-top:1px solid var(--border);padding-top:8px">
        <div id="core-area"></div>
        <details id="adv-area" style="margin-top:12px">
          <summary style="font-size:12px;color:var(--dim);cursor:pointer;user-select:none">☰ 高级参数</summary>
          <div class="form-grid" style="margin-top:8px">
            <label for="start-launcher">启动方式</label>
            <span><select id="start-launcher">
              <option value="python">python</option>
              <option value="torchrun">torchrun</option>
              <option value="deepspeed">deepspeed</option>
            </select></span>
            <label for="start-nproc">进程数 nproc</label>
            <span><input id="start-nproc" type="number" min="1" max="8" value="2" style="width:90px" />
              <div class="help">仅 torchrun 启动方式生效（python / deepspeed 忽略）</div></span>
          </div>
          <div id="opt-area" style="margin-top:8px"></div>
        </details>
      </div>
    </div>
    <div class="m-foot">
      <button type="button" class="btn ghost" data-close="modal-mask">取消</button>
      <button type="submit" class="btn" id="start-submit">启动训练</button>
    </div>
  </form>
  <datalist id="dl-files">
    ${trainer.fileCandidates().map((p) => `<option value="${p}"></option>`).join("")}
  </datalist>`;
}

/* ── 推导字段：变体=配置模板 → 目录随所选模板推导，目录内同类条目全部可选 ──
   选定配置文件名后，后端 defaults 下发「固定目录 + 目录内候选」（学生/基座模型、
   数据、教师按模板落点推导）——前端渲染为下拉：目录锁定，目录内条目全部列出可选，
   默认选中模板推导目标；提交收集所选项（脚本侧为既有覆写旗标，CLI 语义零变化），
   未推导出时保持「空值不传 → YAML 单轨裁决」原状。 */
const FIXED_NAMES = new Set(["model", "model_path", "data_path", "data", "teacher_model_path"]);

function isFixedField(task, f) {
  return !!task.variant_flag && FIXED_NAMES.has(f.name);
}

/* ── 单字段渲染（core / adv 两区共用）──
   ctx: {ckpt: 是否渲染 checkpoint 下拉（任务含非推导 ckpt 类字段：pretrain 续训模型）;
         fixed: 推导字段名集合（目录固定 + 目录内条目可选）} */
function fieldRowHtml(f, ctx) {
  const id = "f-" + f.name;
  const locked = !!(ctx.fixed && ctx.fixed.has(f.name));
  if (f.type === "bool") {
    return `<label for="${id}">${f.label}</label>
      <input id="${id}" data-fname="${f.name}" type="checkbox" style="width:auto" />`;
  }
  const req = f.required ? ' <span class="req-star">*必填</span>' : "";
  const help = f.help ? `<div class="help">${esc(f.help)}</div>` : "";
  if (f.type === "choice") {
    return `<label for="${id}">${f.label}</label>
      <span><select id="${id}" data-fname="${f.name}">
        <option value="">默认</option>
        ${f.choices.map((c) => `<option value="${c}">${c}</option>`).join("")}
      </select>${help}</span>`;
  }
  const isNum = f.type === "int" || f.type === "float";
  if (!isNum && f.type === "path" && locked) {
    // 推导字段：候选 = 模板推导目录内全部同类条目（defaults 下发 cands），
    // 全部列出可选；提交收集所选项 → 脚本既有覆写旗标承接；
    // 必有默认值 → 不标必填（空值仅出现在推导失败时，后端必填检查兜底）
    return `<label for="${id}" style="font-size:12px">${esc(f.label)}</label>
      <span>
        <select id="${id}" data-fname="${f.name}" data-derived="1" style="font-family:var(--mono);font-size:12px"></select>
        ${help}
      </span>`;
  }
  if (!isNum && f.type === "path" && f.suggest === "configs") {
    // 配置模板：候选 = 配置清单（选项只显文件名，value 仍为路径 → CLI --model 契约不变）。
    // 不标必填红星（未选时由必填检查兜底反馈）
    const opts = (trainer.cfgs || [])
      .map((c) => `<option value="${esc(c.path)}">${esc(c.name)}</option>`)
      .join("");
    return `<label for="${id}">${esc(f.label)}</label>
      <span><select id="${id}" data-fname="${f.name}" data-configs="1">
        <option value="">（请选择）</option>
        ${opts}
      </select>${help}</span>`;
  }
  if (!isNum && f.type === "path" && f.suggest === "ckpt" && ctx.ckpt) {
    // ckpt 下拉：空项 = 不传 CLI（脚本回落）；选项只显文件名（归属由候选过滤
    // 或组标签承担）；末尾「自定义路径…」暴露 -custom 输入框（手填任意路径）；
    // 不标必填红星（空选由必填检查兜底反馈）
    return `<label for="${id}">${esc(f.label)}</label>
      <span>
        <select id="${id}" data-fname="${f.name}" data-ckpt="${f.name}"></select>
        <input id="${id}-custom" data-fname="${f.name}" class="ckpt-custom" type="text"
          autocomplete="off" placeholder="输入自定义路径…" style="display:none" />
        ${help}
        <div class="fnote" data-fnote="${f.name}" style="display:none"></div>
      </span>`;
  }
  // path 候选按 suggest 归类: ckpt→checkpoint datalist（configs 由「配置模板」
  // 下拉分支接管）；未标注的 path 字段（目录、数据文件等）不挂候选
  const listId = isNum ? "" : f.suggest === "ckpt" ? "dl-files" : "";
  const listAttr = listId ? ` list="${listId}"` : "";
  const attr = isNum
    ? `type="number" step="${f.type === "float" ? "any" : "1"}"`
    : `type="text"${listAttr}`;
  // 普通 path 字段也挂 fnote 容器（数据/教师等由 defaults 旁显回落值）
  const note =
    !isNum && f.type === "path"
      ? `<div class="fnote" data-fnote="${f.name}" style="display:none"></div>`
      : "";
  return `<label for="${id}" style="font-size:12px">${esc(f.label)}${req}</label>
    <span><input id="${id}" data-fname="${f.name}" ${attr} autocomplete="off" style="font-family:var(--mono);font-size:12px" />${help}${note}</span>`;
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

/* 字段分区：核心（模型/数据/保存/续训）常驻可见，其余收进高级参数 */
const CORE_ORDER = {
  model: 0,
  model_path: 0,
  teacher_model_path: 1,
  data: 2,
  data_path: 2,
  save_dir: 3,
  output_dir: 3,
  resume: 4,
};

function splitFields(task) {
  const core = [];
  const adv = [];
  for (const f of task.fields || []) {
    (CORE_ORDER[f.name] !== undefined ? core : adv).push(f);
  }
  core.sort((a, b) => CORE_ORDER[a.name] - CORE_ORDER[b.name]);
  return { core, adv };
}

async function refreshTaskUi(box, taskKey, task) {
  const { core, adv } = splitFields(task);
  const fixed = new Set((task.fields || []).filter((f) => isFixedField(task, f)).map((f) => f.name));
  // ckpt 下拉：任务存在「非推导的 ckpt 类字段」（suggest=ckpt 注入）即渲染，
  // 现仅 pretrain 续训模型（变体任务的模型字段走推导下拉）
  const ckpt = (task.fields || []).some((f) => f.suggest === "ckpt" && !isFixedField(task, f));
  const ctx = { ckpt, fixed };
  box.querySelector("#core-area").innerHTML = core.length
    ? `<div class="form-grid">${core.map((f) => fieldRowHtml(f, ctx)).join("")}</div>`
    : '<div class="hint" style="font-size:12px">无可配置字段</div>';
  box.querySelector("#opt-area").innerHTML = adv.length
    ? `<div class="form-grid">${adv.map((f) => fieldRowHtml(f, ctx)).join("")}</div>`
    : "";
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
  // path 记忆仅 pretrain（其余任务的 path 语义由 defaults 预填/旁显接管）
  if (taskKey === "pretrain") applyPrefill(box, taskKey, task);
  await applyContext(box, taskKey, task);
}

/* ── 上下文联动（任务/变体切换）：defaults 预填 + 旁显 + ckpt 下拉 + 错配 ── */
async function applyContext(box, taskKey, task) {
  if (!task) return;
  let variant = "";
  if (task.variant_flag) {
    const variantEl = box.querySelector("#start-variant");
    variant = variantEl ? variantEl.value : "";
  } else if (taskKey === "pretrain") {
    // 预训练：配置模板 = 表单「配置模板」下拉（value 为路径 → 模板名供 defaults 解析）
    const cfgEl = box.querySelector('select[data-configs="1"]');
    variant = cfgEl && cfgEl.value ? cfgEl.value.split("/").pop().replace(/\.ya?ml$/i, "") : "";
  }
  const res = await trainer.loadDefaults(taskKey, variant);
  if (res === null) return; // 过期响应：已有更新的上下文在途
  const defs = res.fields || {};
  // 配置模板：ckpt 候选归属 = 模板 checkpoint_dir 前缀（空 → 回退变体名约定）
  box.dataset.ckPrefix = res.checkpoint_dir || "";
  updateSaveDirs(box, task, defs);
  // 预训练：数据目录同随模板推导（YAML data_dir 前缀），与保存目录同为预填可改
  if (taskKey === "pretrain" && defs.data && defs.data.path) {
    const dEl = box.querySelector('[data-fname="data"]');
    if (dEl) dEl.value = defs.data.path;
  }
  updateDerivedFields(box, task, defs);
  updateNotes(box, task, defs);
  fillCkptOptions(box, task, variant);
  updateMismatch(box, taskKey, task);
}

/* 保存目录预填：显式展示「将会保存到哪」；上下文变化即覆盖（保证对应性） */
function updateSaveDirs(box, task, defs) {
  for (const f of task.fields || []) {
    if (f.name !== "save_dir" && f.name !== "output_dir") continue;
    const el = box.querySelector(`[data-fname="${f.name}"]`);
    const d = defs[f.name];
    if (el && d && d.path) el.value = d.path;
  }
}

/* 推导字段填充：下拉 = 模板推导目录内全部同类条目，
   默认选中模板推导目标（不在候选内则保持首个）。 */
function updateDerivedFields(box, task, defs) {
  for (const f of task.fields || []) {
    if (!isFixedField(task, f)) continue;
    const sel = box.querySelector(`select[data-fname="${f.name}"]`);
    if (!sel) continue;
    const d = defs[f.name] || {};
    const cands = d.cands || [];
    let html = "";
    if (cands.length) {
      html = cands.map((c) => `<option value="${esc(c.path)}">${esc(c.name)}</option>`).join("");
    } else if (d.path) {
      html = `<option value="${esc(d.path)}">${esc(d.path.split("/").pop())}</option>`;
    } else {
      html = '<option value="">（未推导出可选文件）</option>';
    }
    sel.innerHTML = html;
    if (d.path && [...sel.options].some((o) => o.value === d.path)) sel.value = d.path;
  }
}

/* 旁显（模型/数据/教师等留空回落值；已直接预填的目录不重复旁显）。
   推导字段的值在下拉内，不旁显。 */
function updateNotes(box, task, defs) {
  for (const f of task.fields || []) {
    const note = box.querySelector(`[data-fnote="${f.name}"]`);
    if (!note) continue;
    const d = defs[f.name];
    if (!d || !d.path || f.name === "save_dir" || f.name === "output_dir" || f.name === "data") {
      note.style.display = "none";
      note.textContent = "";
      continue;
    }
    const mark =
      d.exists === true
        ? ' <span class="ok">✓ 存在</span>'
        : d.exists === false
          ? ' <span class="err">✗ 不存在</span>'
          : "";
    note.innerHTML = `留空 → 自动回落：<code>${esc(d.path)}</code>${mark}`;
    note.style.display = "";
  }
}

/* ckpt 下拉填充：候选按归属过滤（模板 ckpt 前缀优先，缺省按变体名），按阶段
   分组（上游组优先、组内按时间倒序）；旧值不在新列表时插为保留项 */
function fillCkptOptions(box, task, variant) {
  for (const f of task.fields || []) {
    const sel = box.querySelector(`select[data-ckpt="${f.name}"]`);
    if (!sel) continue;
    const emptyLabel =
      f.name === "resume" ? "（不续训）" : f.required ? "（请选择）" : "（默认）";
    const cur = sel.value;
    // 归属过滤：优先模板 checkpoint_dir 前缀，defaults 未取到时按变体名兜底
    const ckPrefix = box.dataset.ckPrefix || "";
    const items = (trainer.models || []).filter((m) =>
      ckPrefix ? m.path.startsWith(ckPrefix + "/") : m.variant === variant,
    );
    const up = task.upstream_stage || "";
    const stages = [...new Set(items.map((m) => m.stage))];
    stages.sort((a, b) => (a === up ? -1 : b === up ? 1 : String(a).localeCompare(String(b))));
    let html = `<option value="">${emptyLabel}</option>`;
    for (const st of stages) {
      const grp = items
        .filter((m) => m.stage === st)
        .sort((a, b) => String(b.mtime).localeCompare(String(a.mtime)));
      // "· 上游" 仅任务确有上游阶段(up 非空)且命中该组时标注——
      // sft/lora 的 upstream_stage 为空(聚焦变体根), 不得误标
      html += `<optgroup label="${esc((st || "根目录") + (up !== "" && st === up ? " · 上游" : ""))}">`;
      html += grp
        .map((m) => `<option value="${esc(m.path)}">${esc(m.name)}</option>`)
        .join("");
      html += "</optgroup>";
    }
    html += '<option value="__custom__">自定义路径…</option>';
    sel.innerHTML = html;
    if (cur && cur !== "__custom__") {
      if (![...sel.options].some((o) => o.value === cur)) {
        const o = document.createElement("option");
        o.value = cur;
        o.textContent = cur.split("/").pop();
        sel.insertBefore(o, sel.lastElementChild);
      }
      sel.value = cur;
    } else {
      sel.value = cur === "__custom__" ? "__custom__" : "";
    }
    syncCkptCustom(box, f.name);
  }
}

function syncCkptCustom(box, fname) {
  const sel = box.querySelector(`select[data-ckpt="${fname}"]`);
  const inp = box.querySelector(`#f-${fname}-custom`);
  if (!sel || !inp) return;
  inp.style.display = sel.value === "__custom__" ? "" : "none";
}

/* 错配检测：模型显式值不属于「当前变体的上游阶段」→ 黄条 + 启动前确认。
   自定义外部路径（非 checkpoints/ 前缀）不判定（外部模型无从对应）。
   返回提示文本或 null（_submitStart 用返回值决定是否弹确认）。 */
function updateMismatch(box, taskKey, task) {
  const bar = box.querySelector("#mismatch-bar");
  if (!bar || !task) return null;
  let text = null;
  const mf = (task.fields || []).find((f) => f.name === "model" || f.name === "model_path");
  if (mf && task.variant_flag) {
    const sel = box.querySelector(`select[data-ckpt="${mf.name}"]`);
    const inp = box.querySelector(`#f-${mf.name}-custom`);
    let val = "";
    if (sel) val = sel.value === "__custom__" ? (inp ? inp.value.trim() : "") : sel.value;
    const variant = box.querySelector("#start-variant").value;
    // 变体=配置模板：归属基准 = 模板 checkpoint_dir（defaults 下发）；
    // 未取到时回退 checkpoints/<variant>/ 约定
    const prefix = box.dataset.ckPrefix || `checkpoints/${variant}`;
    const up = task.upstream_stage || "";
    const vp = `${prefix}/`;
    if (val && val.startsWith("checkpoints/")) {
      if (!val.startsWith(vp)) {
        text = `所选模型不在变体 ${variant} 的目录（${prefix}）下：${val}`;
      } else if (up && !val.startsWith(vp + up + "/")) {
        text = `所选模型不在 ${variant} 的 ${up} 上游目录下：${val}`;
      }
    }
  }
  bar.textContent = text || "";
  bar.style.display = text ? "" : "none";
  return text;
}

/* ── 启动表单内的二段确认浮层 ──
   不能复用 confirmDialog：它复用 #modal-box，会把启动表单整个冲掉。
   独立浮层叠在 .modal-mask 之上（#confirm-overlay 提升 z-index）。 */
function confirmOverlay({ title = "确认", body = "", okText = "确认" } = {}) {
  return new Promise((resolve) => {
    const mask = document.createElement("div");
    mask.className = "modal-mask show";
    mask.id = "confirm-overlay";
    mask.innerHTML = `<div class="modal" style="width:min(440px,90vw)">
      <div class="m-head"><b>${esc(title)}</b><span class="spacer"></span>
        <button type="button" class="btn ghost sm" data-x>✕</button></div>
      <div class="m-body">${body}</div>
      <div class="m-foot">
        <button type="button" class="btn ghost" data-no>取消</button>
        <button type="button" class="btn" data-yes>${esc(okText)}</button>
      </div>
    </div>`;
    document.body.appendChild(mask);
    let settled = false;
    const done = (v) => {
      if (settled) return;
      settled = true;
      document.removeEventListener("keydown", onKey, true);
      mask.remove();
      resolve(v);
    };
    const onKey = (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        e.preventDefault();
        done(false);
      } else if (e.key === "Tab") {
        // 捕获阶段全量拦断：否则非边界的 Tab 会漏到 util.js 的焦点陷阱
        // （openDialogRoot 认的是固定 modal-mask，会把焦点拽回启动表单）
        e.stopPropagation();
        const btns = [...mask.querySelectorAll("button")];
        if (!btns.length) return;
        const first = btns[0];
        const last = btns[btns.length - 1];
        const cur = document.activeElement;
        if (!mask.contains(cur)) {
          e.preventDefault();
          first.focus();
        } else if (e.shiftKey && cur === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && cur === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey, true);
    mask.querySelectorAll("[data-x]").forEach((b) => b.addEventListener("click", () => done(false)));
    mask.querySelector("[data-no]").addEventListener("click", () => done(false));
    const yes = mask.querySelector("[data-yes]");
    yes.addEventListener("click", () => done(true));
    yes.focus();
  });
}

/* 记忆机制退役：path 记忆仅服务 pretrain，打开面板时清掉 localStorage 遗留 */
function prunePrefill() {
  try {
    const saved = readPrefill();
    let changed = false;
    for (const k of Object.keys(saved)) {
      if (k !== "pretrain") {
        delete saved[k];
        changed = true;
      }
    }
    if (changed) localStorage.setItem(PREFILL_KEY, JSON.stringify(saved));
  } catch (_) {
    /* 存储不可用时静默 */
  }
}

/* ── 日志面板：工具条（C1~C4/C7）+ 全局行同步到所有 .log 容器 ── */
/* 关键约定：**分类只发生在屏幕**。trainer.lines 始终保留全量原文，
   复制/下载给出的是完整日志 —— 否则排障时会缺掉被隐藏的那一半信息。
   日志行批量写：逐行 append + scrollTop 会强制 reflow，回放上千行时卡
   主线程。攒 200 行冲刷一次；慢速流用 rAF 合帧（每帧至多一次）。 */

const MAX_LOG_NODES = 2000;

/* 行分类 */
function logKind(text) {
  if (text.startsWith("@@GLEAM_METRIC")) return "machine"; // 哨兵行（metrics.py 的 SENTINEL）
  if (/\d+\/\d+ \[/.test(text)) return "pbar"; // tqdm 帧，与后端 _STEP_RE 同源形状
  return "text";
}

/* 级别启发式（C3）：formatter 是 "%(message)s"，日志文本里**没有级别字段**，
   所以只能按关键词判。排障要找的就是这些词。
   ⚠️ 不要用 \b(error|warn)\b 这种"词边界"写法 —— Python 的异常/警告类名全是
   `ValueError` / `RuntimeError` / `UserWarning` / `FutureWarning` 形状，
   词边界在拼接处**不存在**，那样写会把最常见的两类行整片漏掉（单测已锁）。
   代价是 "0 errors" 这类良性行也会被判为错误 —— 对"吸引注意力"的用途，宁可多报。 */
const RE_LOG_ERR =
  /error|traceback|exception|failed|failure|fatal|critical|oom|assertion|\bnan\b|\binf\b|out of memory/i;
const RE_LOG_WARN = /warn|deprecat|skipping|skipped|retrying/i;

function initLogSync() {
  const state = { follow: true, pbar: false, machine: false, level: "all", query: "" };
  let newCount = 0;

  const BAR_HTML = `
    <button class="lg-toggle" type="button" data-lg="follow" aria-pressed="true"
      title="自动滚到最新；手动向上滚动会暂停，滚回底部自动恢复">
      <span class="lg-dot"></span>跟随</button>
    <button class="lg-toggle" type="button" data-lg="pbar" aria-pressed="false"
      title="tqdm 逐帧进度行（默认折叠为一行原地刷新）">进度帧</button>
    <button class="lg-toggle" type="button" data-lg="machine" aria-pressed="false"
      title="结构化机器行 @@GLEAM_METRIC（默认隐藏；指标条已在消费这些字段）">机器行</button>
    <select class="lg-sel" data-lg="level"
      title="按级别过滤 —— 日志文本不含级别字段，此处为关键词启发式">
      <option value="all">全部</option>
      <option value="warn">WARN+</option>
      <option value="err">ERROR+</option>
    </select>
    <input class="lg-search" type="search" data-lg="query" placeholder="搜索…" />
    <span class="log-hit" data-lg="hit"></span>
    <span class="spacer"></span>
    <button class="btn ghost sm" type="button" data-lg="copy" title="复制屏幕上当前可见的日志">复制</button>
    <button class="btn ghost sm" type="button" data-lg="download"
      title="下载完整日志文件（含进度帧与机器行）">下载</button>`;

  const logs = () => $$(".log");

  function syncToggles() {
    $$("[data-lg=follow]").forEach((b) =>
      b.setAttribute("aria-pressed", String(state.follow))
    );
    $$("[data-lg=pbar]").forEach((b) => b.setAttribute("aria-pressed", String(state.pbar)));
    $$("[data-lg=machine]").forEach((b) =>
      b.setAttribute("aria-pressed", String(state.machine))
    );
    $$("[data-lg=level]").forEach((s) => (s.value = state.level));
  }

  function applyClasses() {
    logs().forEach((el) => {
      el.classList.toggle("show-pbar", state.pbar);
      el.classList.toggle("show-machine", state.machine);
      el.classList.toggle("lvl-warn", state.level === "warn");
      el.classList.toggle("lvl-err", state.level === "err");
    });
  }

  function applySearch() {
    const q = state.query.trim().toLowerCase();
    let shown = 0;
    logs().forEach((el, ci) => {
      for (const ln of el.children) {
        const hide = q !== "" && !ln.textContent.toLowerCase().includes(q);
        ln.classList.toggle("f-hidden", hide);
        if (ci === 0 && !hide) shown++;
      }
    });
    const txt = q ? `命中 ${shown} 行` : "";
    $$("[data-lg=hit]").forEach((e) => (e.textContent = txt));
  }

  function updateJump() {
    const show = !state.follow;
    $$("[data-log-jump]").forEach((b) => {
      b.hidden = !show;
      b.textContent = show ? `↓ 已暂停跟随${newCount ? `（新 ${newCount} 行）` : ""}` : "";
    });
  }

  function setFollow(on) {
    state.follow = on;
    if (on) newCount = 0;
    syncToggles();
    updateJump();
    if (on) logs().forEach((el) => (el.scrollTop = el.scrollHeight));
  }

  function appendTo(el, text) {
    const kind = logKind(text);
    // C2：连续进度帧原地刷新一行，不追加。哨兵行默认不可见，
    // 所以「最后一个是哨兵行」时也应视为可折叠（帧与哨兵是交替到达的）。
    const last = el.lastElementChild;
    const collapsible =
      el._pbar &&
      el._pbar.parentNode === el &&
      (last === el._pbar || (last && last.classList.contains("ln-machine")));
    if (kind === "pbar" && collapsible) {
      el._pbar.textContent = text;
      return;
    }
    const d = document.createElement("div");
    let cls = "ln";
    if (kind === "pbar") cls += " ln-pbar";
    else if (kind === "machine") cls += " ln-machine";
    else if (RE_LOG_ERR.test(text)) cls += " ln-err";
    else if (RE_LOG_WARN.test(text)) cls += " ln-warn";
    d.className = cls;
    d.textContent = text;
    el.appendChild(d);
    // 哨兵行不打断折叠链：帧与哨兵交替到达时，帧仍刷新上一进度行
    if (kind === "pbar") el._pbar = d;
    else if (kind !== "machine") el._pbar = null;
    while (el.children.length > MAX_LOG_NODES) el.removeChild(el.firstChild);
  }

  let buf = [];
  let raf = 0;
  const flush = () => {
    raf = 0;
    if (!buf.length) return;
    const batch = buf;
    buf = [];
    logs().forEach((el) => {
      for (const t of batch) appendTo(el, t);
      if (state.follow) el.scrollTop = el.scrollHeight; // 每批只强制一次滚动定位
    });
    if (!state.follow) {
      newCount += batch.length;
      updateJump();
    }
    if (state.query) applySearch(); // 新行也要参与搜索过滤
  };

  /* 工具条注入（两页共用同一份模板）+ 绑定 */
  $$("[data-log-bar]").forEach((bar) => {
    bar.innerHTML = BAR_HTML;
    bar.querySelector("[data-lg=follow]").addEventListener("click", () =>
      setFollow(!state.follow)
    );
    bar.querySelector("[data-lg=pbar]").addEventListener("click", () => {
      state.pbar = !state.pbar;
      applyClasses();
      syncToggles();
    });
    bar.querySelector("[data-lg=machine]").addEventListener("click", () => {
      state.machine = !state.machine;
      applyClasses();
      syncToggles();
    });
    bar.querySelector("[data-lg=level]").addEventListener("change", (e) => {
      state.level = e.target.value;
      applyClasses();
      applySearch();
    });
    const si = bar.querySelector("[data-lg=query]");
    let t = 0;
    si.addEventListener("input", () => {
      clearTimeout(t);
      t = setTimeout(() => {
        state.query = si.value;
        applySearch();
      }, 150);
    });
    bar.querySelector("[data-lg=copy]").addEventListener("click", async () => {
      const el = logs()[0];
      if (!el) return;
      const keep = (n) =>
        !n.classList.contains("f-hidden") &&
        (state.pbar || !n.classList.contains("ln-pbar")) &&
        (state.machine || !n.classList.contains("ln-machine")) &&
        (state.level === "all" ||
          (state.level === "err" && n.classList.contains("ln-err")) ||
          (state.level === "warn" &&
            (n.classList.contains("ln-warn") || n.classList.contains("ln-err"))));
      const text = [...el.children].filter(keep).map((n) => n.textContent).join("\n");
      try {
        await navigator.clipboard.writeText(text);
        toast(`已复制当前可见的 ${text ? text.split("\n").length : 0} 行`, "ok");
      } catch (_) {
        toast("复制失败：浏览器拒绝了剪贴板访问", "err");
      }
    });
    bar.querySelector("[data-lg=download]").addEventListener("click", () => {
      // 下载给**全量原文**（含被隐藏的进度帧与机器行）—— 排障需要完整记录
      const text = trainer.lines.join("\n");
      const rid = (trainer.run && trainer.run.run_id) || "trainer";
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
      a.download = `${rid}.log`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      toast(`已下载完整日志 ${trainer.lines.length} 行`, "ok");
    });
  });
  syncToggles();

  /* 跟随开关：滚到底=跟随，向上滚=自动暂停（DevTools / tail -f 的通行做法） */
  logs().forEach((el) => {
    el.addEventListener("scroll", () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 8;
      if (state.follow && !atBottom) setFollow(false);
      else if (!state.follow && atBottom) setFollow(true);
    });
  });
  $$("[data-log-jump]").forEach((b) =>
    b.addEventListener("click", () => setFollow(true))
  );

  trainer.on("log", (text) => {
    if (text === "__clear__") {
      buf = [];
      newCount = 0;
      if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
      logs().forEach((el) => {
        el.textContent = "";
        el._pbar = null;
      });
      applySearch();
      updateJump();
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
}

/* ── 停止按钮与 header 训练徽章 ── */
// 注: 本函数由 boot 的 DOMContentLoaded 回调调用; 旧实现内部再嵌套一层
// DOMContentLoaded 监听, 在事件派发中注册永不触发 → 徽章/停止按钮失效。
/* F6：训练结束主动提示。
   本项目的训练以「几十小时」计，而结束信号原先只落在 header 徽章的文字上 ——
   你从别的标签页切回来根本不会注意到。
   标题前缀 + toast 两条路：前者让「切回来一眼看到」，后者在页面内留痕。
   （不做 Notification API：需要用户授权，且权限弹窗本身是打扰。） */
const BASE_TITLE = document.title;

function clearExitTitle() {
  document.title = BASE_TITLE;
}

function noticeExit(ev) {
  const st = trainer.run || {};
  const lm = st.last_metric || {};
  const task = st.task || "任务";
  const tail = lm.loss != null ? ` · loss ${fmtNum(lm.loss)}` : "";
  if (ev.status === "lost") {
    document.title = "⚠ 训练绑定丢失 · " + BASE_TITLE;
    toast("与训练进程失去绑定 —— 界面数据可能已过期，请刷新确认", "err", 9000);
  } else if (ev.code != null && ev.code !== 0) {
    document.title = "❌ 训练失败 · " + BASE_TITLE;
    // R4：失败时用户要看的是 traceback，不是推理页 —— 所以指向日志。
    toast(`${task} 失败（exit=${ev.code}）${tail}`, "err", 12000, [
      { label: "查看日志", onClick: () => gotoLogs(task) },
    ]);
  } else {
    document.title = "✅ 训练完成 · " + BASE_TITLE;
    // R3：**不自动切页** —— 你很可能正在看曲线或日志，自动跳转会打断。
    // 按钮是「邀请」而不是「劫持」（GitHub 的 Create PR / Vercel 的 Visit 同款）。
    toast(`${task} 已完成${tail}`, "ok", 12000, [
      { label: "去推理验证", onClick: () => clickTab("inference") },
    ]);
  }
}

/* 片二：切 tab —— 直接触发那个按钮的 click，与用户手点走**完全同一条**路径
   （含 aria 状态与 hash 深链更新），也不依赖 util.js 内部符号是否已暴露。 */
function clickTab(name) {
  const b = document.querySelector('#tabs .tab[data-tab="' + name + '"]');
  if (b) b.click();
}

/* R4：失败时的去向 —— 切到该任务所属的 tab，并把日志区滚入视野。
   日志面板在两训练页都有；「查看日志」比「重试训练」更贴合失败当下要看的东西
   （重试要重新配参，是更大的动作，不适合塞进 toast）。 */
function gotoLogs(task) {
  clickTab(task === "pretrain" ? "pretrain" : "posttrain");
  const el =
    document.querySelector(".tabpane.active .log-wrap") ||
    document.querySelector(".tabpane.active .log");
  if (el && el.scrollIntoView) el.scrollIntoView({ block: "center", behavior: "smooth" });
}

/* ── H18 运行条：状态机与文案（纯函数，可单测）── */
function runBarState(st) {
  if (!st) return "idle";
  if (st.status === "stopping") return "stopping";
  if (st.running) return "running";
  if (st.status === "finished") return "finished";
  if (st.status === "failed") return "failed";
  return "idle";
}

function runBarText(st, lm) {
  const s = runBarState(st);
  if (s === "stopping") return "停止中…";
  if (s === "running") {
    const step = lm && lm.step != null ? String(lm.step) : "—";
    const total = st.total_steps ? String(st.total_steps) : "?";
    return `${st.task || "任务"} · ${step} / ${total}`;
  }
  if (s === "finished") return `已完成 ${st.task || ""}`.trim();
  if (s === "failed") {
    // exit_code 可能缺失（进程被杀 / 状态还没落定）—— 不能直接拼，否则文案变成 "exit undefined"
    const code = st.exit_code != null ? ` (exit ${st.exit_code})` : "";
    return `失败 ${st.task || ""}${code}`.trim();
  }
  return "空闲";
}

/* 写入运行条。文本/状态不变就不写 DOM —— 这条每 2s 跑一次（D5）。 */
function syncRunBars(st, lm) {
  const bars = $$("[data-run-bar]");
  if (!bars.length) return;
  const state = runBarState(st);
  const text = runBarText(st, lm);
  for (const bar of bars) {
    if (bar.dataset.state !== state) bar.dataset.state = state;
    const el = bar.querySelector(".rb-state-text");
    if (el && el.textContent !== text) el.textContent = text;
  }
}

function initTrainHeader() {
  $$("[data-stop-btn]").forEach((b) => {
    b.addEventListener("click", () => trainer.stop());
  });
  trainer.on("exit", noticeExit);
  // 回到前台即清掉标题前缀（它只为「切回来看到」而存在）
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) clearExitTitle();
  });
  trainer.on("status", (st) => {
    // H18：停止中再点停止没有意义
    $$("[data-stop-btn]").forEach(
      (b) => (b.disabled = !st.running || st.status === "stopping")
    );
    $$("[data-start-btn]").forEach((b) => (b.disabled = !!st.running)); // 运行中禁用启动（变暗）
    // C6：指标条不逐帧播报（每 2s 刷新会变成噪音），只把「任务状态变化」推给读屏。
    // 文本不变就不写 —— 读屏只在真正变化时出声。
    const sr = $("#sr-status");
    if (sr) {
      const label =
        st.status === "stopping"
          ? `${st.task || "任务"} 正在停止`
          : st.running
            ? `${st.task || "任务"} 训练中`
            : st.status === "finished"
              ? `${st.task || "任务"} 已完成`
              : st.status === "failed"
                ? `${st.task || "任务"} 失败`
                : "空闲";
      if (sr.textContent !== label) sr.textContent = label;
    }
    const lm = st.last_metric || {};
    // H18：运行条独立于 header 徽章 —— 徽章缺失不该让运行条停更
    syncRunBars(st, lm);
    const badge = $("#train-text");
    const dot = $("#train-dot");
    if (!badge) return;
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

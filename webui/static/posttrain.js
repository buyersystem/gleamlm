/* ② 后训练 tab：流水线阶段卡（点击启动，表单由后端 /train/tasks 元数据驱动）、
   当前阶段 loss/lr 曲线（loss/lr 切换）+ 历史 run 勾选 A/B 灰显对比、实验历史列表。 */
"use strict";

/* 作用域隔离: 与 pretrain.js 的同名顶层函数(onStatus/onMetric/onExit/loadRuns/
   renderRuns/renderStatusLine/ensureHist)须互不可见 — 本文件在 pretrain.js 之后
   加载, 裸全局会覆盖对方实现导致预训练面板静默失效。包 IIFE 使闭包内自解析。 */
(function () {

/* 任务卡分两块：FLOW 主链按顺序排在左面板（卡间箭头连接），
   SUB 可选路线在右面板。卡片上只写方法名。 */
const SUB = [
  { task: "sft_lora", t: "LoRA" },
  { task: "ppo", t: "PPO" },
  { task: "grpo", t: "GRPO" },
];
const FLOW = [
  { task: "sft", t: "SFT" },
  { task: "dpo_data", t: "DPO数据", gen: true },
  { task: "dpo", t: "DPO" },
  { task: "opd", t: "OPD", opt: true },
];

const pt2 = {
  logY: false, // B6：loss 图对数纵轴（默认关）
  runs: [],
  hist: new Map(),
  checked: [],
  mainId: null,
  series: null,
};

function pt2RunList() {
  return pt2.runs.filter((r) => (r.config || {}).task !== "pretrain");
}

function liveTaskId() {
  const st = trainer.run || {};
  // 仅 running 时才算 live（同 livePretrainId: 已结束 run 不得再视为 live，
  // 否则拦掉回放、列表误挂 LIVE 徽章）
  return st.task && st.task !== "pretrain" && st.running ? st.run_id : null;
}

/* ── 阶段卡 ── */
function taskDesc(task) {
  const f = trainer.meta && trainer.meta.tasks && trainer.meta.tasks[task];
  return f || null;
}

/* 阶段卡与可选路线卡的静态清单（顺序即主链顺序） */
function stageSpecs() {
  return [
    ...FLOW.map((it) => ({
      task: it.task,
      name: it.t,
      extra: (it.opt ? " opt" : "") + (it.gen ? " gen" : ""),
      opt: !!it.opt,
      gen: !!it.gen,
      main: true,
    })),
    ...SUB.map((s) => ({
      task: s.task,
      name: s.t,
      extra: "",
      opt: false,
      gen: false,
      main: false,
    })),
  ];
}

// 箭头：SVG 细线，悬停前一张卡时整支变蓝
const STAGE_ARROW = () => `<svg class="flow-arrow" width="24" height="10" viewBox="0 0 24 10" aria-hidden="true">
  <line x1="1.5" y1="5" x2="13.5" y2="5"/>
  <path class="head" d="M12.2 1.6 L19.6 5 L12.2 8.4"/>
</svg>`;

/* D3：阶段状态返回「类名 + 文字」。
   原实现只算类名（边框颜色），状态完全靠颜色表达 ——
   色觉障碍、投影、灰度截图下全部失效（WCAG 1.4.1），
   而 theme.css 里的 .stage-card .s 状态文字槽位早就写好却从未输出。 */
function stageState(task, spec, latest, st) {
  const isLive = st.task === task && st.running;
  const last = latest.get(task);
  if (isLive) {
    const step = (st.last_metric || {}).step;
    return { cls: " run", tone: "run", text: step != null ? `${step} 步` : "启动中" };
  }
  if (last && last.status !== "running") {
    const m = /exit=(-?\d+)/.exec(last.note || "");
    if (m && m[1] !== "0") return { cls: " fail", tone: "fail", text: `exit ${m[1]}` };
    if (last.status === "finished") return { cls: " done", tone: "done", text: "完成" };
  }
  return { cls: "", tone: "idle", text: spec.gen ? "待生成" : spec.opt ? "可选" : "待执行" };
}

function renderStages() {
  const st = trainer.run || {};
  const box = $("#pt2-stages");
  const altBox = $("#pt2-alts");
  if (!box || !altBox) return;
  const latest = new Map();
  for (const r of pt2RunList()) {
    const t = (r.config || {}).task;
    if (t && !latest.has(t)) latest.set(t, r);
  }
  const specs = stageSpecs();
  const mainSpecs = specs.filter((s) => s.main);
  const sig = specs.map((s) => s.task).join(",");
  // D5：结构只在任务集合变化时重建 —— 否则每 2s 的状态轮询会把
  // 悬停态、键盘焦点、正在进行的文本选择全部清掉。
  if (box.dataset.sig !== sig) {
    box.dataset.sig = sig;
    const mk = (s) => {
      if (!taskDesc(s.task)) return "";
      const verb = s.task === "dpo_data" ? "生成" : "启动";
      return `<div class="stage-card${s.extra}" data-task="${s.task}" title="${verb} ${esc(s.name)}">
      <div class="t">${esc(s.name)}</div>
      <div class="s"></div>
    </div>`;
    };
    const parts = [];
    mainSpecs.forEach((s, i) => {
      parts.push(mk(s));
      if (i < mainSpecs.length - 1) parts.push(`<div class="flow-arrow">${STAGE_ARROW()}</div>`);
    });
    box.innerHTML = `<div class="flow-grid">${parts.join("")}</div>`;
    altBox.innerHTML = specs.filter((s) => !s.main).map(mk).join("");
    for (const b of [box, altBox]) {
      b.querySelectorAll(".stage-card").forEach((c) => {
        const act = () => trainer.openStartModal([c.dataset.task]);
        c.addEventListener("click", act);
        makeActivatable(c, act, `启动 ${c.dataset.task}`); // E2：键盘可达
      });
    }
  }
  // 原地刷新状态（类名 + 状态文字），不动结构
  for (const b of [box, altBox]) {
    b.querySelectorAll(".stage-card").forEach((c) => {
      const s = specs.find((x) => x.task === c.dataset.task);
      if (!s || !taskDesc(s.task)) return;
      const state = stageState(s.task, s, latest, st);
      const want = `stage-card${s.extra}${state.cls}`;
      if (c.className !== want) c.className = want;
      const sEl = c.querySelector(".s");
      if (!sEl) return;
      if (sEl.textContent !== state.text) sEl.textContent = state.text;
      if (sEl.dataset.tone !== state.tone) sEl.dataset.tone = state.tone;
    });
  }
}

/* ── 事件 ── */
function onStatus(st) {
  const liveId = liveTaskId();
  if (liveId && pt2.mainId !== liveId && st.running) {
    pt2.mainId = liveId;
    pt2.series = null;
    loadRuns(false);
  }
  renderRuns();
  renderStatusLine();
  drawPt2();
  renderStages();
}

function onMetric(m) {
  const st = trainer.run || {};
  if (!m) return;
  if (st.task && st.task !== "pretrain" && st.run_id === m.run_id) {
    pt2.mainId = m.run_id;
    pt2.series = m.series;
    drawPt2();
    renderStages();
  }
}

function onExit() {
  loadRuns(true);
}

async function loadRuns(reloadMain) {
  // 片一：同预训练页 —— 失败不清空，只置标记（stale-but-usable）
  const hadData = pt2.runs.length > 0;
  try {
    const runs = await api("/api/train/runs");
    pt2.runs = Array.isArray(runs) ? runs : [];
    pt2.runsErr = "";
    clearFetchFail("pt2-runs");
  } catch (err) {
    pt2.runsErr = (err && err.message) || "请求失败";
    noteFetchFail("pt2-runs", hadData);
  }
  const mine = pt2RunList();
  if (!pt2.mainId || !mine.some((r) => r.id === pt2.mainId) || reloadMain) {
    pt2.mainId = mine.length ? mine[0].id : null;
    pt2.series = null;
  }
  pt2.checked = pt2.checked.filter((id) => mine.some((r) => r.id === id));
  renderRuns();
  renderStatusLine();
  renderStages();
  if (pt2.mainId && !pt2.series) ensureHist(pt2.mainId);
  drawPt2();
}

/* ── 实验历史列表 ── */
function chipFor(r) {
  const t = (r.config || {}).task || "?";
  if (r.status === "running" || r.status === "interrupted") {
    // 仅当前真正 live 的 run 显运行中; 其余（服务重启后的 DB 残留 / 清扫标记）显中断
    const st = trainer.run || {};
    const live = !!st.running && st.run_id === r.id;
    return r.status === "running" && live
      ? `<span class="chip running">${t} · 运行中</span>`
      : `<span class="chip">${t} · 中断</span>`;
  }
  const m = /exit=(-?\d+)/.exec(r.note || "");
  if (m && m[1] !== "0") return `<span class="chip failed">${t} · exit ${m[1]}</span>`;
  return `<span class="chip finished">${t} · 完成</span>`;
}

function runDyn(r) {
  const cfg = r.config || {};
  const isMain = r.id === pt2.mainId;
  const live = r.id === liveTaskId();
  const chip = chipFor(r);
  return {
    isMain,
    live,
    dyn: `${isMain}|${live}|${chip}|${cfg.variant || ""}`,
    name: `${esc(r.id)}${live ? ' <span class="chip running" style="padding:0 8px">LIVE</span>' : ""}`,
    // 同 pretrain：主曲线只靠 .list-item.on 表达（副标题保留 variant 标记）
    sub: `${fmtTime(r.created_at)} · ${chip}${cfg.variant ? ` · <b>${esc(cfg.variant)}</b>` : ""}`,
    title: esc(r.id),
  };
}

function bindDelRun(b) {
  b.addEventListener("click", async (e) => {
    e.stopPropagation();
    const id = b.dataset.run;
    const okDel = await confirmDialog({
      title: "删除实验？",
      okText: "删除",
      body: `<p style="margin:0">实验 <b>${esc(id)}</b></p>
        <p style="margin:0;color:var(--dim);font-size:13px">将同时删除其指标记录与日志文件，<b>不可恢复</b>。</p>`,
    });
    if (!okDel) return;
    try {
      await api(`/api/train/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
      pt2.hist.delete(id);
      pt2.checked = pt2.checked.filter((x) => x !== id);
      loadRuns(true);
    } catch (err) {
      toast("删除失败：" + err.message, "err");
    }
  });
}

/* P2：与预训练页同一套 —— 运行中「停止」、结束「删除」，操作跟着对象状态走。 */
/* H12：A/B 角标。checked[0] → A（图里蓝色虚线），checked[1] → B（紫色虚线）。
   与图表里的 cmp 序号同源 —— 角标不是装饰，是图例的另一半。 */
function cmpBadgeHtml(checked, id) {
  const i = checked.indexOf(id);
  if (i === 0) return '<span class="ab-badge a" title="对比 A —— 图中蓝色虚线">A</span>';
  if (i === 1) return '<span class="ab-badge b" title="对比 B —— 图中紫色虚线">B</span>';
  return "";
}

/* 勾选变化时刷角标。runDyn() 的签名不含 checked，所以角标走不了常规的原地更新路径。 */
function syncCmpBadge(li, checked, id) {
  const want = cmpBadgeHtml(checked, id);
  const cur = li.querySelector(".ab-badge");
  if (!want) {
    if (cur) cur.remove();
    return;
  }
  if (cur) {
    if (cur.outerHTML !== want) cur.outerHTML = want;
    return;
  }
  const anchor = li.querySelector(".del-run"); // H17：行内已无 .stop-run
  if (anchor) anchor.insertAdjacentHTML("beforebegin", want);
  else li.insertAdjacentHTML("beforeend", want);
}

/* H17：行内只留「删除」—— 停止已收归任务列表标题栏（紧跟启动）。
   同一动作不开两个入口：停止是破坏性操作，入口收在一处才配得上 E7 的二次确认。
   运行中的行因此没有行内按钮（live → 空串）。 */
function actionBtnHtml(r, live) {
  if (live) return "";
  return `<button class="del-run" type="button" data-run="${esc(r.id)}" title="删除该 run 及其指标、日志文件（不可恢复）">✕</button>`;
}

function bindRowActions(scope) {
  const del = scope.querySelector(".del-run");
  if (del && !del.dataset.bound) bindDelRun(del);
}

function bindRunRows(box) {
  box.querySelectorAll("input[data-run]").forEach((cb) => {
    cb.addEventListener("change", () => togglePt2Compare(cb.dataset.run, cb.checked));
  });
  box.querySelectorAll(".list-item").forEach((li) => {
    const act = () => setPt2Main(li.dataset.id);
    li.addEventListener("click", (e) => {
      if (e.target.closest("input, .del-run")) return;
      act();
    });
    makeActivatable(li, act, `设为主曲线 ${li.dataset.id}`); // E2
  });
  box.querySelectorAll(".del-run").forEach((b) => bindDelRun(b));
}

function renderRuns() {
  const box = $("#pt2-runs");
  if (!box) return;
  const mine = pt2RunList();
  // 片一：警告条独立（同预训练页，不得并进 box）
  const warnEl = $("#pt2-runs-warn");
  if (warnEl) {
    const show = mine.length > 0 && failShown("pt2-runs") && !isLinkDown();
    warnEl.hidden = !show;
    if (show) warnEl.textContent = `⚠ 任务列表读取失败（当前显示的是上次数据）· ${pt2.runsErr}`;
  }
  if (!mine.length) {
    box.dataset.sig = "";
    // 片一：失败态优先于空态；同时修正原文案「点击任务按钮」
    // （实际按钮叫「启动训练」，与预训练页不一致 —— 两页现在统一）
    if (isLinkDown()) {
      box.innerHTML = '<div class="hint sm">与后端失联 —— 恢复后自动加载</div>';
    } else if (failShown("pt2-runs")) {
      box.innerHTML = failHtml(`无法读取任务列表 · ${pt2.runsErr}`);
      bindRetry(box, () => loadRuns(true));
    } else {
      box.innerHTML = '<div class="hint sm">点击「启动训练」开启后训练</div>';
    }
    return;
  }
  const sig = mine.map((r) => r.id).join("|");
  if (box.dataset.sig !== sig) {
    box.dataset.sig = sig;
    box.innerHTML = mine
      .map((r) => {
        const d = runDyn(r);
        return `<div class="list-item ${d.isMain ? "on" : ""}" data-id="${esc(r.id)}" title="${d.title}">
    <input type="checkbox" class="check" data-run="${esc(r.id)}" ${pt2.checked.includes(r.id) ? "checked" : ""} title="勾选与主曲线 A/B 对比（最多 2 条）" />
    <div style="flex:1;min-width:0">
      <div class="name" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${d.name}</div>
      <div class="sub">${d.sub}</div>
    </div>
    ${cmpBadgeHtml(pt2.checked, r.id)}
    ${actionBtnHtml(r, d.live)}
  </div>`;
      })
      .join("");
    bindRunRows(box);
    return;
  }
  // D5：集合未变 → 只刷动态部分（保住悬停/焦点/文本选择）
  for (const r of mine) {
    const li = box.querySelector(`.list-item[data-id="${CSS.escape(r.id)}"]`);
    if (!li) continue;
    const d = runDyn(r);
    const wantChecked = pt2.checked.includes(r.id);
    const cb = li.querySelector("input[data-run]");
    if (cb && cb.checked !== wantChecked && document.activeElement !== cb) {
      cb.checked = wantChecked;
    }
    if (li.dataset.dyn === d.dyn) continue;
    li.dataset.dyn = d.dyn;
    li.classList.toggle("on", d.isMain);
    li.title = d.title;
    // 空值保护：这里一旦抛异常，2s 轮询会静默失效（页面停在旧数据上，
    // 不报错也不提示）—— 正是本项目反复吃亏的那类故障。
    const nameEl = li.querySelector(".name");
    const subEl = li.querySelector(".sub");
    if (nameEl) nameEl.innerHTML = d.name;
    if (subEl) subEl.innerHTML = d.sub;
    syncCmpBadge(li, pt2.checked, r.id); // H12：勾选变了角标要跟着变
    // H17：运行中无行内按钮，结束后补上 ✕ 删除。
    // 稳态下不写 DOM（这条路径每 2s 跑一次，见 D5）。
    const stopBtn = li.querySelector(".stop-run");
    const delBtn = li.querySelector(".del-run");
    if (d.live) {
      // H17：运行中的行不再有行内按钮（停止已收归标题栏）
      if (stopBtn) stopBtn.remove();
      if (delBtn) delBtn.remove();
    } else if (!delBtn) {
      if (stopBtn) stopBtn.remove(); // 兼容旧 DOM（页面是从更早版本热切换过来的）
      li.insertAdjacentHTML("beforeend", actionBtnHtml(r, false));
      bindRowActions(li);
    }
  }
}

function togglePt2Compare(id, on) {
  if (on) {
    if (pt2.checked.length >= 2) {
      toast("对比最多勾选 2 个历史 run（主曲线之外）", "err");
      renderRuns();
      return;
    }
    if (!pt2.checked.includes(id)) pt2.checked.push(id);
  } else {
    pt2.checked = pt2.checked.filter((x) => x !== id);
  }
  ensureHist(id);
  drawPt2();
}

async function ensureHist(id) {
  if (!id || pt2.hist.has(id)) return;
  // 片一：失败不写空对象（原因同预训练页：has(id) 会让重试拉不动）
  try {
    const d = await api(`/api/train/metrics?run_id=${encodeURIComponent(id)}`);
    pt2.hist.set(id, d.series || {});
    pt2.histErr = "";
    clearFetchFail("pt2-hist:" + id);
  } catch (err) {
    pt2.histErr = (err && err.message) || "请求失败";
    noteFetchFail("pt2-hist:" + id, false);
  }
  setChartErr(
    ["pt2-loss-err", "pt2-lr-err"],
    failShown("pt2-hist:" + id) && !isLinkDown()
      ? `无法读取该任务的曲线数据 · ${pt2.histErr}`
      : "",
    () => ensureHist(id),
  );
  drawPt2();
}

function setPt2Main(id) {
  if (trainer.run && trainer.run.running) {
    if (liveTaskId() === id) return;
    toast("训练运行中，主曲线跟随当前任务 — 可在停止后回看历史曲线与日志");
    return;
  }
  pt2.mainId = id;
  pt2.series = null;
  renderRuns();
  renderStatusLine();
  ensureHist(id);
  if (id !== liveTaskId()) trainer.replay(id);
}

/* ── H4：后训练各阶段专属指标（设计文档 §7.3）──
   名单与后端 `_EXTRA_KEYS`（webui/routers/training.py）是**同一份契约**：
   改一处必须同时改另一处，接线校验会比对两者。
   顺序即优先级；指标条只留 2 格，所以取前 2 个真实存在的。 */
const PT2_EXTRA_KEYS = ["margin", "acc", "reward", "kl", "len"];
const PT2_EXTRA_DEFAULT = ["margin", "acc"]; // 无数据时的槽位标签（与提案图一致）
/* 同一族的配对默认值：只拿到 margin 时第 2 格仍叫 acc，只拿到 reward 时叫 kl */
const PT2_EXTRA_FAMILY = { margin: "acc", acc: "acc", reward: "kl", kl: "kl", len: "len" };

/* 该序列**实际拥有**的专属指标（最多 2 个）。live 时并入 last_metric ——
   第一个点还没落库时序列为空，只看序列会让指标条晚一个周期才亮。 */
function pt2Extras(series, lm) {
  const has = (k) => !!(series && series[k] && series[k].length) || !!(lm && lm[k] != null);
  return PT2_EXTRA_KEYS.filter(has).slice(0, 2);
}

/* 槽位标签：有数据用数据键名，无数据用默认（DPO 家族）。 */
function pt2SlotLabels(xk) {
  if (!xk.length) return PT2_EXTRA_DEFAULT.slice();
  return [xk[0], xk[1] || PT2_EXTRA_FAMILY[xk[0]] || PT2_EXTRA_DEFAULT[1]];
}

/* ── 曲线：左 loss 主图 / 右 副图（默认 lr，有专属指标时切换，见 H4）── */
let pt2LossChart = null;
let pt2LrChart = null;

/* ── 片三：导出曲线 CSV（同预训练页：原始 series、不抽稀、不四舍五入）──
   前缀与左栏角标对齐（main / A / B）。 */
async function exportCsv() {
  const items = [];
  if (pt2.mainId) items.push({ tag: "main", id: pt2.mainId });
  pt2.checked.forEach((id, i) => items.push({ tag: i === 0 ? "A" : "B", id }));
  if (!items.length) {
    toast("没有可导出的曲线 —— 先在左栏选一个 run", "err");
    return;
  }
  const btn = $("#pt2-csv");
  if (btn) btn.disabled = true;
  try {
    await Promise.all(items.map((r) => ensureHist(r.id)));
    const named = items.map((r) => ({
      name: r.tag,
      series: r.id === pt2.mainId && pt2.series ? pt2.series : pt2.hist.get(r.id),
    }));
    const csv = seriesToCsv(named);
    if (!csv) {
      toast("该 run 还没有曲线数据（可能还没开始记录）", "err");
      return;
    }
    downloadText(csv, `${pt2.mainId}.csv`, "text/csv;charset=utf-8");
    toast(`已导出 ${csv.trim().split("\r\n").length - 1} 行 · ${items.length} 条曲线`, "ok");
  } finally {
    if (btn) btn.disabled = false;
  }
}
function drawPt2() {
  const main = pt2.series || pt2.hist.get(pt2.mainId) || null;
  const lm = (trainer.run || {}).last_metric || {};
  const loss = [];
  const sub = [];
  const C = LineChart.palette(); // B2：色板来自 :root
  const xk = pt2Extras(main, lm);
  const subTitle = $("#pt2-sub-title");
  if (subTitle) {
    const want = xk.length ? xk.join(" / ") : "学习率 LR";
    if (subTitle.textContent !== want) subTitle.textContent = want; // D5：不变不写
  }
  if (main) {
    loss.push({ name: shortPt2Id(pt2.mainId), color: C.loss, points: main.loss || [] });
    if (xk.length) {
      // 专属指标模式：副图只画主 run（提案图如此），跨 run 的数值比较交给差异条
      xk.forEach((k, i) =>
        sub.push({ name: k, color: i === 0 ? C.x1 : C.x2, points: main[k] || [] })
      );
    } else {
      sub.push({ name: shortPt2Id(pt2.mainId), color: C.lr, points: main.lr || [] });
    }
  }
  // B4/Q3：对比线按勾选顺序 cmp=0/1，用不同线型区分（原先是统一灰）
  pt2.checked.forEach((id, i) => {
    const s = pt2.hist.get(id);
    if (!s) return;
    loss.push({ name: shortPt2Id(id), color: C.loss, points: s.loss || [], cmp: i });
    // lr 模式保留对比线（与改动前一致）；专属指标模式不加，避免副图图例被撑爆
    if (!xk.length) sub.push({ name: shortPt2Id(id), color: C.lr, points: s.lr || [], cmp: i });
  });
  if (!pt2LossChart) pt2LossChart = new LineChart($("#pt2-loss-chart"));
  if (!pt2LrChart) pt2LrChart = new LineChart($("#pt2-lr-chart"));
  // zeroY：两张图的纵轴都从 0 起（副图切到 margin/acc 时同样；
  // 若该指标出现负值，yDomain 会保留 dataMin 而不是贴 0 —— 不裁掉负半轴）
  pt2LossChart.render({
    series: loss, yLabel: "loss", xLabel: "step", logY: pt2.logY, zeroY: true,
  });
  pt2LrChart.render({
    series: sub, yLabel: xk.length ? "value" : "lr", xLabel: "step", zeroY: true,
  });
  renderMetrics();
  renderAb(); // D4
}

/* D4：A/B 最终指标差异条（后训练轨） */
function renderAb() {
  const seriesOf = (id) => pt2.hist.get(id) || (id === pt2.mainId ? pt2.series : null);
  if (!pt2.mainId || !pt2.checked.length) {
    renderAbDiff("#pt2-abdiff", null, null);
    return;
  }
  const bId = pt2.checked[0];
  const aS = seriesOf(pt2.mainId);
  const bS = seriesOf(bId);
  // H4：主指标恒为 loss；专属指标只在**两侧都有点**时才比（最多 2 个 →
  // 最多 3 片，与提案图的 loss/margin/acc 三片一致）。
  const metrics = ["loss"].concat(
    PT2_EXTRA_KEYS.filter((k) => lastPoint(aS, k) && lastPoint(bS, k)).slice(0, 2)
  );
  renderAbDiff(
    "#pt2-abdiff",
    { name: shortPt2Id(pt2.mainId), series: aS },
    { name: shortPt2Id(bId), series: bS },
    metrics,
  );
}

/* 槽位标签随数据切换（只在变化时写 —— D5：稳态零 DOM 写入）。 */
function setSlotLabels(rootSel, labels) {
  const root = $(rootSel);
  if (!root) return;
  labels.slice(0, 2).forEach((txt, i) => {
    const el = root.querySelector(`[data-mk="x${i + 1}"]`);
    if (el && el.textContent !== txt) el.textContent = txt;
  });
}

/* ── D1/D2 + H4 指标条（后训练）：live run 取 last_metric，否则取所选 run 末端值。
   槽位与提案图一致为**六格**：step · loss · <专属> · <专属> · lr · ETA。
   第 3/4 格是阶段专属指标，标签随数据切换；无数据时保留槽位与默认标签、值为「—」
   —— 槽位数不变，格宽比例才与提案图一致；数据到位即自动可见（不需要再动前端）。
   tok·s 与 GPU 已移出本条：tok·s 进日志标题栏的运行摘要（只覆盖"正在跑"的场景），
   GPU 不再显示（header 徽章是面板侧 NVML 全卡占用，两处并存反而容易误读）。── */
function renderMetrics() {
  const st = trainer.run || {};
  const lm = st.last_metric || {};
  const live = !!st.task && st.task !== "pretrain" && st.running === true;
  const series = pt2.series || pt2.hist.get(pt2.mainId) || null;
  const tail = (k) => {
    const a = series && series[k];
    return a && a.length ? a[a.length - 1] : null;
  };
  const xk = pt2Extras(series, live ? lm : null);
  const labels = pt2SlotLabels(xk);
  const xval = (i) => {
    if (live) return lm[labels[i]];
    const t = tail(labels[i]);
    return t && t[1];
  };
  let step, loss, lr, eta = "";
  if (live) {
    step = lm.step;
    loss = lm.loss;
    lr = lm.lr;
    eta = estimateEta(st);
  } else {
    const sl = tail("loss");
    const sr = tail("lr");
    step = sl && sl[0];
    loss = sl && sl[1];
    lr = sr && sr[1];
  }
  const x1 = xval(0);
  const x2 = xval(1);
  setMetrics("#pt2-metrics", {
    step: step != null ? step : "",
    loss: loss != null ? fmtNum(loss) : "",
    x1: x1 != null ? fmtNum(x1) : "",
    x2: x2 != null ? fmtNum(x2) : "",
    lr: lr != null ? fmtSci(lr) : "",
    eta,
  });
  setSlotLabels("#pt2-metrics", labels);
}

function shortPt2Id(id) {
  if (!id) return "";
  const s = String(id).split("_");
  return s.length >= 3 ? `${s[0]}_${s[s.length - 1]}` : id;
}

/* ── 日志标题行的状态摘要（单行精简：状态 + 短 run id + step/loss/lr/tok·s，
   详情以 title 悬停展示）。tok·s 于 H4 从指标条移入此处。── */
function renderStatusLine() {
  const el = $("#pt2-status");
  if (!el) return;
  const st = trainer.run || {};
  if (!st.run_id) {
    el.textContent = "空闲";
    el.title = "";
    return;
  }
  const lm = st.last_metric || {};
  if (st.task === "pretrain") {
    // 预训练占用中：只提示等待，run 细节在预训练页
    el.title = st.run_id;
    el.textContent = "预训练运行中";
    return;
  }
  if (!st.task) {
    el.textContent = "";
    el.title = "";
    return;
  }
  const step = lm.step != null ? ` · ${lm.step} step` : "";
  const loss = lm.loss != null ? ` · loss ${fmtNum(lm.loss)}` : "";
  const lr = lm.lr != null ? ` · lr ${fmtSci(lm.lr)}` : "";
  // H4：tok·s 从指标条移到这里（不再占六格中的一格）。
  // 注意: 只覆盖"正在跑"的场景 —— 历史 run 没有 last_metric，不显示吞吐。
  const tokTxt = fmtTok(lm.tok_per_sec);
  const tok = tokTxt ? ` · ${tokTxt} tok/s` : "";
  const state = st.status === "stopping" ? "停止中" : st.running ? "运行中" : "已结束";
  el.title = st.cmd || st.run_id;
  el.innerHTML = `<b>${state} ${esc(shortPt2Id(st.run_id))}</b>${step}${loss}${lr}${tok}`;
}

/* ── boot ── */
/* B6：log y 按钮的可见状态（本 tab 一份，见 pretrain.js 的同名说明）。 */
function applyLogYBtn() {
  const b = $("#pt2-logy");
  if (!b) return;
  b.setAttribute("aria-pressed", pt2.logY ? "true" : "false");
  b.classList.toggle("seg-on", pt2.logY);
}

function initPosttrain() {
  // 片三：导出曲线 CSV
  $("#pt2-csv").addEventListener("click", exportCsv);
  // B6：log y 开关（持久化键与预训练页共用 —— 这是同一个"我习惯看对数轴"的偏好）
  try {
    pt2.logY = localStorage.getItem("webui.logY") === "1";
  } catch (_) {
    pt2.logY = false;
  }
  applyLogYBtn();
  $("#pt2-logy").addEventListener("click", () => {
    pt2.logY = !pt2.logY;
    try {
      localStorage.setItem("webui.logY", pt2.logY ? "1" : "0");
    } catch (_) {
      /* 隐私模式等写入失败：本次会话内仍然生效 */
    }
    applyLogYBtn();
    drawPt2();
  });
  $("#pt2-start-btn").addEventListener("click", () => {
    const meta = trainer.meta;
    if (!meta || !meta.tasks) {
      toast("任务元数据尚未加载，请刷新页面", "err");
      return;
    }
    trainer.openStartModal(Object.keys(meta.tasks).filter((t) => t !== "pretrain"));
  });
  trainer.on("status", onStatus);
  trainer.on("metric", onMetric);
  trainer.on("exit", onExit);
  // meta 就绪补渲染（初次 loadRuns 时 /api/train/tasks 可能未返回, 卡为空）
  trainer.on("ready", () => renderStages());
  loadRuns(true);
  // 切回本页时重绘曲线：tab 隐藏期间首渲的 canvas 位图为 0×0（无布局尺寸），
  // 切回后必须补一次 draw 才会按真实尺寸重建位图
  window.addEventListener("tabchange", (e) => {
    if (e.detail === "posttrain") drawPt2();
  });
}

document.addEventListener("DOMContentLoaded", initPosttrain);
})();

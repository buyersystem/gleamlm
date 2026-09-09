/* ② 后训练 tab：流水线阶段卡（点击启动，表单由后端 /train/tasks 元数据驱动）、
   当前阶段 loss/lr 曲线（loss/lr 切换）+ 历史 run 勾选 A/B 灰显对比、实验历史列表。 */
"use strict";

/* 作用域隔离: 与 pretrain.js 的同名顶层函数(onStatus/onMetric/onExit/loadRuns/
   renderRuns/renderStatusLine/ensureHist)须互不可见 — 本文件在 pretrain.js 之后
   加载, 裸全局会覆盖对方实现导致预训练面板静默失效。包 IIFE 使闭包内自解析。 */
(function () {
const PT2_C = { loss: "#5dd0c9", lr: "#f5a97f" };

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
  return st.task && st.task !== "pretrain" ? st.run_id : null;
}

/* ── 阶段卡 ── */
function taskDesc(task) {
  const f = trainer.meta && trainer.meta.tasks && trainer.meta.tasks[task];
  return f || null;
}

function renderStages() {
  const st = trainer.run || {};
  const box = $("#pt2-stages");
  const altBox = $("#pt2-alts");
  if (!box || !altBox) return;
  // 各任务最近一次 run（完成态着色用）
  const latest = new Map();
  for (const r of pt2RunList()) {
    const t = (r.config || {}).task;
    if (t && !latest.has(t)) latest.set(t, r);
  }
  const stageHtml = (task, name, extra) => {
    const meta = taskDesc(task);
    if (!meta) return "";
    const isLive = st.task === task && st.running;
    const last = latest.get(task);
    let cls = "";
    if (isLive) cls = " run";
    else if (last && last.status !== "running") {
      const m = /exit=(-?\d+)/.exec(last.note || "");
      if (m && m[1] !== "0") cls = " fail";
      else if (last.status === "finished") cls = " done";
    }
    const liveTxt = isLive && st.run_id
      ? `<span class="chip running" style="margin-left:4px">step ${(st.last_metric || {}).step ?? "…"}</span>` : "";
    // dpo_data 是数据生成任务，动作用“生成”
    const verb = task === "dpo_data" ? "生成" : "启动";
    return `<div class="stage-card${extra}${cls}" data-task="${task}" title="${verb} ${esc(name)}">
      <div class="t">${esc(name)}${liveTxt}</div>
    </div>`;
  };
  // 箭头：SVG 细线，悬停前一张卡时整支变蓝
  const ARROW = () => `<svg class="flow-arrow" width="24" height="10" viewBox="0 0 24 10" aria-hidden="true">
  <line x1="1.5" y1="5" x2="13.5" y2="5"/>
  <path class="head" d="M12.2 1.6 L19.6 5 L12.2 8.4"/>
</svg>`;
  const parts = [];
  FLOW.forEach((it, i) => {
    parts.push(stageHtml(it.task, it.t, (it.opt ? " opt" : "") + (it.gen ? " gen" : "")));
    if (i < FLOW.length - 1) parts.push(`<div class="flow-arrow">${ARROW()}</div>`);
  });
  box.innerHTML = `<div class="flow-grid">${parts.join("")}</div>`;
  // 可选路线卡（右面板）
  altBox.innerHTML = SUB.map((s) => stageHtml(s.task, s.t, "")).join("");
  for (const b of [box, altBox]) {
    b.querySelectorAll(".stage-card").forEach((c) => {
      c.addEventListener("click", () => trainer.openStartModal([c.dataset.task]));
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
  try {
    const runs = await api("/api/train/runs");
    pt2.runs = Array.isArray(runs) ? runs : [];
  } catch (_) {
    pt2.runs = [];
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

function renderRuns() {
  const box = $("#pt2-runs");
  const mine = pt2RunList();
  if (!box) return;
  if (!mine.length) {
    box.innerHTML = '<div class="hint" style="padding:8px 4px"> 点击任务按钮启动训练 </div>';
    return;
  }
  box.innerHTML = mine
    .map((r) => {
      const cfg = r.config || {};
      const isMain = r.id === pt2.mainId;
      const live = r.id === liveTaskId();
      return `<div class="list-item ${isMain ? "on" : ""}" data-id="${r.id}" title="${esc(r.id)}">
    <input type="checkbox" class="check" data-run="${esc(r.id)}" ${pt2.checked.includes(r.id) ? "checked" : ""} title="勾选与主曲线 A/B 对比（最多 2 条）" />
    <div style="flex:1;min-width:0">
      <div class="name" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.id)}${live ? ' <span class="chip running" style="padding:0 6px">LIVE</span>' : ""}</div>
      <div class="sub">${fmtTime(r.created_at)} · ${chipFor(r)}${cfg.variant ? ` · <b>${esc(cfg.variant)}</b>` : ""}${isMain ? " · <b>主曲线</b>" : ""}</div>
    </div>
    ${live ? "" : `<button class="del-run" type="button" data-run="${esc(r.id)}" title="删除该 run 及其指标、日志文件（不可恢复）">✕</button>`}
  </div>`;
    })
    .join("");
  box.querySelectorAll("input[data-run]").forEach((cb) => {
    cb.addEventListener("change", () => togglePt2Compare(cb.dataset.run, cb.checked));
  });
  box.querySelectorAll(".list-item").forEach((li) => {
    li.addEventListener("click", (e) => {
      if (e.target.closest("input")) return;
      setPt2Main(li.dataset.id);
    });
  });
  box.querySelectorAll(".del-run").forEach((b) => {
    b.addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = b.dataset.run;
      if (!confirm(`删除实验 ${id}？将同时删除其指标记录与日志文件，不可恢复。`)) return;
      try {
        await api(`/api/train/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
        pt2.hist.delete(id);
        pt2.checked = pt2.checked.filter((x) => x !== id);
        loadRuns(true);
      } catch (err) {
        alert("删除失败：" + err.message);
      }
    });
  });
}

function togglePt2Compare(id, on) {
  if (on) {
    if (pt2.checked.length >= 2) {
      alert("对比最多勾选 2 个历史 run（主曲线之外）");
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
  try {
    const d = await api(`/api/train/metrics?run_id=${encodeURIComponent(id)}`);
    pt2.hist.set(id, d.series || {});
  } catch (_) {
    pt2.hist.set(id, {});
  }
  drawPt2();
}

function setPt2Main(id) {
  if (trainer.run && trainer.run.running) {
    if (liveTaskId() === id) return;
    alert("训练运行中，主曲线跟随当前任务 — 可在停止后回看历史曲线与日志");
    return;
  }
  pt2.mainId = id;
  pt2.series = null;
  renderRuns();
  renderStatusLine();
  ensureHist(id);
  if (id !== liveTaskId()) trainer.replay(id);
}

/* ── 曲线：左 loss / 右 lr 两个独立图（主曲线 + 勾选历史 A/B 灰显）── */
let pt2LossChart = null;
let pt2LrChart = null;
function drawPt2() {
  const main = pt2.series || pt2.hist.get(pt2.mainId) || null;
  const loss = [];
  const lr = [];
  if (main) {
    loss.push({ name: shortPt2Id(pt2.mainId), color: PT2_C.loss, points: main.loss || [] });
    lr.push({ name: shortPt2Id(pt2.mainId), color: PT2_C.lr, points: main.lr || [] });
  }
  for (const id of pt2.checked) {
    const s = pt2.hist.get(id);
    if (!s) continue;
    loss.push({ name: shortPt2Id(id), color: PT2_C.loss, points: s.loss || [], dim: true });
    lr.push({ name: shortPt2Id(id), color: PT2_C.lr, points: s.lr || [], dim: true });
  }
  if (!pt2LossChart) pt2LossChart = new LineChart($("#pt2-loss-chart"));
  if (!pt2LrChart) pt2LrChart = new LineChart($("#pt2-lr-chart"));
  pt2LossChart.render({ series: loss, yLabel: "loss", xLabel: "step" });
  pt2LrChart.render({ series: lr, yLabel: "lr", xLabel: "step" });
  const lm = (trainer.run || {}).last_metric || {};
  const liveL = $("#pt2-loss-live");
  if (liveL) liveL.textContent = main && lm.step != null ? `step ${lm.step} · loss ${fmtNum(lm.loss)}` : "";
  const liveR = $("#pt2-lr-live");
  if (liveR) liveR.textContent = main && lm.step != null ? `lr ${fmtNum(lm.lr, 6)}` : "";
}

function shortPt2Id(id) {
  if (!id) return "";
  const s = String(id).split("_");
  return s.length >= 3 ? `${s[0]}_${s[s.length - 1]}` : id;
}

/* ── 日志标题行的状态摘要（位于「训练日志」与启动按钮之间；
   单行精简：状态 + 短 run id + step/loss/lr，详情以 title 悬停展示）── */
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
  const lr = lm.lr != null ? ` · lr ${fmtNum(lm.lr, 6)}` : "";
  const state = st.status === "stopping" ? "停止中" : st.running ? "运行中" : "已结束";
  el.title = st.cmd || st.run_id;
  el.innerHTML = `<b>${state} ${esc(shortPt2Id(st.run_id))}</b>${step}${loss}${lr}`;
}

/* ── boot ── */
function initPosttrain() {
  $("#pt2-start-btn").addEventListener("click", () => {
    const meta = trainer.meta;
    if (!meta || !meta.tasks) {
      alert("任务元数据尚未加载，请刷新页面");
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

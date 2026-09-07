/* ① 预训练 tab：loss/lr 曲线（当前 run 主色实时 + 历史勾选灰色叠加）、
   run 列表（点击主展示 / 无任务时可重放日志）、YAML 配置编辑器。
   指标数据源 = trainer 的 metric 事件（后端 tracker SQLite，panel 唯一写入方）。 */
"use strict";

/* 作用域隔离: 本文件与 posttrain.js 存在同名顶层函数(onStatus/onMetric/onExit/
   loadRuns/renderRuns/renderStatusLine/ensureHist), 后加载者会覆盖全局名, 导致
   本 tab 注册的回调与内部调用全指向对方实现。包 IIFE 使闭包内自解析, 互不干扰。 */
(function () {
const PT_C = { loss: "#5dd0c9", lr: "#f5a97f" };
const pt = {
  runs: [],       // /api/train/runs 条目（仅 pretrain）
  hist: new Map(),// run_id → series {loss:[[s,v]], lr:[...]}
  checked: [],    // 勾选对比的 run id（≤2）
  mainId: null,   // 主曲线 run id
  series: null,   // 主曲线数据（live metric 事件喂）
  live: false,    // 当前 live run 是否为 pretrain
  busy: false,    // 防抖
};

function ptRunList() {
  return pt.runs.filter((r) => (r.config || {}).task === "pretrain");
}

function livePretrainId() {
  const st = trainer.run || {};
  return st.task === "pretrain" && st.run_id ? st.run_id : null;
}

/* ── 事件接入 ── */
function onStatus(st) {
  const liveId = livePretrainId();
  const stTask = st.task;
  // run 变化（新启动/切换任务）→ 刷新列表并调整主曲线
  if (liveId && pt.mainId !== liveId && st.running) {
    pt.mainId = liveId;
    pt.series = null;
    pt.live = true;
    loadRuns(false);
  } else if (!liveId && stTask !== "pretrain" && st.run_id) {
    pt.live = false;
    renderStatusLine();
  }
  if (liveId && st.running === false && pt.live) {
    pt.live = false; // 刚结束的 pretrain run：曲线保留最后一帧
    renderStatusLine();
    loadRuns(false);
  }
  renderRuns();
  renderStatusLine();
  drawCharts();
}

function onMetric(m) {
  const liveId = livePretrainId();
  if (!m) {
    pt.live = !!liveId && (trainer.run || {}).running;
    return;
  }
  // metric 事件总是当前 live run 的全量 series
  const st = trainer.run || {};
  if (st.task === "pretrain" && st.run_id === m.run_id) {
    pt.mainId = m.run_id;
    pt.series = m.series;
    pt.live = !!st.running;
    drawCharts();
  }
}

function onExit() {
  pt.live = false;
  loadRuns(true);
}

/* ── run 列表 ── */
async function loadRuns(reloadMain) {
  try {
    const runs = await api("/api/train/runs");
    pt.runs = Array.isArray(runs) ? runs : [];
  } catch (_) {
    pt.runs = [];
  }
  const mine = ptRunList();
  if (!pt.mainId || !mine.some((r) => r.id === pt.mainId) || reloadMain) {
    pt.mainId = mine.length ? mine[0].id : null;
    pt.series = null;
    pt.live = livePretrainId() === pt.mainId;
  }
  pt.checked = pt.checked.filter((id) => mine.some((r) => r.id === id));
  renderRuns();
  renderStatusLine();
  if (pt.mainId && !pt.live) ensureHist(pt.mainId);
  drawCharts();
}

function statusChip(r) {
  if (r.status === "running" || r.status === "interrupted") {
    // 仅当前真正 live 的 run 显运行中; 其余（服务重启后的 DB 残留 / 清扫标记）显中断
    return r.status === "running" && livePretrainId() === r.id
      ? '<span class="chip running">运行中</span>'
      : '<span class="chip">中断</span>';
  }
  const m = /exit=(-?\d+)/.exec(r.note || "");
  if (m && m[1] !== "0") return `<span class="chip failed">exit ${m[1]}</span>`;
  return '<span class="chip finished">完成</span>';
}

function renderRuns() {
  const box = $("#pt-runs");
  const mine = ptRunList();
  if (!mine.length) {
    box.innerHTML = '<div class="hint" style="padding:8px 4px"> 点击「＋ 启动预训练」开启训练 </div>';
    return;
  }
  box.innerHTML = mine
    .map((r) => {
      const isMain = r.id === pt.mainId;
      const isChecked = pt.checked.includes(r.id);
      const live = r.id === livePretrainId();
      return `<div class="list-item ${isMain ? "on" : ""}" data-id="${r.id}" title="${esc(r.id)} · 点击设为主曲线${trainer.run && trainer.run.running ? "" : "（无任务运行中，可重放日志）"}">
    <input type="checkbox" class="check" data-run="${esc(r.id)}" ${isChecked ? "checked" : ""} title="勾选与主曲线 A/B 对比（最多 2 条）" />
    <div style="flex:1;min-width:0">
      <div class="name" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.id)}${live ? ' <span class="chip running" style="padding:0 6px">LIVE</span>' : ""}</div>
      <div class="sub">${fmtTime(r.created_at)} · ${statusChip(r)}${isMain ? " · <b>主曲线</b>" : ""}</div>
    </div>
    ${live ? "" : `<button class="del-run" type="button" data-run="${esc(r.id)}" title="删除该 run 及其指标、日志文件（不可恢复）">✕</button>`}
  </div>`;
    })
    .join("");
  box.querySelectorAll("input[data-run]").forEach((cb) => {
    cb.addEventListener("change", () => toggleCompare(cb.dataset.run, cb.checked));
  });
  box.querySelectorAll(".list-item").forEach((li) => {
    li.addEventListener("click", (e) => {
      if (e.target.closest("input")) return;
      setMain(li.dataset.id);
    });
  });
  box.querySelectorAll(".del-run").forEach((b) => {
    b.addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = b.dataset.run;
      if (!confirm(`删除实验 ${id}？将同时删除其指标记录与日志文件，不可恢复。`)) return;
      try {
        await api(`/api/train/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
        pt.hist.delete(id);
        pt.checked = pt.checked.filter((x) => x !== id);
        loadRuns(true);
      } catch (err) {
        alert("删除失败：" + err.message);
      }
    });
  });
}

function toggleCompare(id, on) {
  if (on) {
    if (pt.checked.length >= 2) {
      alert("对比最多勾选 2 个历史 run（主曲线之外）");
      renderRuns();
      return;
    }
    if (!pt.checked.includes(id)) pt.checked.push(id);
  } else {
    pt.checked = pt.checked.filter((x) => x !== id);
  }
  ensureHist(id);
  drawCharts();
}

async function ensureHist(id) {
  if (!id || pt.hist.has(id)) return;
  try {
    const d = await api(`/api/train/metrics?run_id=${encodeURIComponent(id)}`);
    pt.hist.set(id, d.series || {});
  } catch (_) {
    pt.hist.set(id, {});
  }
  drawCharts();
}

function setMain(id) {
  if (trainer.run && trainer.run.running) {
    const st = trainer.run;
    if (st.task === "pretrain" && st.run_id === id) return;
    alert("训练运行中，主曲线跟随当前任务 — 可在停止后回看历史曲线与日志");
    return;
  }
  pt.mainId = id;
  pt.series = null;
  pt.live = false;
  renderRuns();
  renderStatusLine();
  ensureHist(id);
  if (id !== livePretrainId()) trainer.replay(id); // 无 live 任务 → 顺带重放日志
}

/* ── 曲线 ── */
function mainSeriesData() {
  // live / 刚结束的主曲线：series 事件实时喂；历史 run：拉取缓存
  if (pt.series) return pt.series;
  return pt.hist.get(pt.mainId) || null;
}

function lrPhases() {
  const r = pt.runs.find((x) => x.id === pt.mainId);
  const sum = r && r.config && r.config.yaml_summary;
  const lrCfg = sum && sum.lr;
  const st = trainer.run || {};
  const total = st.run_id === pt.mainId ? st.total_steps : null;
  if (!lrCfg || !total || lrCfg.type !== "wsd") return [];
  const w = parseFloat(lrCfg.warmup_ratio);
  if (!isFinite(w) || w <= 0) return [];
  const s = parseFloat(lrCfg.stable_ratio);
  const out = [{ x: Math.round(total * w), label: "warmup 结束" }];
  if (isFinite(s) && s > 0) out.push({ x: Math.round(total * (w + s)), label: "decay 开始" });
  return out;
}

let drawTimer = null;
function drawCharts() {
  clearTimeout(drawTimer);
  drawTimer = setTimeout(() => {
    const main = mainSeriesData();
    const loss = [], lr = [];
    if (main) {
      loss.push({ name: shortId(pt.mainId), color: PT_C.loss, points: main.loss || [] });
      lr.push({ name: shortId(pt.mainId), color: PT_C.lr, points: main.lr || [] });
    }
    for (const id of pt.checked) {
      const s = pt.hist.get(id);
      if (!s) continue;
      loss.push({ name: shortId(id), color: PT_C.loss, points: s.loss || [], dim: true });
      lr.push({ name: shortId(id), color: PT_C.lr, points: s.lr || [], dim: true });
    }
    if (!pt.lossChart) {
      pt.lossChart = new LineChart($("#pt-loss-chart"));
      pt.lrChart = new LineChart($("#pt-lr-chart"));
    }
    pt.lossChart.render({ series: loss, yLabel: "loss", xLabel: "step" });
    pt.lrChart.render({ series: lr, phases: lrPhases(), yLabel: "lr", xLabel: "step" });
    const lm = (trainer.run || {}).last_metric || {};
    $("#pt-loss-live").textContent = main && lm.step != null
      ? `step ${lm.step} · loss ${fmtNum(lm.loss)}` : "";
    $("#pt-lr-live").textContent = main && lm.step != null ? `lr ${fmtNum(lm.lr, 6)}` : "";
  }, 60);
}

function shortId(id) {
  if (!id) return "";
  const s = String(id).split("_");
  return s.length >= 3 ? `${s[0]}_${s[s.length - 1]}` : id;
}

/* ── 日志标题行的 run 摘要（位于「训练日志」与「＋ 启动预训练」之间；
   单行精简：状态 + 短 run id + step/loss/lr/GPU，详情以 title 悬停展示）── */
function renderStatusLine() {
  const el = $("#pt-run-info");
  if (!el) return;
  const st = trainer.run || {};
  if (!st.run_id) {
    el.textContent = "";
    return;
  }
  const lm = st.last_metric || {};
  const sid = shortId(st.run_id);
  const step = lm.step != null ? ` · ${lm.step} step` : "";
  const loss = lm.loss != null ? ` · loss ${fmtNum(lm.loss)}` : "";
  const lr = lm.lr != null ? ` · lr ${fmtNum(lm.lr, 6)}` : "";
  const gpu = lm.gpu_mem != null ? ` · GPU ${lm.gpu_mem}G` : "";
  let head;
  if (st.task === "pretrain") {
    head = st.status === "stopping" ? "停止中" : st.running ? "运行中" : "已结束";
  } else if (st.task) {
    head = `后训练 ${st.task}`;
  } else {
    head = "";
  }
  if (!head) {
    el.textContent = "";
    return;
  }
  el.title = st.run_id;
  el.innerHTML = `<b>${esc(head)} ${esc(sid)}</b>${step}${loss}${lr}${gpu}`;
}

/* ── YAML 配置编辑器（内置只读 / my_configs 可写；保存前复用后端 Pydantic 校验）── */
let cfgCur = null; // {path, writable}

async function openCfgModal() {
  const entries = await api("/api/configs").catch(() => []);
  const opts = entries
    .map((e) => `<option value="${esc(e.path)}">${esc(e.name)}${e.builtin ? "（内置）" : "（我的）"}${e.writable ? " · 可写" : ""}</option>`)
    .join("");
  const box = openModal(`<div class="m-head"><b>预训练配置 YAML</b>
      <span class="chip" style="margin-left:6px">内置模板只读 · 保存前 Pydantic 校验</span>
      <span class="spacer"></span>
      <button type="button" class="btn ghost sm" data-close="modal-mask">✕</button></div>
    <div class="m-body">
      <div style="display:flex;gap:8px;align-items:center">
        <label style="white-space:nowrap">配置文件</label>
        <select id="cfg-sel" class="path-select">${opts}</select>
        <span id="cfg-writ" class="chip">—</span>
      </div>
      <textarea id="cfg-editor" spellcheck="false" placeholder="选择配置后点「打开」；改完「保存修改」即时校验"></textarea>
      <div id="cfg-msg" style="font-size:12px;min-height:16px"></div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <span style="font-size:12px;color:var(--dim)">另存为 my_configs/</span>
        <input id="cfg-name" type="text" style="width:230px" placeholder="my_nano_v2（.yaml 自动补全）" />
        <button id="cfg-copy" class="btn ghost sm" type="button">另存为我的配置</button>
        <span class="spacer"></span>
        <button id="cfg-save" class="btn sm" type="button" disabled>保存修改</button>
      </div>
    </div>`);
  cfgCur = null;
  const sel = box.querySelector("#cfg-sel");
  const ed = box.querySelector("#cfg-editor");
  const writ = box.querySelector("#cfg-writ");
  const save = box.querySelector("#cfg-save");
  const msg = box.querySelector("#cfg-msg");
  const setMsg = (html, ok) => {
    msg.innerHTML = html;
    msg.style.color = ok ? "var(--ok)" : "var(--err)";
  };
  async function openCfg(path) {
    if (!path) return;
    try {
      const c = await api(`/api/config?path=${encodeURIComponent(path)}`);
      cfgCur = { path, writable: !!c.writable };
      ed.value = c.content || "";
      ed.disabled = false;
      writ.textContent = c.writable ? "可写（保存前校验）" : "只读 — 另存为后修改";
      writ.className = "chip " + (c.writable ? "running" : "");
      save.disabled = !c.writable;
      setMsg(`已打开 <b>${esc(path)}</b>，共 ${(c.content || "").split("\n").length} 行`, true);
    } catch (err) {
      setMsg(esc(err.message), false);
    }
  }
  sel.addEventListener("change", () => openCfg(sel.value));
  sel.dispatchEvent(new Event("change"));
  ed.addEventListener("input", () => {
    if (cfgCur) save.disabled = false;
    setMsg("", true);
  });
  save.addEventListener("click", async () => {
    if (!cfgCur) return;
    save.disabled = true;
    try {
      await api("/api/config", {
        method: "POST",
        body: JSON.stringify({ path: cfgCur.path, content: ed.value }),
      });
      setMsg("保存成功（已通过配置校验）", true);
    } catch (err) {
      save.disabled = false;
      setMsg(`校验/保存失败：<br/>${esc(err.message)}`, false);
    }
  });
  box.querySelector("#cfg-copy").addEventListener("click", async () => {
    const name = (box.querySelector("#cfg-name").value || "").trim();
    if (!cfgCur) return;
    if (!name) {
      setMsg("请填写另存的文件名", false);
      return;
    }
    try {
      const r = await api("/api/config/copy", {
        method: "POST",
        body: JSON.stringify({ source: cfgCur.path, dest_name: name }),
      });
      // 列表里补一项并切过去
      const opt = document.createElement("option");
      opt.value = r.path;
      opt.textContent = `${name}（我的） · 可写`;
      sel.appendChild(opt);
      sel.value = r.path;
      setMsg(`已另存为 <b>${esc(r.path)}</b> — 可直接编辑保存`, true);
      await openCfg(r.path);
    } catch (err) {
      setMsg(esc(err.message), false);
    }
  });
}

/* ── boot ── */
function initPretrain() {
  $("#pt-start-btn").addEventListener("click", () => trainer.openStartModal(["pretrain"]));
  $("#pt-cfg-btn").addEventListener("click", openCfgModal);
  trainer.on("status", onStatus);
  trainer.on("metric", onMetric);
  trainer.on("exit", onExit);
  loadRuns(true);
}

document.addEventListener("DOMContentLoaded", initPretrain);
})();

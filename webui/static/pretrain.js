/* ① 预训练 tab：loss/lr 曲线（当前 run 主色实时 + 历史勾选灰色叠加；
   loss 图叠加训练内嵌周期验证的 val_loss 稀疏折线，裸 CE 口径，详见 README
   「训练与验证口径」）、run 列表（点击主展示 / 无任务时可重放日志）、
   YAML 配置编辑器。
   指标数据源 = trainer 的 metric 事件（后端 tracker SQLite，panel 唯一写入方）。 */
"use strict";

/* 作用域隔离: 本文件与 posttrain.js 存在同名顶层函数(onStatus/onMetric/onExit/
   loadRuns/renderRuns/renderStatusLine/ensureHist), 后加载者会覆盖全局名, 导致
   本 tab 注册的回调与内部调用全指向对方实现。包 IIFE 使闭包内自解析, 互不干扰。 */
(function () {
const pt = {
  logY: false,    // B6：loss 图对数纵轴（默认关，loss 跨量级时手动开）
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
  // 仅 running 时才算 live: 训练结束后 run_id 仍留在 status 里, 不检查会把
  // 已完成的 run 继续当 live —— 拦掉 setMain 的日志回放、列表误挂 LIVE 徽章
  return st.task === "pretrain" && st.running ? st.run_id : null;
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
  if (st.task === "pretrain" && st.run_id && !st.running && pt.live) {
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
  // 片一：失败时**不清空** pt.runs（stale-but-usable）—— 一个几十小时的项目，
  // 「上次的数据 + 一条警告」远好过「空白 + 错误」。
  const hadData = pt.runs.length > 0;
  try {
    const runs = await api("/api/train/runs");
    pt.runs = Array.isArray(runs) ? runs : [];
    pt.runsErr = "";
    clearFetchFail("pt-runs");
  } catch (err) {
    pt.runsErr = (err && err.message) || "请求失败";
    noteFetchFail("pt-runs", hadData);
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

/* run 行的「动态部分」单独算出来 —— 用于集合未变时原地刷新（D5）。
   注意 dyn 里含 statusChip 产出的 HTML（带引号），所以它只能存在 dataset 里，
   不能写进 HTML 属性。 */
function runDyn(r) {
  const isMain = r.id === pt.mainId;
  const live = r.id === livePretrainId();
  const chip = statusChip(r);
  return {
    isMain,
    live,
    dyn: `${isMain}|${live}|${chip}`,
    name: `${esc(r.id)}${live ? ' <span class="chip running" style="padding:0 8px">LIVE</span>' : ""}`,
    // 主曲线不再在副标题里写文字 —— 状态由 .list-item.on 的描边+底色表达
    sub: `${fmtTime(r.created_at)} · ${chip}`,
    title: `${esc(r.id)} · 点击设为主曲线${trainer.run && trainer.run.running ? "" : "（无任务运行中，可重放日志）"}`,
  };
}

function bindDelRun(b) {
  b.addEventListener("click", async (e) => {
    e.stopPropagation();
    const id = b.dataset.run;
    // E6：原生 confirm 换成样式化弹窗
    const okDel = await confirmDialog({
      title: "删除实验？",
      okText: "删除",
      body: `<p style="margin:0">实验 <b>${esc(id)}</b></p>
        <p style="margin:0;color:var(--dim);font-size:13px">将同时删除其指标记录与日志文件，<b>不可恢复</b>。</p>`,
    });
    if (!okDel) return;
    try {
      await api(`/api/train/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
      pt.hist.delete(id);
      pt.checked = pt.checked.filter((x) => x !== id);
      loadRuns(true);
    } catch (err) {
      toast("删除失败：" + err.message, "err");
    }
  });
}

/* H12：A/B 角标。checked[0] → A（图里蓝色虚线），checked[1] → B（紫色虚线）。
   与图表里的 cmp 序号同源 —— 所以角标不是装饰，是图例的另一半。 */
function cmpBadgeHtml(checked, id) {
  const i = checked.indexOf(id);
  if (i === 0) return '<span class="ab-badge a" title="对比 A —— 图中蓝色虚线">A</span>';
  if (i === 1) return '<span class="ab-badge b" title="对比 B —— 图中紫色虚线">B</span>';
  return "";
}

/* 勾选变化时刷角标。注意 runDyn() 的签名里不含 checked，
   所以角标不会被原地更新的常规路径带上 —— 必须单独同步。 */
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

/* P2：操作回到对象上 —— 运行中显示「停止」，结束后显示「删除」。
   用同一份 HTML 生成器同时服务首渲与原地更新，避免两处漂移（批次 3 的老问题）。 */
/* H17：行内只留「删除」—— 停止已收归任务列表标题栏（紧跟启动）。
   同一动作不开两个入口：停止是破坏性操作，入口收在一处才配得上 E7 的二次确认。
   运行中的行因此没有行内按钮（live → 空串）。 */
function actionBtnHtml(r, live) {
  if (live) return "";
  return `<button class="del-run" type="button" data-run="${esc(r.id)}" title="删除该 run 及其指标、日志文件（不可恢复）">✕</button>`;
}

/* 绑定行内操作按钮。加 data-bound 守卫 —— 原地更新可能在同一元素上调用多次。 */
function bindRowActions(scope) {
  const del = scope.querySelector(".del-run");
  if (del && !del.dataset.bound) bindDelRun(del);
}

function bindRunRows(box) {
  box.querySelectorAll("input[data-run]").forEach((cb) => {
    cb.addEventListener("change", () => toggleCompare(cb.dataset.run, cb.checked));
  });
  box.querySelectorAll(".list-item").forEach((li) => {
    const act = () => setMain(li.dataset.id);
    li.addEventListener("click", (e) => {
      if (e.target.closest("input, .del-run")) return;
      act();
    });
    makeActivatable(li, act, `设为主曲线 ${li.dataset.id}`); // E2：键盘可达
    bindRowActions(li);
  });
}

function renderRuns() {
  const box = $("#pt-runs");
  if (!box) return;
  const mine = ptRunList();
  // 片一：警告条独立于列表容器 —— 并进 box 会破坏 D5 的 sig 原地更新
  const warnEl = $("#pt-runs-warn");
  if (warnEl) {
    const show = mine.length > 0 && failShown("pt-runs") && !isLinkDown();
    warnEl.hidden = !show;
    if (show) warnEl.textContent = `⚠ 任务列表读取失败（当前显示的是上次数据）· ${pt.runsErr}`;
  }
  if (!mine.length) {
    box.dataset.sig = "";
    // 片一：失败态**优先**于空态 —— 顺序反了就把「故障」说成了「没有数据」
    if (isLinkDown()) {
      box.innerHTML = '<div class="hint sm">与后端失联 —— 恢复后自动加载</div>';
    } else if (failShown("pt-runs")) {
      box.innerHTML = failHtml(`无法读取任务列表 · ${pt.runsErr}`);
      bindRetry(box, () => loadRuns(true));
    } else {
      box.innerHTML = '<div class="hint sm">点击「启动训练」开启预训练</div>';
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
    <input type="checkbox" class="check" data-run="${esc(r.id)}" ${pt.checked.includes(r.id) ? "checked" : ""} title="勾选与主曲线 A/B 对比（最多 2 条）" />
    <div style="flex:1;min-width:0">
      <div class="name" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${d.name}</div>
      <div class="sub">${d.sub}</div>
    </div>
    ${cmpBadgeHtml(pt.checked, r.id)}
    ${actionBtnHtml(r, d.live)}
  </div>`;
      })
      .join("");
    bindRunRows(box);
    return;
  }
  // D5：run 集合没变 → 只刷动态部分。零结构变更，所以鼠标悬停、键盘焦点、
  // 复选框焦点、正在进行的文本选择都不会被 2s 轮询打断。
  for (const r of mine) {
    const li = box.querySelector(`.list-item[data-id="${CSS.escape(r.id)}"]`);
    if (!li) continue;
    const d = runDyn(r);
    const wantChecked = pt.checked.includes(r.id);
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
    syncCmpBadge(li, pt.checked, r.id); // H12：勾选变了角标要跟着变
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

function toggleCompare(id, on) {
  if (on) {
    if (pt.checked.length >= 2) {
      toast("对比最多勾选 2 个历史 run（主曲线之外）", "err");
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
  // 片一：失败时**不写入空对象** —— 原先 set(id, {}) 会让曲线永久空白，
  // 而且 has(id) 为真 → 下次进来直接早退，连重试都拉不动。改为仅在成功时缓存。
  try {
    const d = await api(`/api/train/metrics?run_id=${encodeURIComponent(id)}`);
    pt.hist.set(id, d.series || {});
    pt.histErr = "";
    clearFetchFail("pt-hist:" + id);
  } catch (err) {
    pt.histErr = (err && err.message) || "请求失败";
    noteFetchFail("pt-hist:" + id, false); // 无旧数据可留 → 立即显示
  }
  setChartErr(
    ["pt-loss-err", "pt-lr-err"],
    failShown("pt-hist:" + id) && !isLinkDown()
      ? `无法读取该任务的曲线数据 · ${pt.histErr}`
      : "",
    () => ensureHist(id),
  );
  drawCharts();
}

function setMain(id) {
  if (trainer.run && trainer.run.running) {
    const st = trainer.run;
    if (st.task === "pretrain" && st.run_id === id) return;
    toast("训练运行中，主曲线跟随当前任务 — 可在停止后回看历史曲线与日志");
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

/* ── 片三：导出曲线 CSV ──
   两条硬约束：
   ① 数据源必须是**原始 series**（mainSeriesData() / pt.hist），不能用 chart 内部数据
      —— chart.js 的 decimate(…, 2000) 是为画图抽稀的，直接复用会**静默丢掉 90% 的点**。
   ② 对比 run 可能还没拉过 series（只有画图时才会 ensureHist）→ 导出前补齐。
   范围（R6）：主曲线 + 勾选的 A/B；列名前缀与左栏角标对齐（main / A / B）。 */
async function exportCsv() {
  const items = [];
  if (pt.mainId) items.push({ tag: "main", id: pt.mainId });
  pt.checked.forEach((id, i) => items.push({ tag: i === 0 ? "A" : "B", id }));
  if (!items.length) {
    toast("没有可导出的曲线 —— 先在左栏选一个 run", "err");
    return;
  }
  const btn = $("#pt-csv");
  if (btn) btn.disabled = true;
  try {
    await Promise.all(items.map((r) => ensureHist(r.id))); // 已缓存的是空操作
    const named = items.map((r) => ({
      name: r.tag,
      series: r.id === pt.mainId && pt.series ? pt.series : pt.hist.get(r.id),
    }));
    const csv = seriesToCsv(named);
    if (!csv) {
      toast("该 run 还没有曲线数据（可能还没开始记录）", "err");
      return;
    }
    downloadText(csv, `${pt.mainId}.csv`, "text/csv;charset=utf-8");
    toast(`已导出 ${csv.trim().split("\r\n").length - 1} 行 · ${items.length} 条曲线`, "ok");
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* ── 曲线 ── */
function mainSeriesData() {
  // live / 刚结束的主曲线：series 事件实时喂；历史 run：拉取缓存
  if (pt.series) return pt.series;
  return pt.hist.get(pt.mainId) || null;
}

/* P1：lr 副图的 WSD 三阶段。返回 { phases, bands } —— 分界点 + 底色区间。
   原先只产出两条分界线，底色说明不了「这一段是什么」。 */
function lrPhases() {
  const r = pt.runs.find((x) => x.id === pt.mainId);
  const sum = r && r.config && r.config.yaml_summary;
  const lrCfg = sum && sum.lr;
  const st = trainer.run || {};
  const total = st.run_id === pt.mainId ? st.total_steps : null;
  if (!lrCfg || !total || lrCfg.type !== "wsd") return { phases: [], bands: [] };
  const w = parseFloat(lrCfg.warmup_ratio);
  if (!isFinite(w) || w <= 0) return { phases: [], bands: [] };
  const s = parseFloat(lrCfg.stable_ratio);
  const wEnd = Math.round(total * w);
  const hasStable = isFinite(s) && s > 0;
  const sEnd = hasStable ? Math.round(total * (w + s)) : total;
  const C = LineChart.palette();
  const phases = [{ x: wEnd, label: "warmup 结束" }];
  if (hasStable && sEnd < total) phases.push({ x: sEnd, label: "decay 开始" });
  // P1：三阶段底色。分界线说明「在哪切」，底色说明「这一段是什么」——§6.1 要的是两者都有。
  return {
    phases,
    bands: [
      { from: 0, to: wEnd, label: "warmup", color: C.bandWarmup },
      { from: wEnd, to: sEnd, label: "stable", color: C.bandStable },
      { from: sEnd, to: total, label: "decay", color: C.bandDecay },
    ],
  };
}

let drawTimer = null;
function drawCharts() {
  clearTimeout(drawTimer);
  drawTimer = setTimeout(() => {
    const C = LineChart.palette(); // B2：色板来自 :root，不再本地写死
    const main = mainSeriesData();
    const loss = [], lr = [];
    if (main) {
      loss.push({ name: shortId(pt.mainId), color: C.loss, points: main.loss || [] });
      // 周期验证序列（稀疏: 每 eval_interval 步一点）叠在主 loss 图上;
      // 无验证的 run 没有该键, 不 push 空序列
      if ((main.val_loss || []).length) {
        loss.push({ name: shortId(pt.mainId) + " · val", color: C.val, points: main.val_loss || [] });
      }
      lr.push({ name: shortId(pt.mainId), color: C.lr, points: main.lr || [] });
    }
    // B4/Q3：对比系列按勾选顺序给 cmp=0/1 —— 让两条历史线用不同线型区分，
    // 而不是原先统一涂灰（叠两条后认不出哪条是哪条）
    pt.checked.forEach((id, i) => {
      const s = pt.hist.get(id);
      if (!s) return; // 注意：这里是 forEach 回调，不是 for 循环 —— 用 return 不用 continue
      loss.push({ name: shortId(id), color: C.loss, points: s.loss || [], cmp: i });
      if ((s.val_loss || []).length) {
        loss.push({ name: shortId(id) + " · val", color: C.val, points: s.val_loss || [], cmp: i });
      }
      lr.push({ name: shortId(id), color: C.lr, points: s.lr || [], cmp: i });
    });
    if (!pt.lossChart) {
      pt.lossChart = new LineChart($("#pt-loss-chart"));
      pt.lrChart = new LineChart($("#pt-lr-chart"));
    }
    // zeroY：纵轴从 0 起（与 lr 图一致；log y 打开时自动失效，见 yDomain）
    pt.lossChart.render({
      series: loss, yLabel: "loss", xLabel: "step", logY: pt.logY, zeroY: true,
    });
    const lp = lrPhases(); // P1：带三阶段底色
    pt.lrChart.render({ series: lr, phases: lp.phases, bands: lp.bands, yLabel: "lr", xLabel: "step", zeroY: true });
    renderMetrics();
    renderAb(); // D4
  }, 60);
}

/* D4：A/B 最终指标差异条。主曲线 vs 勾选的第一条；拿不到两条就整条隐藏。 */
function renderAb() {
  const seriesOf = (id) => (id === pt.mainId ? mainSeriesData() : pt.hist.get(id));
  if (!pt.mainId || !pt.checked.length) {
    renderAbDiff("#pt-abdiff", null, null);
    return;
  }
  const bId = pt.checked[0];
  renderAbDiff(
    "#pt-abdiff",
    { name: shortId(pt.mainId), series: seriesOf(pt.mainId) },
    { name: shortId(bId), series: seriesOf(bId) },
  );
}

/* ── D1/D2 指标条：live run 取 last_metric；否则取所选 run 的末端值
   （历史 run 没有起点时间 → ETA 留空，不编造）。──
   原先这些字段分散在 loss/lr 两个 chip 与日志标题行三处，现收成一条。 */
function renderMetrics() {
  const st = trainer.run || {};
  const lm = st.last_metric || {};
  const live = st.task === "pretrain" && st.running === true;
  const series = mainSeriesData();
  const tail = (k) => {
    const a = series && series[k];
    return a && a.length ? a[a.length - 1] : null;
  };
  let step, loss, lr, tok, vppl, eta = "";
  if (live) {
    step = lm.step;
    loss = lm.loss;
    lr = lm.lr;
    tok = lm.tok_per_sec;
    vppl = lm.val_ppl;
    eta = estimateEta(st);
  } else {
    const sl = tail("loss");
    const sk = tail("tok_per_sec");
    const sv = tail("val_ppl");
    const sr = tail("lr");
    step = sl && sl[0];
    loss = sl && sl[1];
    lr = sr && sr[1];
    tok = sk && sk[1];
    vppl = sv && sv[1];
  }
  setMetrics("#pt-metrics", {
    step: step != null ? step : "",
    loss: loss != null ? fmtNum(loss) : "",
    lr: lr != null ? fmtNum(lr, 6) : "",
    tok: fmtTok(tok),
    vppl: vppl != null ? fmtNum(vppl, 2) : "",
    eta,
  });
}

function shortId(id) {
  if (!id) return "";
  const s = String(id).split("_");
  return s.length >= 3 ? `${s[0]}_${s[s.length - 1]}` : id;
}

/* ── 日志标题行的 run 摘要：step/loss/lr/tok/val ppl/ETA 已在指标条（D1），
   这里只留「状态 + 短 run id」与验证状态提示；详情走 title 悬停 ── */
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
  let head;
  if (st.task === "pretrain") {
    head = st.status === "stopping" ? "停止中" : st.running ? "运行中" : "已结束";
  } else if (st.task) {
    head = `后训练 ${st.task}`;
  } else {
    el.textContent = "";
    return;
  }
  el.title = st.run_id;
  let detail = "";
  if (st.task === "pretrain") {
    const hasValData = !!(st.fields || {}).val_data;
    // val ppl 已移入指标条；此处只保留验证状态，详情挂 title 悬停
    if (lm.val_ppl != null) {
      // 验证步不能直接用 lm.step —— 它已被后续训练行覆盖成最新步
      const vstep = lm.val_step != null ? lm.val_step : lm.step != null ? lm.step : 0;
      el.title =
        st.run_id +
        ` — step ${vstep} 验证 · val loss ${fmtNum(lm.val_loss)} · val ppl ${fmtNum(lm.val_ppl, 2)}`;
    } else if (hasValData) {
      // 已启用验证但还没到首个 eval_interval: 过渡态, 别让行空白
      detail = " · 首验前";
      el.title = st.run_id + " — 已启用验证，未到首个验证步";
    } else {
      // 没填验证数据: 干净显示 head + id, 悬停说明如何启用
      el.title = st.run_id + " — 未启用验证（启动时填写「验证数据」）";
    }
  } else {
    const step = lm.step != null ? ` · ${lm.step} step` : "";
    const loss = lm.loss != null ? ` · loss ${fmtNum(lm.loss)}` : "";
    const lr = lm.lr != null ? ` · lr ${fmtNum(lm.lr, 6)}` : "";
    const gpu = lm.gpu_mem != null ? ` · GPU ${lm.gpu_mem}G` : "";
    detail = `${step}${loss}${lr}${gpu}`;
  }
  el.innerHTML = `<b>${esc(head)} ${esc(sid)}</b>${detail}`;
}

/* ── YAML 配置编辑器（内置只读 / manual/my_configs 可写；保存前复用后端 Pydantic 校验）── */
let cfgCur = null; // {path, writable}

async function openCfgModal() {
  const entries = await api("/api/configs").catch(() => []);
  const opts = entries
    .map((e) => `<option value="${esc(e.path)}">${esc(e.name)}${e.builtin ? "（内置）" : "（我的）"}${e.writable ? " · 可写" : ""}</option>`)
    .join("");
  const box = openModal(`<div class="m-head"><b>预训练配置 YAML</b>
      <span class="chip" style="margin-left:8px">内置模板只读 · 保存前 Pydantic 校验</span>
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
        <span style="font-size:12px;color:var(--dim)">另存为 manual/my_configs/</span>
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
/* B6：log y 按钮的可见状态。两个 tab 各写一份 —— 它们是互相独立的 IIFE，
   局部函数不共享（共享出口在 trainer 上）。 */
function applyLogYBtn() {
  const b = $("#pt-logy");
  if (!b) return;
  b.setAttribute("aria-pressed", pt.logY ? "true" : "false");
  b.classList.toggle("seg-on", pt.logY);
}

function initPretrain() {
  // 片三：导出曲线 CSV
  $("#pt-csv").addEventListener("click", exportCsv);
  $("#pt-start-btn").addEventListener("click", () => trainer.openStartModal(["pretrain"]));
  $("#pt-cfg-btn").addEventListener("click", openCfgModal);
  // B6：log y 开关。持久化 —— 习惯对数轴的人不该每次刷新都重按一次。
  try {
    pt.logY = localStorage.getItem("webui.logY") === "1";
  } catch (_) {
    pt.logY = false;
  }
  applyLogYBtn();
  $("#pt-logy").addEventListener("click", () => {
    pt.logY = !pt.logY;
    try {
      localStorage.setItem("webui.logY", pt.logY ? "1" : "0");
    } catch (_) {
      /* 隐私模式等写入失败：本次会话内仍然生效 */
    }
    applyLogYBtn();
    drawCharts();
  });
  trainer.on("status", onStatus);
  trainer.on("metric", onMetric);
  trainer.on("exit", onExit);
  loadRuns(true);
  // 切回本页时重绘曲线：tab 隐藏期间首渲的 canvas 位图为 0×0（无布局尺寸），
  // 切回后必须补一次 draw 才会按真实尺寸重建位图
  window.addEventListener("tabchange", (e) => {
    if (e.detail === "pretrain") drawCharts();
  });
}

document.addEventListener("DOMContentLoaded", initPretrain);
})();

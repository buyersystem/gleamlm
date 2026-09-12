/* GleamLM WebUI 公共工具：fetch 封装 / DOM 简写 / 外观设置（三 tab 共享）。 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function esc(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

async function api(path, opts = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  let data = null;
  try {
    data = await resp.json();
  } catch (_) {
    /* 非 JSON 响应（如流式端点误用），保持 null */
  }
  if (!resp.ok) {
    const detail = data && data.detail;
    let msg;
    if (typeof detail === "string") {
      msg = detail;
    } else if (detail && Array.isArray(detail.errors)) {
      msg = detail.errors.map((e) => `${e.loc}: ${e.msg}`).join("\n");
    } else {
      msg = JSON.stringify(detail || resp.statusText || `HTTP ${resp.status}`);
    }
    throw new Error(msg);
  }
  return data;
}

/* ── 外观设置（背景图 / 透明度，localStorage 持久化）── */
const BG_IMAGES = {
  none: "",
  night: "/images/gleamlm-title2.png",
  night2: "/images/luna_night2.png",
};

function currentLook() {
  let look = { bg: "none", alpha: 0 };
  try {
    look = { ...look, ...JSON.parse(localStorage.getItem("webui.look") || "{}") };
  } catch (_) {
    /* 损坏的存储按默认处理 */
  }
  return look;
}

function applyLook() {
  const look = currentLook();
  const url = BG_IMAGES[look.bg] || "";
  document.body.style.setProperty("--bg-image", url ? `url("${url}")` : "none");
  document.body.style.setProperty("--bg-alpha", String(look.alpha ?? 0));
  // 玻璃联动: 背景图开启且透明度 >0 时, 主体卡片同步半透明（alpha 越高越透）;
  // 无背景或 alpha=0 不加类, 视觉与旧版完全一致。
  const a = look.alpha ?? 0;
  const on = !!url && a > 0;
  document.body.classList.toggle("bg-on", on);
  if (on) document.body.style.setProperty("--glass-a", String(Math.max(0.3, 1 - a * 0.6)));
}

function saveLook(patch) {
  const look = { ...currentLook(), ...patch };
  localStorage.setItem("webui.look", JSON.stringify(look));
  applyLook();
}

function initAppearance() {
  applyLook();
  const bgSel = $("#look-bg");
  const alphaSel = $("#look-alpha");
  const alphaVal = $("#look-alpha-val");
  if (!bgSel) return;
  const look = currentLook();
  bgSel.value = look.bg in BG_IMAGES ? look.bg : "none";
  alphaSel.value = look.alpha;
  alphaVal.textContent = look.alpha;
  bgSel.addEventListener("change", () => saveLook({ bg: bgSel.value }));
  alphaSel.addEventListener("input", () => {
    alphaVal.textContent = alphaSel.value;
    saveLook({ alpha: parseFloat(alphaSel.value) });
  });
}

/* 时间格式化（run 列表等） */
function fmtTime(ts) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function fmtNum(v, digits = 4) {
  if (v == null || !isFinite(v)) return "-";
  return Number(v).toFixed(digits);
}

/* 科学计数法（SFT/DPO 的 lr：decay 后期可达 1e-7，.6f 会渲染成 0.000000）
   指数补零到两位，与后端 f"{lr:.2e}" 的观测格式一致 */
function fmtSci(v, digits = 2) {
  if (v == null || !isFinite(v)) return "-";
  return Number(v).toExponential(digits).replace(/e([+-])(\d)$/, "e$10$2");
}

/* ── toast：非阻塞反馈（GUI清单 E6/E9 的统一出口）──
   片二：第 4 参 actions 支持动作按钮（[{"label","onClick"}]），**缺省时行为与原先
   完全一致** —— 既有 30+ 处调用都不用改。加它是为了让「训练完成」能给下一步去向：
   原先只能报结果，用户得自己想起「接下来要去推理页」。
   带按钮的 toast **悬停暂停倒计时** —— 否则常在正要点击时消失。 */
function toast(text, kind = "info", ms = 4200, actions = null) {
  let host = $("#toast-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "toast-host";
    host.className = "toast-host";
    document.body.appendChild(host);
  }
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.setAttribute("role", "status");
  const kill = () => el.remove();
  let timer = 0;
  if (actions && actions.length) {
    el.classList.add("has-acts");
    const msg = document.createElement("span");
    msg.textContent = text;
    el.appendChild(msg);
    const acts = document.createElement("span");
    acts.className = "toast-acts";
    for (const a of actions) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn ghost sm";
      b.textContent = a.label;
      b.addEventListener("click", (e) => {
        // 不 stopPropagation 的话会冒泡到 el 的「点击即关」——
        // 先关再执行，看起来像按钮没响应。
        e.stopPropagation();
        clearTimeout(timer);
        kill();
        if (a.onClick) a.onClick();
      });
      acts.appendChild(b);
    }
    el.appendChild(acts);
    el.addEventListener("mouseenter", () => clearTimeout(timer));
    el.addEventListener("mouseleave", () => {
      timer = setTimeout(kill, 2500);
    });
  } else {
    el.textContent = text;
  }
  host.appendChild(el);
  timer = setTimeout(kill, ms);
  el.addEventListener("click", () => {
    clearTimeout(timer);
    kill();
  });
  return el;
}

/* ── 指标条（D1/D2）：一次写入一组「键 → 文本」 ──
   data-m 值为空时显示占位破折号，避免出现空白格。 */
function setMetrics(rootSel, values) {
  const root = $(rootSel);
  if (!root) return;
  for (const k of Object.keys(values)) {
    const el = root.querySelector(`[data-m="${k}"]`);
    if (!el) continue;
    const raw = values[k];
    const txt = raw == null || raw === "" ? "—" : String(raw);
    if (el.textContent !== txt) el.textContent = txt;
  }
}

function fmtTok(v) {
  if (v == null || !isFinite(v) || v <= 0) return "";
  return v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v));
}

function fmtDur(sec) {
  if (!isFinite(sec) || sec <= 0) return "";
  const h = Math.floor(sec / 3600);
  const m = Math.round((sec % 3600) / 60);
  if (h >= 48) return `${Math.round(h / 24)}d`;
  if (h >= 24) return `${Math.floor(h / 24)}d${h % 24}h`;
  if (h > 0) return `${h}h${String(m).padStart(2, "0")}m`;
  return `${Math.max(1, m)}m`;
}

function fmtClock(ms) {
  const d = new Date(ms);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/* ── F3：后端失联提示 ──
   本项目的训练以「几十小时」计，期间 WebUI 进程可能崩掉、机器可能休眠。
   原先两处轮询的 catch 都是静默的 —— 界面会永久停在最后一次成功的数据上，
   看起来像「训练还在跑」。
   阈值取 3 次：单次请求超时（GC 停顿、请求排队）不该弹警，那会变成噪音。 */
let _pollFail = 0;
let _linkDown = false;
let _lastOkClock = "";

function notePoll(ok) {
  const el = $("#link-warn");
  const dot = $("#link-dot"); // P3：header 的连接状态徽章
  const txt = $("#link-text");
  if (ok) {
    _pollFail = 0;
    _lastOkClock = fmtClock(Date.now());
    if (dot) dot.className = "dot run";
    if (txt) txt.textContent = "正常";
    if (_linkDown) {
      _linkDown = false;
      if (el) el.innerHTML = "";
    }
    return;
  }
  _pollFail++;
  if (_pollFail < 3 || _linkDown) return;
  _linkDown = true;
  if (dot) dot.className = "dot err";
  if (txt) txt.textContent = "已断开";
  if (el) {
    el.innerHTML =
      `<div class="warnbar err">⚠ 与后端失联（最后更新 ${_lastOkClock || "未知"}）` +
      ` —— 界面数据可能已过期，连接恢复后自动消失</div>`;
  }
}

/* ── 片一：局部失败态（把「取数失败」与「没有数据」分开）──
   原先各处的 `catch (_) { x = [] }` 把请求失败渲染成了空态 —— 后端还活着、
   只是某个端点失败时，header 的失联徽章**不会亮**，用户只能猜（会以为记录丢了）。
   三条规则：
   ① 失败态优先于空态 —— 顺序反了就又回到「把故障说成没数据」
   ② 失败不覆盖已有数据（stale-but-usable）—— 只置标记，不清数据
   ③ 阈值分两档：首次加载立即显示；已有数据时连续 FAIL_N 次才提示（避免轮询闪烁） */
const FAIL_N = 3; // 与 notePoll 的失联阈值同档（2s 轮询 ≈ 6s）
const _fetchFail = new Map();

/* 记一次失败，返回「是否应当展示失败态」。
   hadData=true 表示容器已有旧数据可留 → 走节流；首次加载（无数据可留）立即显示。 */
function noteFetchFail(key, hadData) {
  const st = _fetchFail.get(key) || { n: 0, shown: false };
  st.n += 1;
  st.shown = !hadData || st.n >= FAIL_N;
  _fetchFail.set(key, st);
  return st.shown;
}

/* 清失败标记；返回「之前是否显示着失败态」，调用方据此决定要不要重绘。 */
function clearFetchFail(key) {
  const st = _fetchFail.get(key);
  if (!st) return false;
  _fetchFail.delete(key);
  return st.shown;
}

function failShown(key) {
  const st = _fetchFail.get(key);
  return !!(st && st.shown);
}

/* 局部失败态在**全局失联**时降级为不显示：失联条已经表达了「整个后端不通」，
   再说「这个列表没读到」是同一件事的后果 —— 两条警告并存只稀释信号。
   （这个交叠点是读代码时发现的：后端进程整个挂掉时两者会同时亮。） */
function isLinkDown() {
  return _linkDown;
}

/* 失败态 HTML：位置与 .hint 空态一致，多一行原因 + 重试按钮。
   文案必须指名**哪个数据源**失败 —— 笼统的「加载失败」等于没说。 */
function failHtml(msg) {
  return (
    `<div class="hint warn">${esc(msg)}` +
    `<div class="hint-act"><button class="btn ghost sm" type="button" data-retry>重试</button></div></div>`
  );
}

/* 渲染后绑一次重试（节点是新造的，所以每次重建都要重绑）。 */
function bindRetry(root, fn) {
  const b = root && root.querySelector("[data-retry]");
  if (b) b.addEventListener("click", fn);
}

/* 曲线失败的叠加提示：.chart-box 已是 relative，所以直接 inset 铺满。
   两张图共用一份数据源 → 同一句话同时控制（传 id 数组）。
   dataset 比对避免重复重建（重建会丢焦点）。 */
function setChartErr(ids, msg, retry) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (!el) continue;
    if (el.dataset.msg === (msg || "")) continue;
    el.dataset.msg = msg || "";
    el.hidden = !msg;
    if (!msg) {
      el.innerHTML = "";
      continue;
    }
    el.innerHTML = `${esc(msg)}<button class="btn ghost sm" type="button" data-retry>重试</button>`;
    const b = el.querySelector("[data-retry]");
    if (b && retry) b.addEventListener("click", retry);
  }
}

/* ── 片三：曲线导出 CSV（纯函数，可单测）──
   R5 列布局 B：x 轴**并集** + 缺失填空。
   为什么不能直接横向拼：同一 run 里各指标的采样步长不同
   （loss 每 50 步 / val_loss 每 eval_interval / margin 每 global_step）——
   它们的 x 不重合。
   R7 精度：值用 String(v) 不四舍五入。日志里显示 4 位是给人看的，
   库里存的是原始浮点 —— 导出若也截断，等于「导出后精度低于数据源」。
   named: [{ name: "main"|"A"|"B", series: {loss:[[x,y],...], ...} }, ...]
   单条曲线时列名带不带前缀？不带（`loss` 比 `main_loss` 好读）；
   多条时必须带，否则两个 run 的 loss 列名重名。 */
function seriesToCsv(named) {
  const list = (named || []).filter((r) => r && r.series && typeof r.series === "object");
  if (!list.length) return "";
  const multi = list.length > 1;
  const keys = [];
  for (const r of list) {
    for (const k of Object.keys(r.series)) if (!keys.includes(k)) keys.push(k);
  }
  if (!keys.length) return "";
  const xs = new Set();
  const maps = list.map((r) => {
    const m = {};
    for (const k of keys) {
      const mp = new Map();
      for (const p of r.series[k] || []) {
        if (!Array.isArray(p) || p.length < 2) continue;
        if (p[1] == null || !isFinite(p[1])) continue;
        mp.set(p[0], p[1]);
        xs.add(p[0]);
      }
      m[k] = mp;
    }
    return m;
  });
  const sx = [...xs].sort((a, b) => a - b);
  if (!sx.length) return "";
  const head = ["step"];
  for (const r of list) for (const k of keys) head.push(multi ? r.name + "_" + k : k);
  const lines = [head.join(",")];
  for (const x of sx) {
    const row = [String(x)];
    for (const m of maps) {
      for (const k of keys) {
        const v = m[k].get(x);
        row.push(v == null ? "" : String(v));
      }
    }
    lines.push(row.join(","));
  }
  return lines.join("\r\n") + "\r\n";
}

/* 片三：触发一次浏览器下载。抽成公共的，避免第三份 Blob 样板。 */
function downloadText(text, filename, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime || "text/plain;charset=utf-8" }));
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* ETA（D2）：由 started_at + total_steps + 当前 step 估平均步速外推。
   前 5% 不显示 —— 启动/编译阶段样本太少，估出来只会误导。
   （用「自起点均速」而非滑动窗口：无需额外状态，且长跑下更稳。） */
function estimateEta(st) {
  const lm = (st && st.last_metric) || {};
  const step = Number(lm.step);
  const total = Number(st && st.total_steps);
  if (!step || !total || total <= step) return "";
  if (step / total < 0.05) return "";
  if (!st.started_at) return "";
  const t0 = Date.parse(String(st.started_at).replace(" ", "T"));
  if (!isFinite(t0)) return "";
  const elapsed = (Date.now() - t0) / 1000;
  if (elapsed <= 0) return "";
  return fmtDur((total - step) * (elapsed / step));
}

/* ── modal 通用开关 ── */
let _modalPrevFocus = null;
let _dialogKeyBound = false;

/* E4：焦点陷阱。两个弹窗容器（静态 look-mask + 动态 modal-mask）共用一套按键处理 ——
   原先 Esc 只关 modal-mask，look-mask 要手动点 ✕；且 Tab 会跑到遮罩后面的页面上。 */
const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]),' +
  ' textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/* 当前打开的弹窗根节点；无则 null。 */
function openDialogRoot() {
  for (const id of ["modal-mask", "look-mask"]) {
    const mask = document.getElementById(id);
    if (!mask || !mask.classList.contains("show")) continue;
    const root = mask.querySelector(".modal");
    if (root) return root;
  }
  return null;
}

function restoreFocus() {
  if (_modalPrevFocus && _modalPrevFocus.isConnected) _modalPrevFocus.focus();
  _modalPrevFocus = null;
}

/* Tab / Shift+Tab 在弹窗内循环。offsetParent 为 null 的（display:none）不参与。 */
function trapTab(e, root) {
  const items = $$(FOCUSABLE, root).filter(
    (el) => el.offsetParent !== null || el === document.activeElement
  );
  if (!items.length) {
    e.preventDefault();
    return;
  }
  const first = items[0];
  const last = items[items.length - 1];
  const cur = document.activeElement;
  if (e.shiftKey) {
    if (cur === first || !root.contains(cur)) {
      e.preventDefault();
      last.focus();
    }
  } else if (cur === last || !root.contains(cur)) {
    e.preventDefault();
    first.focus();
  }
}

function bindDialogKeys() {
  if (_dialogKeyBound) return;
  _dialogKeyBound = true;
  document.addEventListener("keydown", (e) => {
    const root = openDialogRoot();
    if (!root) return;
    if (e.key === "Escape") {
      const x = root.querySelector("[data-x], [data-close]");
      if (x) x.click();
      else {
        root.closest(".modal-mask").classList.remove("show");
        restoreFocus();
      }
      return;
    }
    if (e.key === "Tab") trapTab(e, root);
  });
}

function openModal(html) {
  const box = $("#modal-box");
  box.innerHTML = html;
  $("#modal-mask").classList.add("show");
  bindClose(box);
  // E4：记住来源焦点 → 焦点移入弹窗 → 关闭归还；Esc 关闭；Tab 不逃逸到遮罩后面。
  _modalPrevFocus = document.activeElement;
  const first = box.querySelector(
    "input:not([type=hidden]), select, textarea, button:not([data-close])"
  );
  if (first) first.focus();
  bindDialogKeys();
  return box;
}

function closeModal() {
  $("#modal-mask").classList.remove("show");
  restoreFocus();
}

function bindClose(root) {
  $$("[data-close]", root).forEach((b) =>
    b.addEventListener("click", () => {
      const mask = document.getElementById(b.dataset.close);
      if (mask) mask.classList.remove("show");
      restoreFocus(); // E4：✕ 与 Esc 走同一条归还路径（原先点 ✕ 后焦点丢失）
    })
  );
}

/* ── 确认弹窗：替代 window.confirm（E6/E7）──
   body 允许传 HTML（调用方负责只放自己生成的内容，不要塞日志/路径原文）。 */
function confirmDialog({ title = "确认", body = "", okText = "确认", danger = true } = {}) {
  return new Promise((resolve) => {
    const box = openModal(`<div class="m-head"><b>${esc(title)}</b>
        <span class="spacer"></span>
        <button type="button" class="btn ghost sm" data-x>✕</button></div>
      <div class="m-body">${body}</div>
      <div class="m-foot">
        <button type="button" class="btn ghost" data-x>取消</button>
        <button type="button" class="btn${danger ? " danger" : ""}" data-ok>${esc(okText)}</button>
      </div>`);
    let settled = false;
    const done = (v) => {
      if (settled) return;
      settled = true;
      closeModal();
      resolve(v);
    };
    box.querySelectorAll("[data-x]").forEach((b) =>
      b.addEventListener("click", () => done(false))
    );
    const ok = box.querySelector("[data-ok]");
    ok.addEventListener("click", () => done(true));
    ok.focus();
    // Esc / 关闭按钮走 done(false)：监听 show 类被摘掉
    const watch = new MutationObserver(() => {
      if (!$("#modal-mask").classList.contains("show")) {
        watch.disconnect();
        if (!settled) {
          settled = true;
          resolve(false);
          if (_modalPrevFocus && _modalPrevFocus.isConnected) _modalPrevFocus.focus();
          _modalPrevFocus = null;
        }
      }
    });
    watch.observe($("#modal-mask"), { attributes: true, attributeFilter: ["class"] });
  });
}

/* ── E2：让「可点击的 div」也支持键盘（Enter / Space 激活）──
   只在事件源就是该元素本身时响应 —— 否则内部按钮/复选框的按键会冒泡上来二次触发。
   （role=button 内嵌可聚焦子元素不是理想结构，但对「整行可点 + 行内还有按钮」
   的列表，这是代价最小且键盘真正可达的做法。） */
function makeActivatable(el, onActivate, label) {
  el.setAttribute("role", "button");
  el.setAttribute("tabindex", "0");
  if (label) el.setAttribute("aria-label", label);
  el.addEventListener("keydown", (e) => {
    if (e.target !== el) return;
    if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
      e.preventDefault(); // Space 默认会滚动页面
      onActivate();
    }
  });
}

/* ── E6：必填校验的字段级反馈（替代 alert）──
   弹窗只说了「缺哪个参数」，却不告诉你它在表单哪个位置。
   标红 + 聚焦 + 平滑滚到视野内，启动表单字段多时才找得到。 */
function showFieldError(root, fname, msg) {
  const el = root && root.querySelector(`[data-fname="${fname}"]`);
  if (el) {
    el.setAttribute("data-err", "1");
    el.focus();
    if (el.scrollIntoView) el.scrollIntoView({ block: "center" });
    el.addEventListener("input", () => el.removeAttribute("data-err"), { once: true });
  }
  toast(msg, "err");
  return el;
}

function lastPoint(series, key) {
  const arr = series && series[key];
  return arr && arr.length ? arr[arr.length - 1] : null;
}

/* ── D4 + H4：A/B 最终指标差异条 ──
   每个指标一片「<指标> Δ」；方向按指标语义判定（loss/kl 越低越好，margin/acc/reward 越高越好）。
   A/B 的身份由左栏 run 行的角标承担（H12），这里只给差值 —— 与提案图一致；
   两个端点的绝对值放进片子的 title，悬停仍可查（信息不丢）。
   某指标在任一侧缺失就跳过它；全缺则整条隐藏 —— 不编造。 */
const AB_LOWER_BETTER = { loss: true, kl: true, ppl: true };
function renderAbDiff(rootSel, a, b, metrics) {
  const el = $(rootSel);
  if (!el) return;
  const list = metrics && metrics.length ? metrics : ["loss"];
  const cells = [];
  for (const k of list) {
    const A = a && lastPoint(a.series, k);
    const B = b && lastPoint(b.series, k);
    if (!A || !B) continue;
    const d = B[1] - A[1]; // B 相对 A
    const flat = Math.abs(d) < 1e-9;
    const better = AB_LOWER_BETTER[k] ? d < 0 : d > 0;
    const cls = flat ? "" : better ? "better" : "worse";
    const mark = flat ? "" : d < 0 ? " ▼" : " ▲";
    cells.push(
      `<div class="c" title="${esc(k)}: ${esc(a.name)} ${fmtNum(A[1])} → ` +
        `${esc(b.name)} ${fmtNum(B[1])}">` +
        `<span class="k">${esc(k)} Δ</span>` +
        `<span class="v ${cls}">${d >= 0 ? "+" : ""}${d.toFixed(4)}${mark}</span></div>`
    );
  }
  if (!cells.length) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = cells.join("");
}

/* ── 页面壳：tab 切换 / 外观 / header 状态轮询 ── */
function initShell() {
  initAppearance();
  bindDialogKeys();
  // 静态弹窗（外观 look-mask 等）的 data-close ✕ 需在此绑定；动态注入的由 openModal 绑定
  bindClose(document);
  $("#look-btn").addEventListener("click", () => {
    _modalPrevFocus = document.activeElement; // E4：静态弹窗也要记住来源焦点
    $("#look-mask").classList.add("show");
    const first = $("#look-mask").querySelector("select, input, button");
    if (first) first.focus();
  });
  // tab 切换（E5：同步 aria-selected 与 roving tabindex；F5：写入 hash）
  $$("#tabs .tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  // E5：tablist 的左右方向键切换 —— roving tabindex 的标准交互
  $("#tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const vis = $$("#tabs .tab").filter((t) => t.style.display !== "none");
    const i = vis.indexOf(document.activeElement);
    if (i < 0) return;
    e.preventDefault();
    const next = vis[(i + (e.key === "ArrowRight" ? 1 : -1) + vis.length) % vis.length];
    next.focus();
    switchTab(next.dataset.tab);
  });
  window.switchTab = switchTab;

  function switchTab(name, writeHash = true) {
    const btn = $('#tabs .tab[data-tab="' + name + '"]');
    // 能力列表（--no-train 等）会把不可用的 tab 置为 display:none —— 不切过去
    if (!btn || btn.style.display === "none") return;
    $$("#tabs .tab").forEach((t) => {
      const on = t === btn;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
    });
    $$(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "pane-" + name));
    if (writeHash && location.hash.slice(1) !== name) location.hash = name;
    window.dispatchEvent(new CustomEvent("tabchange", { detail: name }));
  }

  // F5：深链 —— 刷新后停在原 tab；浏览器前进/后退也能切
  window.addEventListener("hashchange", () => {
    const n = location.hash.slice(1);
    if (n) switchTab(n, false);
  });
  const initialTab = location.hash.slice(1);
  if (initialTab) switchTab(initialTab, false);

  // 警告条内的操作按钮（如「卸载」）走事件委托：warn-line 的 innerHTML 按需重建，
  // 直接绑按钮会随重建丢失。按钮的 data-unload 携带模型身份 —— 卸载的就是
  // 警告里点名的那个模型。unloadModel 定义在 chat.js（defer 顺序在 util.js 之后，
  // 但点击发生在脚本全部加载后），typeof 守卫兼容裁掉 chat.js 的部署。
  $("#warn-line").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-unload]");
    if (!b) return;
    if (typeof unloadModel === "function") unloadModel(b.dataset.unload);
  });

  // header 状态：GPU 显存 / 推理模型 / tab 能力列表
  async function pollHeader() {
    if (document.hidden) return; // F4：后台标签页不再每 3s 打请求
    try {
      const info = await api("/api/info");
      const gpus = info.gpu || [];
      if (gpus.length) {
        const line = gpus.map((g) => `${g.index}:${g.used_gb}/${g.total_gb}`).join(" ");
        $("#gpu-text").textContent = line;
      } else {
        $("#gpu-text").textContent = "无 GPU 信息";
      }
      const inf = info.inference;
      if (inf && inf.loaded) {
        $("#inf-text").textContent = inf.model_path.split("/").slice(-2).join("/");
        $("#inf-dot").className = "dot " + (inf.device.startsWith("cuda") ? "gpu" : "idle");
      } else {
        $("#inf-text").textContent = "未加载";
        $("#inf-dot").className = "dot idle";
      }
      // GPU 互斥警告条统一收敛：加载且占满 → 提示；卸载/切 CPU/显存够 → 清空。
      // 原先警告只在 loaded 分支里更新，卸载后旧警告会残留（A10 显存紧张防 OOM）。
      const gpuFree = info.gpu && info.gpu.some((g) => g.used_gb < g.total_gb * 0.95);
      const warn = $("#warn-line");
      // 警告点名模型 + 卸载按钮携带同一身份（卸载的就是这里显示的模型）
      const infName = inf && inf.model_path ? inf.model_path.replace(/\\/g, "/").split("/").pop() : "";
      const w = inf && inf.loaded && info.train && inf.device.startsWith("cuda") && gpuFree === false
        ? `<div class="warnbar err">⚠ 推理模型 ${esc(infName)} 已加载并占满显存 — 启动训练前请先卸载 <button class="btn sm" type="button" data-unload="${esc(inf.model_path)}" title="卸载 ${esc(infName)}，释放显存（可随时重新加载）">卸载</button></div>`
        : "";
      if (warn && warn.innerHTML !== w) warn.innerHTML = w;
      // 能力列表控制 tab 显隐（--no-train 部署只留推理）
      const tabSet = new Set(info.tabs || []);
      $$("#tabs .tab").forEach((t) => {
        t.style.display = tabSet.has(t.dataset.tab) ? "" : "none";
      });
      // 校正：当前 tab 若因为能力列表被隐藏（例如 --no-train 删掉训练页），
      // 回落到第一个可用项 —— 否则用户会停在一个空的隐藏面板上。
      const cur = $("#tabs .tab.active");
      if (cur && cur.style.display === "none") {
        const first = $$("#tabs .tab").find((t) => t.style.display !== "none");
        if (first) switchTab(first.dataset.tab, false);
      }
    } catch (_) {
      notePoll(false); // F3：失联计数 —— 连续失败后顶部提示，不再静默
    }
  }
  pollHeader();
  setInterval(pollHeader, 3000);
  // F4：切回前台立即补一次刷新，不必等下一个轮询周期
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    pollHeader();
    if (typeof trainer !== "undefined" && trainer._runId) trainer.poll();
  });
}

document.addEventListener("DOMContentLoaded", initShell);

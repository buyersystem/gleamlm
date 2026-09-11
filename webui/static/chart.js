/* 轻量 Canvas 折线图（零依赖，多系列 + 阶段分界线 + 悬停取值）。
   用法:
     const chart = new LineChart(canvas);
     chart.render({ series, phases, yLabel, xLabel });
     // series: [{name, color, points:[[x,y],...], dim?}]   dim=true 灰显（历史对比）
     // phases: [{x, label}]  x 处画分界线（如 WSD warmup/stable/decay 分界）
   render 可反复调用（数据追加后整体重绘，数据量 ≤ 数万点场景足够）。 */
"use strict";

/* ── B1/B2：颜色一律从 :root 的 --chart-* 读，不在 canvas 里写死 ──
   canvas 拿不到 CSS 变量，硬编码又会脱离主题（换个色板就得满文件找）。
   只在构造时读一次并缓存：外观面板只改背景，不改色板，无需重建。
   任一变量缺失时回退到这里的兜底值 —— 灰度上线、零风险。 */
const CHART_FALLBACK = {
  grid: "#1f2430",
  axis: "#8b93a7",
  faint: "#8b93a7",
  phase: "rgba(180,140,255,0.5)",
  tipBg: "rgba(23,26,35,0.92)",
  tipBd: "#262b3a",
  loss: "#5dd0c9",
  lr: "#f5a97f",
  val: "#b48ce0",
  x1: "#e3b341",
  x2: "#7fd17f",
  cmpA: "#5b8cff",
  cmpB: "#d08cff",
  cmpDim: "#3a4152",
  bandWarmup: "rgba(180,140,255,0.10)",
  bandStable: "rgba(91,140,255,0.07)",
  bandDecay: "rgba(245,169,127,0.09)",
};

function _cssVar(name, fallback) {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  } catch (_) {
    return fallback;
  }
}

function readChartColors() {
  const f = CHART_FALLBACK;
  return {
    grid: _cssVar("--chart-grid", f.grid),
    axis: _cssVar("--chart-axis", f.axis),
    faint: _cssVar("--chart-faint", f.faint),
    phase: _cssVar("--chart-phase", f.phase),
    tipBg: _cssVar("--chart-tip-bg", f.tipBg),
    tipBd: _cssVar("--chart-tip-bd", f.tipBd),
    loss: _cssVar("--chart-loss", f.loss),
    lr: _cssVar("--chart-lr", f.lr),
    val: _cssVar("--chart-val", f.val),
    x1: _cssVar("--chart-x1", f.x1),
    x2: _cssVar("--chart-x2", f.x2),
    cmpA: _cssVar("--chart-cmp-a", f.cmpA),
    cmpB: _cssVar("--chart-cmp-b", f.cmpB),
    bandWarmup: _cssVar("--chart-band-warmup", f.bandWarmup),
    bandStable: _cssVar("--chart-band-stable", f.bandStable),
    bandDecay: _cssVar("--chart-band-decay", f.bandDecay),
  };
}

/* B2 + 补漏：色板的**唯一出口**。pretrain.js / posttrain.js 都通过它取色
   （原先两份副本各自写死，改色要动两处）。
   ⚠️ 这是 static 方法 —— 批次 2 引入调用点却漏了定义，导致两个训练页的图表
   直接抛错停摆。任何以 `X.y()` 形式新增的调用，都必须同时确认 X 上有 y 的定义。 */
function paletteOf() {
  const c = readChartColors();
  return {
    loss: c.loss,
    lr: c.lr,
    val: c.val,
    x1: c.x1,
    x2: c.x2,
    cmpA: c.cmpA,
    cmpB: c.cmpB,
    bandWarmup: c.bandWarmup,
    bandStable: c.bandStable,
    bandDecay: c.bandDecay,
  };
}

/* B4/Q3：对比线用「同色相 + 虚线 + 更细」区分，不再统一涂灰 ——
   原实现所有 dim 系列都是 #3a4152，勾两条历史 run 后根本认不出哪条是哪条。
   dim 保留为兜底（旧调用方仍可用）。 */
function seriesStyle(s, c) {
  if (s.cmp === 0) return { color: c.cmpA, width: 1.4, dash: [5, 3] };
  if (s.cmp === 1) return { color: c.cmpB, width: 1.4, dash: [2, 3] };
  if (s.dim) return { color: CHART_FALLBACK.cmpDim, width: 1, dash: [] };
  return { color: s.color, width: 1.8, dash: [] };
}

/* 图例色块：虚线系列用 repeating-gradient 画成虚线段，不额外加 DOM */
function swatchStyle(st) {
  if (!st.dash.length) return `background:${st.color}`;
  return `background:repeating-linear-gradient(90deg, ${st.color} 0 4px, transparent 4px 7px)`;
}

class LineChart {
  /* 静态入口：两页都写 `LineChart.palette()`。
     实现转调 paletteOf()，让色板逻辑可以被单测直接断言（不必构造 canvas）。 */
  static palette() {
    return paletteOf();
  }

  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = { series: [], phases: [], yLabel: "", xLabel: "step" };
    this.mouse = null;
    this._raf = 0;
    this._legendSig = null;
    this.c = readChartColors();
    // B5：canvas 对读屏是空盒子 —— 给 role + 随数据更新的 aria-label
    canvas.setAttribute("role", "img");
    // 悬停/触摸重绘经 rAF 合帧（全量重绘毫秒级，逐像素同步 draw 会卡）
    const at = (cx, cy) => {
      const r = canvas.getBoundingClientRect();
      this.mouse = { x: cx - r.left, y: cy - r.top };
      this._schedule();
    };
    canvas.addEventListener("mousemove", (e) => at(e.clientX, e.clientY));
    // B5：触屏没有 hover，不绑 touch 就永远取不到曲线值。
    // 不 preventDefault —— 保留在图上拖动滚页面的能力。
    canvas.addEventListener(
      "touchstart",
      (e) => {
        const t = e.touches[0];
        if (t) at(t.clientX, t.clientY);
      },
      { passive: true },
    );
    canvas.addEventListener(
      "touchmove",
      (e) => {
        const t = e.touches[0];
        if (t) at(t.clientX, t.clientY);
      },
      { passive: true },
    );
    const clear = () => {
      this.mouse = null;
      this._schedule();
    };
    canvas.addEventListener("touchend", clear);
    canvas.addEventListener("touchcancel", clear);
    canvas.addEventListener("mouseleave", clear);
    // 窗口缩放后位图按新布局尺寸重绘（rAF 合帧）
    window.addEventListener("resize", () => this._schedule());
  }

  /* rAF 合帧：同一帧内只重绘一次 */
  _schedule() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      this.draw();
    });
  }

  render(data) {
    this._retry = 0; // 新数据重给 rAF 补绘机会（见 draw 的 0 尺寸分支）
    this.data = data;
    // 每系列抽稀至 maxPoints（默认 2000），桶内 min/max 保峰谷（见文件尾 decimate）
    this.data = {
      ...data,
      series: (data.series || []).map((s) => ({
        ...s,
        points: decimate((s.points || []).filter((p) => p[1] != null && isFinite(p[1])), data.maxPoints || 2000),
      })),
    };
    this.draw();
    this._syncLegend();
  }

  /* B3 图例：注入到 .chart-box 内（每个图自带，无需改 HTML）。
     用签名比对避免每 2s 轮询都重写 DOM。 */
  _syncLegend() {
    const box = this.cv.parentElement;
    if (!box || !box.classList.contains("chart-box")) return;
    const series = this.data.series || [];
    const items = series.map((s) => {
      const st = seriesStyle(s, this.c);
      return { name: s.name || "", style: swatchStyle(st), color: st.color };
    });
    const sig = items.map((i) => i.name + "|" + i.style).join("~");
    if (sig === this._legendSig) return;
    this._legendSig = sig;
    let el = box.querySelector(".chart-legend");
    if (!el) {
      el = document.createElement("div");
      el.className = "chart-legend";
      box.appendChild(el);
    }
    el.textContent = "";
    for (const it of items) {
      const span = document.createElement("span");
      span.className = "it";
      const sw = document.createElement("span");
      sw.className = "sw";
      sw.setAttribute("style", it.style);
      span.appendChild(sw);
      span.appendChild(document.createTextNode(it.name));
      el.appendChild(span);
    }
    // B5：给读屏一句摘要（最新值与系列名）
    const last = series.length
      ? `，最新 ${formatLast(series[0])}`
      : "";
    this.cv.setAttribute(
      "aria-label",
      series.length
        ? `折线图：${items.map((i) => i.name).join("、")}${last}`
        : "折线图：暂无数据",
    );
  }

  _size() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.cv.clientWidth, h = this.cv.clientHeight;
    if (this.cv.width !== Math.round(w * dpr)) {
      this.cv.width = Math.round(w * dpr);
      this.cv.height = Math.round(h * dpr);
    }
    return { w, h, dpr };
  }

  draw() {
    const { w, h, dpr } = this._size();
    // 隐藏容器（tab 未激活）中首渲时布局尺寸为 0，直接画会固化成 0×0 空白位图；
    // 已可见但仍为 0（布局未定）时 rAF 补绘几次，待布局完成重画
    if ((!w || !h) && this.cv.offsetParent !== null && (this._retry || 0) < 5) {
      this._retry = (this._retry || 0) + 1;
      requestAnimationFrame(() => this.draw());
      return;
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const series = this.data.series || [];
    const all = series.flatMap((s) => s.points || []);
    if (!all.length) {
      ctx.fillStyle = this.c.faint;
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("启动任务后，等待训练指标…", w / 2, h / 2);
      return;
    }
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    // x 轴从 0 起（step 是进度计数，随首点起步会把刚启动的曲线挤到右端）
    // y 轴默认自适应；zeroY 时也从 0 起（lr 图看 WSD 相对衰减幅度）
    let xMin = Math.min(0, ...xs), xMax = Math.max(...xs);
    // B6：log y 纵轴（默认关）。loss 跨量级（如 1e5 尖峰）时线性轴会把正常区间压成一条直线。
    // 做法：y 域换到 log10 空间做线性映射，刻度与悬停值再反变换回原值显示。
    const yd = yDomain(ys, { logY: this.data.logY, zeroY: this.data.zeroY });
    const logY = yd.log;
    let yMin = yd.min, yMax = yd.max;
    // x 右端带一档余量; 0 是硬起点, 左端不再 pad（仅理论负值数据保留左 pad）
    const pad = Math.max(1, (xMax - xMin) * 0.04);
    if (xMin < 0) xMin -= pad;
    xMax += pad;
    // B7：整根轴用同一份刻度精度（必须在 pad 之后取量级）。
    // B6：log 轴的量级要按**反变换后的显示值**取，否则精度会按 log 值算错。
    const yFmt = makeTickFmt(logY ? 10 ** yMax : Math.max(Math.abs(yMin), Math.abs(yMax)));

    // 底边距 34: 刻度数字 (top 基线) 与 xLabel (bottom 基线) 需要各占一行,
    // 26 时两者在 h-20~h-9 与 h-13~h-2 重叠约 4px — 刻度贴 plot 底、标签另起行
    const m = { l: 56, r: 14, t: 14, b: 34 };
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    const X = (x) => m.l + ((x - xMin) / (xMax - xMin)) * pw;
    // B6：Y 收「数据值」（曲线点、悬停点）；YT 收「已变换值」（刻度循环）。
    // 线性轴下两者等价，log 轴下差一步 log10 —— 分开命名免得混用。
    const Y = (v) => m.t + (1 - ((logY ? Math.log10(Math.max(v, 1e-12)) : v) - yMin) / (yMax - yMin)) * ph;
    const YT = (t) => m.t + (1 - (t - yMin) / (yMax - yMin)) * ph;
    const yDisp = (t) => (logY ? 10 ** t : t); // 刻度标签：变换值 → 原值

    // 网格 + y 刻度
    ctx.font = "11px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const steps = 5;
    for (let i = 0; i <= steps; i++) {
      const val = yMin + ((yMax - yMin) * i) / steps;
      const y = YT(val);
      ctx.strokeStyle = this.c.grid;
      ctx.beginPath();
      ctx.moveTo(m.l, y);
      ctx.lineTo(w - m.r, y);
      ctx.stroke();
      ctx.fillStyle = this.c.axis;
      ctx.fillText(yFmt(yDisp(val)), m.l - 7, y);
    }
    // x 刻度
    const xt = 5;
    for (let i = 0; i <= xt; i++) {
      const x = xMin + ((xMax - xMin) * i) / xt;
      ctx.fillStyle = this.c.axis;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(fmtStepTick(x), X(x), h - m.b + 6);
    }
    // xLabel 单独一行（m.b=34 后与刻度数字不再重叠）
    ctx.fillStyle = this.c.faint;
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText(this.data.xLabel || "step", w - m.r, h - 2);
    ctx.textAlign = "left";
    ctx.save();
    ctx.translate(0, 0);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(this.data.yLabel || "", -m.t - 4, 14);
    ctx.restore();

    // P1：lr 副图的 WSD 三阶段底色。分界线说明「在哪切」，底色说明「这一段是什么」——
    // 设计文档 §6.1 要的是两者都有。画在网格之上、曲线之下，避免压住数据。
    for (const bd of this.data.bands || []) {
      if (bd.to < xMin || bd.from > xMax) continue;
      const x0 = X(Math.max(bd.from, xMin));
      const x1 = X(Math.min(bd.to, xMax));
      const wBand = Math.max(1, x1 - x0);
      ctx.fillStyle = bd.color;
      ctx.fillRect(x0, m.t, wBand, ph);
      // 区间名只在够宽时画，免得窄区间上文字互相叠
      if (bd.label && wBand > 34) {
        ctx.fillStyle = this.c.phase;
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(bd.label, (x0 + x1) / 2, m.t + 8);
      }
    }

    // 阶段分界线（WSD：warmup 结束 / stable 结束）
    for (const ph of this.data.phases || []) {
      if (ph.x < xMin || ph.x > xMax) continue;
      ctx.strokeStyle = this.c.phase;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(X(ph.x), m.t);
      ctx.lineTo(X(ph.x), h - m.b);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = this.c.phase;
      ctx.font = "10px sans-serif";
      ctx.textAlign = ph.x > (xMin + xMax) / 2 ? "right" : "left";
      ctx.fillText(ph.label || "", X(ph.x) + (ph.x > (xMin + xMax) / 2 ? -4 : 4), m.t + 8);
    }

    // 各系列曲线
    for (const s of series) {
      const pts = (s.points || []).filter((p) => p[1] != null && isFinite(p[1]));
      if (pts.length < 1) continue;
      const st = seriesStyle(s, this.c);
      ctx.strokeStyle = st.color;
      ctx.lineWidth = st.width;
      ctx.setLineDash(st.dash);
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = X(p[0]), y = Y(p[1]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 悬停：最近 x 点竖线 + 数值
    if (this.mouse && this.mouse.x > m.l && this.mouse.x < w - m.r) {
      const mx = xMin + ((this.mouse.x - m.l) / pw) * (xMax - xMin);
      let best = null; // {dist, x, s, y}
      for (const s of series) {
        for (const p of s.points || []) {
          const d = Math.abs(p[0] - mx);
          if (!best || d < best.dist) {
            best = { dist: d, x: p[0], s: s.name, y: p[1], color: seriesStyle(s, this.c).color };
          }
        }
      }
      if (best) {
        ctx.strokeStyle = this.c.axis;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(X(best.x), m.t);
        ctx.lineTo(X(best.x), h - m.b);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = this.c.tipBg;
        ctx.strokeStyle = best.color;
        ctx.beginPath();
        ctx.arc(X(best.x), Y(best.y), 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.font = "11px monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        const txt = `${best.s} x=${fmtStepTick(best.x)} y=${yFmt(best.y)}`;
        const tw = ctx.measureText(txt).width + 12;
        let bx = X(best.x) + 8;
        if (bx + tw > w - m.r) bx = X(best.x) - tw - 8;
        // B8：tooltip 原先固定在绘图区顶部（m.t+4），而阶段标签也画在顶部 ——
        // 两者直接叠在一起。改为跟随数据点纵向移动，并让开顶部的阶段标签带。
        const topGuard = m.t + ((this.data.phases || []).length ? 30 : 8);
        const ty = Math.max(topGuard, Math.min(Y(best.y) - 24, h - m.b - 22));
        ctx.fillStyle = this.c.tipBg;
        ctx.strokeStyle = this.c.tipBd;
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(bx, ty, tw, 18, 5) : ctx.rect(bx, ty, tw, 18);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = best.color;
        ctx.fillText(txt, bx + 6, ty + 3);
      }
    }
  }
}

/* 图例摘要用：取某系列最后一点的人类可读文本 */
function formatLast(s) {
  const pts = (s && s.points) || [];
  if (!pts.length) return "";
  const p = pts[pts.length - 1];
  return `step ${fmtStepTick(p[0])} 的 ${s.name} = ${fmtTick(p[1])}`;
}

/* B6：算 y 域（含 6% padding）。抽成纯函数是为了可测 ——
   log 轴的边界（数据全非正 / 全部相等）用浏览器验不了，只能靠断言。
   返回 { min, max, log }；log 为 false 表示**没有真的用对数轴**。
   非正数据（loss=0）令 log10 无定义 → 静默退化为线性轴，而不是画一张空图。 */
function yDomain(ys, opts) {
  const wantLog = !!(opts && opts.logY);
  const zeroY = !!(opts && opts.zeroY);
  const pos = ys.filter((v) => v > 0);
  const useLog = wantLog && pos.length > 0;
  let min;
  let max;
  if (useLog) {
    const lo = Math.log10(Math.min(...pos));
    const hi = Math.log10(Math.max(...pos));
    min = lo;
    max = hi > lo ? hi : lo + 1; // 值全等时给一档跨度，避免后面除零
  } else {
    min = Math.min(...ys);
    max = Math.max(...ys);
  }
  if (!useLog && zeroY && min > 0) min = 0;
  const span = max - min || Math.abs(max) || 1;
  /* 纵坐标「从 0 起」：数据非负时下界**就落在 0**，不再往下留 6% padding ——
     否则轴线实际停在负数处、0 悬在轴上方一截，看着并不像从 0 开始。
     数据含负值时（DPO 的 margin 可以为负）保留 dataMin：不能为了贴 0 裁掉负半轴。
     logY 模式下整段失效（log(0) 无定义），由上面的 useLog 分支决定。 */
  const lo = !useLog && zeroY && min >= 0 ? 0 : min - span * 0.06;
  return { min: lo, max: max + span * 0.06, log: useLog };
}

/* B7：原先每个刻度按自己的量级决定小数位 —— 同一根轴上会出现
   「2.500 / 5.000 / 12.5」这种位数不齐的混排，看着像精度不同。
   改为按该轴的最大绝对量级定一次精度，全体刻度同格式。
   （fmtTick 保留给无轴上下文的单点场景，如图例摘要。） */
function makeTickFmt(maxAbs) {
  const a = Math.abs(maxAbs);
  if (a >= 1e6) return (v) => (v / 1e6).toFixed(1) + "M";
  if (a >= 1e4) return (v) => (v / 1e3).toFixed(1) + "k";
  if (a >= 1000) return (v) => v.toFixed(0);
  if (a >= 10) return (v) => v.toFixed(1);
  if (a >= 0.01) return (v) => v.toFixed(3);
  return (v) => v.toExponential(1);
}

function fmtTick(v) {
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (Math.abs(v) >= 1e4) return (v / 1e3).toFixed(1) + "k";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  if (Math.abs(v) >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}

/* x 轴（step）刻度/悬停用：step 是整数计数，任何量级都不出现小数；
   刻度值本身由范围均分产生（可能非整），先四舍五入再套 k/M 后缀。 */
function fmtStepTick(v) {
  const r = Math.round(v);
  if (Math.abs(r) >= 1e6) return Math.round(r / 1e6) + "M";
  if (Math.abs(r) >= 1e4) return Math.round(r / 1e3) + "k";
  return String(r);
}

/* 抽稀：点数 > max 时按桶 min/max 包络保留峰谷。
   等距抽样会削掉 loss 尖峰/lr 衰减拐点；每桶输出段内极值点则形状保真。
   桶数 = max/4，每桶至多输出首/极值/尾 4 点（含去重）→ 输出 ≤ max 有界。
   入参须已滤非有限值（桶内比较依赖 y 可排序）。 */
function decimate(pts, max) {
  const n = pts.length;
  if (n <= max) return pts;
  const buckets = Math.max(8, max >> 2);
  const out = [];
  const step = n / buckets;
  const push = (i) => {
    const last = out[out.length - 1];
    // 相邻重复点（同 x）跳过：折线无贡献，且会干扰悬停最近点计数
    if (last && last[0] === pts[i][0]) return;
    out.push(pts[i]);
  };
  for (let b = 0; b < buckets; b++) {
    const i0 = Math.floor(b * step);
    const i1 = b === buckets - 1 ? n - 1 : Math.max(i0 + 1, Math.floor((b + 1) * step) - 1);
    push(i0);
    if (i1 - i0 > 1) {
      let mn = i0, mx = i0;
      for (let i = i0 + 1; i <= i1; i++) {
        if (pts[i][1] < pts[mn][1]) mn = i;
        if (pts[i][1] > pts[mx][1]) mx = i;
      }
      if (mn !== i0 && mn !== i1) push(mn);
      if (mx !== mn && mx !== i0 && mx !== i1) push(mx);
    }
    push(i1);
  }
  return out;
}

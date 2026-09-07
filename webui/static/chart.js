/* 轻量 Canvas 折线图（零依赖，多系列 + 阶段分界线 + 悬停取值）。
   用法:
     const chart = new LineChart(canvas);
     chart.render({ series, phases, yLabel, xLabel });
     // series: [{name, color, points:[[x,y],...], dim?}]   dim=true 灰显（历史对比）
     // phases: [{x, label}]  x 处画分界线（如 WSD warmup/stable/decay 分界）
   render 可反复调用（数据追加后整体重绘，数据量 ≤ 数万点场景足够）。 */
"use strict";

class LineChart {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = { series: [], phases: [], yLabel: "", xLabel: "step" };
    this.mouse = null;
    canvas.addEventListener("mousemove", (e) => {
      const r = canvas.getBoundingClientRect();
      this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
      this.draw();
    });
    canvas.addEventListener("mouseleave", () => {
      this.mouse = null;
      this.draw();
    });
  }

  render(data) {
    this.data = data;
    this.draw();
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
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const series = this.data.series || [];
    const all = series.flatMap((s) => s.points || []);
    if (!all.length) {
      ctx.fillStyle = "#5c6474";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("启动任务后，等待训练指标…", w / 2, h / 2);
      return;
    }
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    let xMin = Math.min(...xs), xMax = Math.max(...xs);
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    // x 轴带一档余量；y 轴 pad 5%
    const pad = Math.max(1, (xMax - xMin) * 0.04);
    xMin -= pad; xMax += pad;
    const ySpan = yMax - yMin || Math.abs(yMax) || 1;
    yMin -= ySpan * 0.06; yMax += ySpan * 0.06;

    const m = { l: 56, r: 14, t: 14, b: 26 };
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    const X = (x) => m.l + ((x - xMin) / (xMax - xMin)) * pw;
    const Y = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * ph;

    // 网格 + y 刻度
    ctx.font = "11px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const steps = 5;
    for (let i = 0; i <= steps; i++) {
      const val = yMin + ((yMax - yMin) * i) / steps;
      const y = Y(val);
      ctx.strokeStyle = "#1f2430";
      ctx.beginPath();
      ctx.moveTo(m.l, y);
      ctx.lineTo(w - m.r, y);
      ctx.stroke();
      ctx.fillStyle = "#8b93a7";
      ctx.fillText(fmtTick(val), m.l - 7, y);
    }
    // x 刻度
    const xt = 5;
    for (let i = 0; i <= xt; i++) {
      const x = xMin + ((xMax - xMin) * i) / xt;
      ctx.fillStyle = "#8b93a7";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(fmtStepTick(x), X(x), h - m.b + 6);
    }
    ctx.fillStyle = "#5c6474";
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText(this.data.xLabel || "step", w - m.r, h - 2);
    ctx.textAlign = "left";
    ctx.save();
    ctx.translate(0, 0);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(this.data.yLabel || "", -m.t - 4, 14);
    ctx.restore();

    // 阶段分界线（WSD：warmup 结束 / stable 结束）
    for (const ph of this.data.phases || []) {
      if (ph.x < xMin || ph.x > xMax) continue;
      ctx.strokeStyle = "rgba(180,140,255,0.5)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(X(ph.x), m.t);
      ctx.lineTo(X(ph.x), h - m.b);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(180,140,255,0.9)";
      ctx.font = "10px sans-serif";
      ctx.textAlign = ph.x > (xMin + xMax) / 2 ? "right" : "left";
      ctx.fillText(ph.label || "", X(ph.x) + (ph.x > (xMin + xMax) / 2 ? -4 : 4), m.t + 8);
    }

    // 各系列曲线
    for (const s of series) {
      const pts = (s.points || []).filter((p) => p[1] != null && isFinite(p[1]));
      if (pts.length < 1) continue;
      ctx.strokeStyle = s.dim ? "#3a4152" : s.color;
      ctx.lineWidth = s.dim ? 1 : 1.8;
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = X(p[0]), y = Y(p[1]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // 悬停：最近 x 点竖线 + 数值
    if (this.mouse && this.mouse.x > m.l && this.mouse.x < w - m.r) {
      const mx = xMin + ((this.mouse.x - m.l) / pw) * (xMax - xMin);
      let best = null; // {dist, x, s, y}
      for (const s of series) {
        for (const p of s.points || []) {
          const d = Math.abs(p[0] - mx);
          if (!best || d < best.dist) best = { dist: d, x: p[0], s: s.name, y: p[1], color: s.dim ? "#8b93a7" : s.color };
        }
      }
      if (best) {
        ctx.strokeStyle = "rgba(139,147,167,0.6)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(X(best.x), m.t);
        ctx.lineTo(X(best.x), h - m.b);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#0f1117";
        ctx.strokeStyle = best.color;
        ctx.beginPath();
        ctx.arc(X(best.x), Y(best.y), 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.font = "11px monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        const txt = `${best.s} x=${fmtStepTick(best.x)} y=${fmtTick(best.y)}`;
        const tw = ctx.measureText(txt).width + 12;
        let bx = X(best.x) + 8;
        if (bx + tw > w - m.r) bx = X(best.x) - tw - 8;
        ctx.fillStyle = "rgba(23,26,35,0.92)";
        ctx.strokeStyle = "#262b3a";
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(bx, m.t + 4, tw, 18, 5) : ctx.rect(bx, m.t + 4, tw, 18);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = best.color;
        ctx.fillText(txt, bx + 6, m.t + 7);
      }
    }
  }
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

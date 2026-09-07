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

/* ── modal 通用开关 ── */
function openModal(html) {
  const box = $("#modal-box");
  box.innerHTML = html;
  $("#modal-mask").classList.add("show");
  bindClose(box);
  return box;
}

function closeModal() {
  $("#modal-mask").classList.remove("show");
}

function bindClose(root) {
  $$("[data-close]", root).forEach((b) =>
    b.addEventListener("click", () => $("#" + b.dataset.close).classList.remove("show"))
  );
}

/* ── 页面壳：tab 切换 / 外观 / header 状态轮询 ── */
function initShell() {
  initAppearance();
  // 静态弹窗（外观 look-mask 等）的 data-close ✕ 需在此绑定；动态注入的由 openModal 绑定
  bindClose(document);
  $("#look-btn").addEventListener("click", () => $("#look-mask").classList.add("show"));
  // tab 切换
  $$("#tabs .tab").forEach((t) =>
    t.addEventListener("click", () => switchTab(t.dataset.tab))
  );
  window.switchTab = switchTab;
  function switchTab(name) {
    $$("#tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    $$(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "pane-" + name));
    window.dispatchEvent(new CustomEvent("tabchange", { detail: name }));
  }
  // header 状态：GPU 显存 / 推理模型 / tab 能力列表
  async function pollHeader() {
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
        const gpuFree = info.gpu && info.gpu.some((g) => g.used_gb < g.total_gb * 0.95);
        // 推理模型占显存时给训练 tab 互斥提示（A10 显存紧张防 OOM）
        const warn = $("#warn-line");
        const w = info.train && inf.device.startsWith("cuda") && gpuFree === false
          ? `<div class="warnbar err">⚠ 推理模型已加载并占满显存 — 启动训练前请先卸载推理模型或切换到 CPU 推理</div>`
          : "";
        if (warn && warn.innerHTML !== w) warn.innerHTML = w;
      } else {
        $("#inf-text").textContent = "未加载";
        $("#inf-dot").className = "dot idle";
      }
      // 能力列表控制 tab 显隐（--no-train 部署只留推理）
      const tabSet = new Set(info.tabs || []);
      $$("#tabs .tab").forEach((t) => {
        t.style.display = tabSet.has(t.dataset.tab) ? "" : "none";
      });
    } catch (_) {
      /* 服务未就绪时静默 */
    }
  }
  pollHeader();
  setInterval(pollHeader, 3000);
}

document.addEventListener("DOMContentLoaded", initShell);

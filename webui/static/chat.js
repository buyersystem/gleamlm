/* ③ 推理 tab：模型管理（/api/models checkpoint 树 + /v1/models/load 热切换）
   与 ChatML 对话台（/v1/chat/completions 流式 SSE，多轮自动带最近 6 轮）。 */
"use strict";

const chat = {
  convs: [],    // [{role, content}] 全部轮次（发送时裁最近 6 轮给后端）
  model: "",    // 已加载模型相对路径（本 tab 最近一次 load 的 req path）
  busy: false,
  ctrl: null,   // AbortController（生成中再次点发送 = 停止）
};

async function loadModels() {
  const [m, st] = await Promise.all([
    api("/api/models").catch(() => ({ groups: {} })),
    api("/v1/models/status").catch(() => ({ loaded: false, model_path: "" })),
  ]);
  const stPath = String(st.model_path || "").replace(/\\/g, "/");
  if (!chat.model && st.loaded) chat.model = stPath.split("/").pop(); // 刷新后只认 basename
  const cur = st.loaded ? stPath.split("/").pop() : "";
  $("#chat-cur").textContent = st.loaded
    ? stPath.split("/").slice(-2).join("/")
    : "未加载";
  $("#chat-cur").title = stPath;
  const groups = m.groups || {};
  const box = $("#chat-models");
  const vks = Object.keys(groups);
  if (!vks.length) {
    box.innerHTML =
      '<div class="hint" style="padding:14px 4px">checkpoints/ 下暂无 .pt<br/>' +
      "训练产出后点「刷新」即时出现</div>";
    return;
  }
  box.innerHTML = vks
    .map((v) => {
      const rows = groups[v]
        .map((f) => {
          const name = f.stage ? `${f.stage}/${f.name}` : f.name;
          const on = st.loaded && cur === f.name;
          return `<div class="list-item ${on ? "on" : ""}" title="${esc(f.path)}"
     data-path="${esc(f.path)}">
    <span class="name" style="flex:1;overflow:hidden;text-overflow:ellipsis">${esc(name)}</span>
    <span class="sub" style="white-space:nowrap">${f.size_mb}MB · ${esc(f.mtime)}</span>
    <button class="btn sm ${on ? "ghost" : ""}" type="button" data-load="${esc(f.path)}">${on ? "已加载" : "加载"}</button>
  </div>`;
        })
        .join("");
      return `<div class="p-body pad0">
    <div style="padding:6px 10px 2px;font-size:12px;color:var(--warn)">◈ ${esc(v)}</div>
    ${rows}
  </div>`;
    })
    .join("");
  // 点击行主体 = 加载该模型；按钮也是（事件委托一次绑定）
  box.querySelectorAll("[data-load]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      loadModel(b.dataset.load, b);
    });
  });
  box.querySelectorAll(".list-item").forEach((li) => {
    li.addEventListener("click", () => {
      const b = li.querySelector("[data-load]");
      if (b && b.disabled === false) loadModel(b.dataset.load, b);
    });
  });
}

async function loadModel(path, btn) {
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "加载中…";
  try {
    await api("/v1/models/load", {
      method: "POST",
      body: JSON.stringify({ model_path: path }),
    });
    chat.model = path;
    await loadModels(); // 刷新列表高亮 + 状态行
    pushMsg("ok", `模型已加载：${path}（对话参数沿用上方滑杆/输入框）`);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = old;
    pushMsg("error", `加载失败：${err.message}`);
  }
}

/* ── 对话区 ── */
function clearHint() {
  const hint = $("#chat-scroll .hint");
  if (hint) hint.remove();
}

function pushMsg(kind, text, streaming) {
  clearHint();
  const wrap = document.createElement("div");
  wrap.className = "msg " + kind;
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (streaming ? " thinking" : "");
  bubble.textContent = text;
  wrap.appendChild(bubble);
  $("#chat-scroll").appendChild(wrap);
  scrollChat();
  return bubble;
}

function scrollChat() {
  const el = $("#chat-scroll");
  el.scrollTop = el.scrollHeight;
}

/* 流式 SSE 解析（data: {json} 帧；[DONE] 收尾）。返回完整生成文本。 */
async function readSseStream(resp, onDelta) {
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let full = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const dl = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!dl) continue;
      const payload = dl.slice(5).trim();
      if (payload === "[DONE]") continue;
      let j;
      try {
        j = JSON.parse(payload);
      } catch (_) {
        continue;
      }
      const d = j.choices && j.choices[0] && j.choices[0].delta
        ? j.choices[0].delta.content || ""
        : j.choices && j.choices[0] && j.choices[0].text ? j.choices[0].text : "";
      if (d) {
        full += d;
        onDelta(full);
      }
    }
  }
  return full;
}

async function sendChat(e) {
  e.preventDefault();
  if (chat.busy) {
    // 生成中再点发送 = 停止
    if (chat.ctrl) chat.ctrl.abort();
    return;
  }
  const ta = $("#chat-input");
  const text = ta.value.trim();
  if (!text) return;
  if (!chat.model) {
    pushMsg("error", "请先在左侧模型列表选择一个 checkpoint 并加载");
    return;
  }
  chat.convs.push({ role: "user", content: text });
  pushMsg("user", text);
  ta.value = "";
  autoGrow(ta);
  const sendBtn = $("#chat-send");
  chat.busy = true;
  sendBtn.textContent = "停止";
  chat.ctrl = new AbortController();
  const bubble = pushMsg("assistant", "", true);
  try {
    const params = {
      messages: chat.convs.slice(-12), // 最近 6 轮（6 组 user/assistant）
      temperature: parseFloat($("#temp").value) || 0.8,
      top_p: parseFloat($("#chat-topp").value) || 0.9,
      top_k: parseInt($("#chat-topk").value, 10) || 0,
      repetition_penalty: parseFloat($("#chat-rep").value) || 1.0,
      max_tokens: parseInt($("#chat-maxtok").value, 10) || 256,
      stream: true,
    };
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
      signal: chat.ctrl.signal,
    });
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try {
        const j = await resp.json();
        msg = (j.detail && j.detail.message) || j.detail || msg;
        if (typeof msg !== "string") msg = JSON.stringify(msg);
      } catch (_) { /* 非 JSON 错误体 */ }
      throw new Error(msg);
    }
    if (!resp.body) throw new Error("响应无流式体");
    const full = await readSseStream(resp, (t) => {
      bubble.textContent = t;
      scrollChat();
    });
    bubble.classList.remove("thinking");
    if (!full) bubble.textContent = "（无输出 — 首 token 即命中 eos/im_end，可尝试降低温度或调小 max_tokens）";
    else chat.convs.push({ role: "assistant", content: full });
  } catch (err) {
    bubble.classList.remove("thinking");
    const aborted = err.name === "AbortError";
    if (aborted && bubble.textContent) {
      // 用户手动停止：已生成部分保留
      chat.convs.push({ role: "assistant", content: bubble.textContent });
    } else if (aborted) {
      bubble.remove();
    } else {
      bubble.textContent = "";
      bubble.parentElement.classList.add("error");
      bubble.textContent = `请求失败：${err.message}`;
      // 512 上下文溢出等 400 错误：清掉已入列的 user 轮次，让用户可换方式重试
      if (/上下文长度|context|max_tokens|temperature/.test(err.message)) {
        chat.convs.pop();
      }
    }
  } finally {
    chat.busy = false;
    chat.ctrl = null;
    sendBtn.textContent = "发送";
    scrollChat();
  }
}

function autoGrow(ta) {
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
}

function initChat() {
  $("#chat-form").addEventListener("submit", sendChat);
  $("#chat-send").addEventListener("click", (e) => {
    if (chat.busy) {
      e.preventDefault();
      sendChat(e);
    }
  });
  const ta = $("#chat-input");
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("#chat-form").dispatchEvent(new Event("submit"));
    }
  });
  ta.addEventListener("input", () => autoGrow(ta));
  $("#temp").addEventListener("input", () => {
    $("#temp-val").textContent = $("#temp").value;
  });
  $("#chat-clear").addEventListener("click", () => {
    chat.convs = [];
    $("#chat-scroll").innerHTML =
      '<div class="hint"><b>GleamLM 已就绪</b><br/>' +
      "左侧选择 checkpoint 并点「加载」；下方输入开始对话，Enter 发送，Shift+Enter 换行。</div>";
  });
  $("#chat-refresh").addEventListener("click", () => loadModels());
  loadModels().catch(() => {});
}

document.addEventListener("DOMContentLoaded", initChat);

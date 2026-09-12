/* ③ 推理 tab：模型管理（/api/models checkpoint 树 + /v1/models/load 热切换）
   与 ChatML 对话台（/v1/chat/completions 流式 SSE，多轮自动带最近 6 轮）。 */
"use strict";

const chat = {
  convs: [],    // [{role, content}] 全部轮次（发送时裁最近 6 轮给后端）
  model: "",    // 已加载模型相对路径（本 tab 最近一次 load 的 req path）
  busy: false,
  ctrl: null,   // AbortController（生成中再次点发送 = 停止）
};

/* P6：「新训练」徽章 —— 24 小时内产出的 checkpoint。
   mtime 是后端给的 "YYYY-MM-DD HH:MM" 字符串，手动解析（各浏览器对
   这个格式的 Date 解析行为并不一致，不能直接 new Date(str)）。 */
function isNewModel(mtime) {
  const m = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})$/.exec(String(mtime || ""));
  if (!m) return false;
  const t = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]).getTime();
  return Date.now() - t < 24 * 3600 * 1000;
}

/* H22：列表里只显示日期 —— 260px 面板下 "2026-09-11 14:13" 要占 156px，
   而同一个 run 的各产物时间只差几分钟，精确到分的价值很低。
   完整时间挪进 title（悬停可见），不丢信息。 */
function shortDate(mtime) {
  const m = /^(\d{4}-\d{2}-\d{2})/.exec(String(mtime || ""));
  return m ? m[1] : String(mtime || "");
}

async function loadModels() {
  // 片一：allSettled 而非逐个 catch —— 要能区分「哪个端点失败」才能说出是哪个数据源。
  // 失败不清空已渲染的列表（stale-but-usable），避免轮询闪烁。
  const box0 = $("#chat-models");
  const hadData = !!(box0 && box0.children.length);
  const [mRes, stRes] = await Promise.allSettled([
    api("/api/models"),
    api("/v1/models/status"),
  ]);
  if (mRes.status === "fulfilled") {
    chat.modelsErr = "";
    clearFetchFail("chat-models");
  } else {
    chat.modelsErr = (mRes.reason && mRes.reason.message) || "请求失败";
    noteFetchFail("chat-models", hadData);
  }
  const m = mRes.status === "fulfilled" ? mRes.value : { groups: {} };
  const st = stRes.status === "fulfilled" ? stRes.value : { loaded: false, model_path: "" };
  const stPath = String(st.model_path || "").replace(/\\/g, "/");
  if (!chat.model && st.loaded) chat.model = stPath.split("/").pop(); // 刷新后只认 basename
  // P5：模型状态条 —— 当前模型 / 设备 / 参数量集中一处（原来是只有 basename 的裸文字）
  const nameEl = $("#chat-cur");
  if (nameEl) {
    nameEl.textContent = st.loaded ? stPath.split("/").pop() : "未加载";
    nameEl.title = stPath;
  }
  const devEl = $("#chat-dev");
  if (devEl) devEl.textContent = st.loaded ? String(st.device || "") : "";
  const subEl = $("#chat-sub");
  if (subEl) {
    // H22：原先拼的是**完整路径**，而 #chat-cur 已经显示了 basename → 重复且过长。
    // 改为只显示所在目录；完整路径仍在 title（nameEl 与 subEl 都有）。
    const stDir = stPath.split("/").filter(Boolean).slice(0, -1).join("/") || stPath;
    subEl.textContent = st.loaded
      ? `${st.params_m ? st.params_m + "M 参数 · " : ""}${stDir}`
      : "选择左侧 checkpoint 并加载";
    subEl.title = stPath;
  }
  const groups = m.groups || {};
  const box = $("#chat-models");
  const vks = Object.keys(groups);
  if (!vks.length) {
    // 片一：失败态优先于空态 —— 否则「读取失败」会被说成「checkpoints/ 下暂无 .pt」
    if (isLinkDown()) {
      box.innerHTML = '<div class="hint sm">与后端失联 —— 恢复后自动加载</div>';
    } else if (failShown("chat-models")) {
      box.innerHTML = failHtml(`无法读取 checkpoint 列表 · ${chat.modelsErr}`);
      bindRetry(box, () => loadModels());
    } else {
      box.innerHTML =
        '<div class="hint sm">checkpoints/ 下暂无 .pt<br/>' +
        "训练产出后点「刷新」即时出现</div>";
    }
    return;
  }
  box.innerHTML = vks
    .map((v) => {
      const rows = groups[v]
        .map((f) => {
          const name = f.stage ? `${f.stage}/${f.name}` : f.name;
          // 动作按钮与加载同一身份：已加载行显示「卸载」，卸载的就是本行 path
          const on = st.loaded && stPath.endsWith(f.path);
          const badge = isNewModel(f.mtime)
            ? '<span class="badge-new" title="24 小时内产出">新训练</span>'
            : "";
          const act = on
            ? `<button class="btn sm ghost" type="button" data-unload="${esc(f.path)}" title="卸载该模型，释放显存（可随时重新加载）">卸载</button>`
            : `<button class="btn sm" type="button" data-load="${esc(f.path)}" title="加载该模型进行推理">加载</button>`;
          return `<div class="list-item ckpt ${on ? "on" : ""}" title="${esc(f.path)} · ${esc(f.mtime)}"
     data-path="${esc(f.path)}">
    <div class="ck-name">
      <span class="name">${esc(name)}</span>
      ${badge}
    </div>
    <span class="sub">${f.size_mb}MB · ${shortDate(f.mtime)}</span>
    ${act}
  </div>`;
        })
        .join("");
      return `<div class="p-body pad0">
    <div style="padding:8px 8px 4px;font-size:12px;color:var(--warn)">◈ ${esc(v)}</div>
    ${rows}
  </div>`;
    })
    .join("");
  // 行内动作按钮：加载 / 卸载（各自带本行 path，事件委托一次绑定）
  box.querySelectorAll("[data-load]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      loadModel(b.dataset.load, b);
    });
  });
  box.querySelectorAll("[data-unload]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      unloadModel(b.dataset.unload, b);
    });
  });
  box.querySelectorAll(".list-item").forEach((li) => {
    li.addEventListener("click", () => {
      // 行点击 = 「加载」快捷方式；卸载只走行内按钮（防误触）
      const lb = li.querySelector("[data-load]");
      if (lb && lb.disabled === false) loadModel(lb.dataset.load, lb);
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
    // E9：加载结果不进对话流 —— 它不是一个对话轮次；且 .msg.ok 没有对应样式
    // （原先就是无底无边的裸文字，插在气泡之间）。改用右上角 toast。
    toast(`模型已加载：${path}`, "ok", 5200);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = old;
    toast(`加载失败：${err.message}`, "err", 7000);
  }
}

/* 卸载推理模型并释放显存 —— 与加载同一身份口径：调用方必须给出 model_path
   （行内按钮 = 本行 path；警告条 = 当前加载的模型），后端与当前加载比对，
   不一致 409 —— 界面状态过期时不会误卸别的模型。 */
async function unloadModel(path, btn) {
  if (!path) return;
  const b = btn || null;
  if (b && b.disabled) return;
  if (chat.busy) {
    toast("生成中，请先停止再卸载", "err");
    return;
  }
  if (b) {
    b.disabled = true;
    b.textContent = "卸载中…";
  }
  try {
    await api("/v1/models/unload", {
      method: "POST",
      body: JSON.stringify({ model_path: path }),
    });
    chat.model = "";
    await loadModels(); // 状态条与列表回空态（行内按钮随列表重建恢复）
    toast(`推理模型已卸载：${path}`, "ok", 5200);
  } catch (err) {
    if (b) {
      b.disabled = false;
      b.textContent = "卸载";
    }
    toast(`卸载失败：${err.message}`, "err", 7000);
    loadModels().catch(() => {}); // 409 等说明界面状态已过期，刷新对齐
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
  // P7：助手回合带操作行（复制 / 重生成 / 删除）—— 默认隐藏，悬停或键盘聚焦时显形
  if (kind === "assistant") buildActs(wrap, bubble);
  scrollChat();
  return bubble;
}

/* P7：气泡与 chat.convs 的下标对应。
   convs 只存 user/assistant，顺序与 #chat-scroll 里这两类气泡完全一致；
   error 气泡不入 convs，也不在选择器内 —— 所以「第几个气泡」就是「convs 第几条」。 */
function convIndexOf(wrap) {
  return $$("#chat-scroll .msg.user, #chat-scroll .msg.assistant").indexOf(wrap);
}

function buildActs(wrap, bubble) {
  const acts = document.createElement("div");
  acts.className = "msg-acts";
  const add = (label, title, fn) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title;
    b.addEventListener("click", fn);
    acts.appendChild(b);
  };
  add("复制", "复制这条回复", async () => {
    try {
      await navigator.clipboard.writeText(bubble.textContent || "");
      toast("已复制");
    } catch (_) {
      toast("复制失败（浏览器未授权剪贴板）", "err");
    }
  });
  add("重生成", "删掉这一轮及之后的对话，对同一个提问重新生成", () => regenerateAt(wrap));
  add("删除", "删除这一轮问答", () => deleteTurnAt(wrap));
  wrap.appendChild(acts);
}

function deleteTurnAt(wrap) {
  if (chat.busy) {
    toast("生成中，请先停止再删除", "err");
    return;
  }
  const i = convIndexOf(wrap);
  if (i < 0) return;
  const bubbles = $$("#chat-scroll .msg.user, #chat-scroll .msg.assistant");
  if (i - 1 >= 0) bubbles[i - 1].remove(); // 配对的提问
  bubbles[i].remove();
  chat.convs.splice(Math.max(0, i - 1), 2);
  toast("已删除该轮问答");
}

/* 重生成该轮：截断到这一轮之前，再对同一个提问走一遍完整发送流程 */
async function regenerateAt(wrap) {
  if (chat.busy) {
    toast("生成中，请先停止再重生成", "err");
    return;
  }
  const i = convIndexOf(wrap);
  if (i < 0) return;
  const ask = chat.convs[i - 1];
  if (!ask || ask.role !== "user") return;
  const bubbles = $$("#chat-scroll .msg.user, #chat-scroll .msg.assistant");
  for (let k = i - 1; k < bubbles.length; k++) bubbles[k].remove(); // 该轮及其之后全移除
  chat.convs = chat.convs.slice(0, i - 1);
  await doSend(ask.content);
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

function sendChat(e) {
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
  ta.value = "";
  autoGrow(ta);
  doSend(text); // 不 await：submit 立即返回，避免表单被卡住
}

/* P7：从「一条用户输入」开始跑完整一轮。抽出来是为了让「重生成」复用 ——
   它先截断到该轮之前，然后走完全同一条路径。 */
async function doSend(text) {
  chat.convs.push({ role: "user", content: text });
  pushMsg("user", text);
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
  // P8：采样参数重置（默认值与 serve/api.py 的请求默认值对齐）
  $("#chat-reset").addEventListener("click", () => {
    const DEFAULTS = {
      temp: "0.8",
      "chat-topk": "50",
      "chat-topp": "0.9",
      "chat-rep": "1.15",
      "chat-maxtok": "256",
    };
    for (const [id, v] of Object.entries(DEFAULTS)) {
      const el = document.getElementById(id);
      if (el) el.value = v;
    }
    const tv = $("#temp-val");
    if (tv) tv.textContent = DEFAULTS.temp;
    toast("采样参数已重置为默认值");
  });
  loadModels().catch(() => {});
  // 片二：切到推理页时重拉模型列表 —— 两个训练页早就有 tabchange
  // 监听（pretrain.js / posttrain.js），只有这里缺；后果是训练完切过来看不到新产的
  // checkpoint，必须刷新页面。（loadModels 内部已有失败态，这里不用再兜）
  window.addEventListener("tabchange", (e) => {
    if (e.detail === "inference") loadModels().catch(() => {});
  });
}

document.addEventListener("DOMContentLoaded", initChat);

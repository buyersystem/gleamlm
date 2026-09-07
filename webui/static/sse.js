/* SSE 客户端：fetch + ReadableStream + TextDecoder。
   不用 EventSource —— 没有 Last-Event-ID；断线重连由调用方携带 seq 参数续传
   （训练日志流后端按行号 seq 回放）。 */
"use strict";

/**
 * 连接 SSE 端点。handlers: { onData(ev), onExit(ev), onError(err), signal }
 * 事件格式: data: {"type":"log"|"exit"|..., ...}（": ping" 心跳注释行自动忽略）
 * 返回 Promise，流自然结束后 resolve。
 */
async function sseFetch(url, handlers = {}) {
  const { onData = () => {}, onExit = () => {}, onError = () => {}, signal } = handlers;
  let resp;
  try {
    resp = await fetch(url, { signal });
  } catch (err) {
    onError(err);
    return;
  }
  if (!resp.ok || !resp.body) {
    onError(new Error(`SSE 连接失败: HTTP ${resp.status} ${url}`));
    return;
  }
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (err) {
      onError(err); // 连接中断（断网/服务重启）——由调用方决定重连
      return;
    }
    if (chunk.done) break;
    buf += dec.decode(chunk.value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!dataLine) continue; // 心跳 ": ping" 等注释帧
      let ev;
      try {
        ev = JSON.parse(dataLine.slice(5).trim());
      } catch (_) {
        continue;
      }
      onData(ev);
      if (ev.type === "exit") {
        onExit(ev);
        return;
      }
    }
  }
}

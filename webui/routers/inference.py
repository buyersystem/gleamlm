"""
GleamLM WebUI — 推理 router（serve/api.py 逻辑 router 化）。

与 serve/api.py 的关系:
  - serve/api.py 保留为纯推理部署入口（vLLM 场景），逻辑与本节同源;
  - webui 推理 tab 需要"进程内换模型"，因此额外提供 /v1/models/load 与
    /v1/models/status（serve 启动即加载模型，无此需求）。
  - ModelServer 维持模块级单例（对齐 serve/api.py 现有结构），采样逻辑不动。
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.chatml import format_chatml
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config
from hf.hf_config import gleamlm_config_from_core
from hf.hf_model import GleamLMForCausalLM, load_from_checkpoint

_ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(_ROUTER_DIR))


class CompletionRequest(BaseModel):
    model: str = "gleamlm"
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15  # 对齐训练评估默认值
    stop: list[str] | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gleamlm"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15  # 对齐训练评估默认值
    stop: list[str] | None = None
    stream: bool = False


class ModelLoadRequest(BaseModel):
    model_path: str
    tokenizer_path: str = ""


class ModelServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.loaded_path: str = ""

    def load(self, model_path: str, tokenizer_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BBPETokenizer.load(tokenizer_path or DEFAULT_TOKENIZER_PATH)

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        cfg = extract_checkpoint_config(ckpt)
        hf_config = gleamlm_config_from_core(cfg)
        self.model = GleamLMForCausalLM(hf_config).to(self.device)
        missing, unexpected = load_from_checkpoint(self.model, ckpt, strict=True)
        if missing or unexpected:
            print(f"[warn] webui load — missing={missing} unexpected={unexpected}")
        self.model.eval()
        self.loaded_path = os.path.abspath(model_path)

        total = sum(p.numel() for p in self.model.parameters())
        print(f"Server loaded: {total / 1e6:.2f}M on {self.device} <- {model_path}")


server = ModelServer()
_load_lock = asyncio.Lock()
router = APIRouter()


def load_state() -> dict:
    """模型加载状态（推理 tab 轮询，驱动模型菜单与 GPU 互斥提示）。"""
    if server.model is None:
        return {"loaded": False, "model_path": "", "device": "", "params_m": 0}
    total = sum(p.numel() for p in server.model.parameters())
    device = str(server.device)
    return {
        "loaded": True,
        "model_path": server.loaded_path,
        "device": device,
        "params_m": round(total / 1e6, 2),
    }


def _require_model() -> None:
    if server.model is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载 — 请先在推理 tab 的模型列表中选择并加载 checkpoint",
        )


@router.post("/models/load")
async def load_model(req: ModelLoadRequest):
    """加载/切换推理模型（webui 进程内热切换，新训练完成的模型即时可用）。"""
    model_path = req.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT_DIR, model_path)
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"checkpoint 不存在: {req.model_path}")

    async with _load_lock:
        if server.model is not None and server.loaded_path == model_path:
            return {"ok": True, "model": req.model_path, "cached": True}
        await asyncio.to_thread(server.load, model_path, req.tokenizer_path)
    return {"ok": True, "model": req.model_path, "cached": False}


@router.get("/models/status")
async def model_status() -> dict:
    return load_state()


def _check_request(req, prompt_len: int) -> None:
    """参数校验: 负 temperature / 非法 max_tokens / 超上下文长度。"""
    _require_model()
    if req.temperature < 0:
        raise HTTPException(status_code=400, detail="temperature 不能为负")
    if req.max_tokens <= 0 or req.max_tokens > 2048:
        raise HTTPException(status_code=400, detail="max_tokens 需在 (0, 2048] 范围")
    max_pos = server.model.config.max_position_embeddings
    if prompt_len + req.max_tokens > max_pos:
        raise HTTPException(
            status_code=400,
            detail=f"prompt({prompt_len}) + max_tokens({req.max_tokens}) 超过上下文长度 {max_pos}",
        )


def _sample_token(logits: torch.Tensor, params, generated: list[int] | None = None) -> torch.Tensor:
    """单步采样: repetition_penalty → top_k → (T>0: 缩放+top_p+multinomial | T≤0: argmax)。

    top_p (nucleus) 语义与核心库 gleamlm.inference.generator.sample_token 一致:
    对缩放后的概率从最高 token 起累积，只保留累积概率 ≤ top_p 的最小 token 集。
    top_p ≤ 0 或 ≥ 1 时不过滤。"""
    # 与 gleamlm.inference.generator.sample_token 同款重复惩罚（HF 算法）
    penalty = getattr(params, "repetition_penalty", 1.0)
    if penalty != 1.0 and generated:
        for gid in set(generated):
            scores = logits[..., gid]
            logits[..., gid] = torch.where(scores < 0, scores * penalty, scores / penalty)
    if params.top_k > 0:
        vals, _ = logits.topk(params.top_k, dim=-1)
        logits = logits.masked_fill(logits < vals[:, -1:], float("-inf"))
    if params.temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / params.temperature
    top_p = getattr(params, "top_p", 1.0)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()  # 至少保留累积区间的第一个 token
        remove[..., 0] = False
        to_remove = remove.scatter(-1, sorted_indices, remove)
        logits = logits.masked_fill(to_remove, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


def _apply_stop(text: str, stops: list[str] | None) -> str:
    """按 stop 字符串截断文本（支持跨 token 边界拼接后命中）。"""
    if not stops:
        return text
    for s in stops:
        if s and s in text:
            return text.split(s)[0]
    return text


def _stop_ids() -> set[int]:
    """ChatML 训练下模型以 im_end/eos 收尾；serve 需同样识别才不超生成。"""
    tk = server.tokenizer
    ids = {tk.eos_id, tk.im_end_id, tk.pad_id}
    ids.discard(None)
    return ids


def _generate(input_ids: torch.Tensor, params, prompt_len: int) -> list[int]:
    """自回归生成，只返回新增 token（不含 prompt，避免把用户输入当 completion 返回）。"""
    tokens: list[int] = []
    generated: list[int] = input_ids[0].tolist()  # 重复惩罚需看已生成序列
    stop_ids = _stop_ids()
    with torch.no_grad():
        for _ in range(params.max_tokens):
            logits, _, _, _ = server.model.model(input_ids)
            nxt = _sample_token(logits[:, -1, :], params, generated)
            token_id = int(nxt.item())
            if token_id in stop_ids:
                break
            input_ids = torch.cat([input_ids, nxt], dim=-1)
            tokens.append(token_id)
            generated.append(token_id)
            if params.stop:
                text = server.tokenizer.decode(tokens, skip_special=True)
                if _apply_stop(text, params.stop) != text:
                    break
    return tokens


def _step(input_ids: torch.Tensor, params, generated: list[int]) -> tuple[torch.Tensor, int, bool]:
    """单步前向+采样（供流式生成在线程池中执行，避免阻塞事件循环）。
    返回 (下一 token, 其 id, 是否命中终止符 eos/im_end/pad)。"""
    with torch.no_grad():
        logits, _, _, _ = server.model.model(input_ids)
        nxt = _sample_token(logits[:, -1, :], params, generated)
    token_id = int(nxt.item())
    return nxt, token_id, token_id in _stop_ids()


async def _stream(input_ids: torch.Tensor, params, prompt_len: int, chat: bool):
    """SSE 流式响应: 每 token 一个 data 帧，完成后发 [DONE]。

    stop 命中时只发出截断前的增量文本（与非流式 _apply_stop 行为一致）。
    """
    total_decoded = ""
    emitted_len = 0
    generated: list[int] = input_ids[0].tolist()
    for _ in range(params.max_tokens):
        nxt, nxt_id, stop_hit = await asyncio.to_thread(_step, input_ids, params, generated)
        input_ids = torch.cat([input_ids, nxt], dim=-1)
        generated.append(nxt_id)
        chunk = server.tokenizer.decode([nxt_id], skip_special=True)
        total_decoded += chunk
        if params.stop:
            trimmed = _apply_stop(total_decoded, params.stop)
            if trimmed != total_decoded:
                stop_hit = True
                total_decoded = trimmed
        delta = total_decoded[emitted_len:]
        emitted_len = len(total_decoded)
        if delta:
            if chat:
                payload = {"choices": [{"delta": {"content": delta}}]}
            else:
                payload = {"choices": [{"text": delta}]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        if stop_hit:
            break
    yield "data: [DONE]\n\n"


@router.post("/completions")
async def completions(req: CompletionRequest):
    _require_model()  # 未加载时 503（tokenizer 在 _check_request 之前就要用）
    input_ids = torch.tensor(
        [[server.tokenizer.bos_id] + server.tokenizer.encode(req.prompt)], device=server.device
    )
    prompt_len = input_ids.size(1)
    _check_request(req, prompt_len)
    if req.stream:
        return StreamingResponse(
            _stream(input_ids, req, prompt_len, chat=False), media_type="text/event-stream"
        )
    tokens = _generate(input_ids, req, prompt_len)
    text = server.tokenizer.decode(tokens, skip_special=True)
    text = _apply_stop(text, req.stop)
    return {"id": "cmpl-1", "object": "text_completion", "choices": [{"text": text}]}


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    _require_model()
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    # 不自动注入 system：SFT 训练仅 20% 样本带 system（80% 无），
    # 纯 user/assistant 帧更贴合训练主流分布；调用方自带 system 时原样保留
    prompt = format_chatml(messages, add_generation_prompt=True)
    input_ids = torch.tensor([server.tokenizer.encode(prompt, add_bos=False)], device=server.device)
    prompt_len = input_ids.size(1)
    _check_request(req, prompt_len)
    if req.stream:
        return StreamingResponse(
            _stream(input_ids, req, prompt_len, chat=True), media_type="text/event-stream"
        )
    tokens = _generate(input_ids, req, prompt_len)
    text = server.tokenizer.decode(tokens, skip_special=True)
    text = _apply_stop(text, req.stop)
    return {
        "id": "chat-1",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }

"""
GleamLM WebUI — 统一门户入口（唯一入口：uvicorn 起这一个 app）。

预训练 / 后训练 / 推理三 tab 门户（docs/前端面板设计.md）：
  /v1/*    推理（routers/inference.py，OpenAI 兼容 + SSE 流式）
  /api/*   训练遥控器 / 配置管理 / checkpoint 扫描（routers/training.py）
  /static  前端静态资源（index.html / theme.css / *.js）
  /images  运行时图片（背景图 / logo，与 static 平级挂载，对齐 luna-agent）

用法（与 README 的训练/推理命令同环境运行）:
  python webui/main.py                          # 纯面板，模型在推理 tab 内加载
  python webui/main.py --model checkpoints/nano/sft/sft_best.pt   # 预加载推理模型
  python webui/main.py --no-train               # 只挂推理（部署场景）
"""

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webui.routers import inference as inference_router_mod
from webui.routers import training as training_router_mod

_WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_WEBUI_DIR, "static")
_IMAGES_DIR = os.path.join(_WEBUI_DIR, "images")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GleamLM WebUI 统一门户")
    p.add_argument("--model", default="", help="启动即加载的推理 checkpoint（可选）")
    p.add_argument("--tokenizer_path", default="", help="BBPE 分词器目录（缺省内置 12K）")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-train", action="store_true", help="只挂推理 router（纯部署）")
    return p.parse_args()


# 模块级解析（uvicorn.run(app) 直跑; 被 -m uvicorn import 时不认识其参数则回落默认）
try:
    _args = parse_args()
except SystemExit:
    _args = argparse.Namespace(
        model="", tokenizer_path="", port=8080, host="127.0.0.1", no_train=False
    )


def _api_info() -> dict:
    """能力列表（控制 tab 显隐）+ GPU 状态 + 推理模型状态。"""
    info = {
        "tabs": ["pretrain", "posttrain", "inference"],
        "train": True,
        "inference": inference_router_mod.load_state(),
        "gpu": training_router_mod.gpu_info(),
        "webui_dir": os.path.basename(_WEBUI_DIR),
    }
    if _args.no_train:
        info["tabs"] = ["inference"]
        info["train"] = False
    return info


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动即加载推理模型（--model 显式给出时）；未给则推理 tab 内手动加载
    if _args.model:
        await asyncio.to_thread(inference_router_mod.server.load, _args.model, _args.tokenizer_path)
    yield


app = FastAPI(title="GleamLM WebUI", version="0.1.0", lifespan=lifespan)

app.include_router(inference_router_mod.router, prefix="/v1")
if not _args.no_train:
    # --no-train: 省略 training router（纯推理部署场景）
    app.include_router(training_router_mod.router, prefix="/api")

os.makedirs(_STATIC_DIR, exist_ok=True)
os.makedirs(_IMAGES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=_IMAGES_DIR), name="images")


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """静态资源（页面/JS/CSS/图片）一律 no-store，避免浏览器缓存旧文件
    （luna-agent server.py 同款做法）。"""
    response = await call_next(request)
    if request.url.path.startswith(("/images/", "/static/")):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/info")
async def api_info() -> dict:
    return _api_info()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": "webui"}


@app.get("/")
async def root():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host=_args.host, port=_args.port)

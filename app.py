# app.py
from typing import Optional, List, Dict
import time
import requests
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, AnyHttpUrl
from logging.config import dictConfig
import logging, sys

# 如果你有自己的处理器，可以保留；若没有可删掉两处引用
from ai_picture_processor import AIPictureProcessor

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d - %(message)s"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "stream": sys.stdout, "formatter": "default"},
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}
dictConfig(LOGGING_CONFIG)


session = requests.Session()

# ================== FastAPI ==================
app = FastAPI(title="ComfyUI Runner (RunningHub)", version="1.0.0")
aipp = AIPictureProcessor()
logger = logging.getLogger(__name__)

# 允许所有来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],   # 允许所有HTTP方法
    allow_headers=["*"]    # 允许所有请求头
)

@app.middleware("http")
async def log_method(request: Request, call_next):
    print(">>", request.method, request.url.path)
    return await call_next(request)

@app.api_route("/health", methods=["GET", "HEAD", "OPTIONS"])
def health():
    return {"ok": True}

@app.get("/test")
def test():
    logger.info("[test] start")
    aipp.start_job("/root/DiffServer/assets/test/test.jpg")
    return {"ok": True, "msg": "test successful"}

@app.get("/testbatch")
def testbatch():
    logger.info("[testbatch] start")
    aipp.start_batch_job("/root/DiffServer/assets/test/test.jpg", "/root/DiffServer//assets/test/test.json")
    return {"ok": True, "msg": "testbatch successful"}

@app.post("/rhcallback")  # 裁图工作流的回调
async def rhcallback(request: Request):
    logger.info("[callback] received")
    payload = await request.json()
    aipp.on_callback(payload)


# ================== 在文件末尾添加 ==================
if __name__ == "__main__":
    import uvicorn
    # 在代码内写死端口号
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8081,  # 这里写死端口号
        log_level="info"
    )

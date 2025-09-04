# app.py
import logging
from typing import Optional, List, Dict
import time
import requests
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, AnyHttpUrl

# 如果你有自己的处理器，可以保留；若没有可删掉两处引用
from ai_picture_processor import AIPictureProcessor

# 配置根 logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


# ================== HTTP 会话与并发控制 ==================
session = requests.Session()

# ================== FastAPI ==================
app = FastAPI(title="ComfyUI Runner (RunningHub)", version="1.0.0")
aipp = AIPictureProcessor()
logger = logging.getLogger("myapp")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/test")
def test():
    logger.info("[test] start")
    aipp.start_job("./assets/test/test.png")
    return {"ok": True, "msg": "test successful"}

@app.get("/testbatch")
def testbatch():
    logger.info("[testbatch] start")
    aipp.start_batch_job("./assets/test/test_batch.png", "./assets/test/test.json")
    return {"ok": True, "msg": "testbatch successful"}

@app.post("/rh_callback")  # 裁图工作流的回调
async def rh_callback(request: Request):
    logger.info("[callback] received")
    payload = await request.json()
    aipp.on_callback(payload)


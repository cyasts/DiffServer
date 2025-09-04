# ai_client.py
import os
import json
from typing import Dict, Any, List, Optional
import requests

class AIClient:
    """
    基于“文件路径”的 RunningHub 调用：
      - upload_image_path(path) -> fileName
      - create_full_task_from_path(path, prompt) -> taskId
      - create_patch_task_from_path(path, prompt) -> taskId
      - download_from_callback(payload, save_dir) -> {"paths":[...]}
    """

    def __init__(self):
        self.api_base = "https://www.runninghub.cn"
        self.api_key = "94fb0c59bdac4af79977df8e0fee71e1"
        self.workflow_single = "1958698927740579842"
        self.workflow_batch = "1963139722744926209"
        self.webhook_url = "http://47.94.138.54:10080/rh_callback"  # 你的回调地址

    # ------------- 上传文件（路径）-------------
    def upload_image_path(self, image_path: str) -> str:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        r = requests.post(
            f"{self.api_base}/task/openapi/upload",
            data={"apiKey": self.api_key, "fileType": "image"},
            files={"file": open(image_path, "rb")}
        )

        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != 0:
            raise RuntimeError(f"upload failed: {resp}")
        return resp["data"]["fileName"]  # e.g. api/xxxx.png 或 .jpg

    def _upload_image_byte(self, img_bytes: bytes) -> str:
        r = requests.post(
            f"{self.api_base}/task/openapi/upload",
            data={"apiKey": self.api_key, "fileType": "image"},
            files={"file": ("image.png", img_bytes, "image/png")}
        )
        r.raise_for_status()
        return r.json()["data"]["fileName"]  # e.g. api/xxxx.png 或 .jpg

    # ------------- 创建任务 -------------
    def _create_task(self, workflow_id: str, node_info_list: List[Dict[str, Any]]) -> str:
        payload = {
            "apiKey": self.api_key,
            "workflowId": workflow_id,
            "nodeInfoList": node_info_list,
            "webhookUrl": self.webhook_url
        }
        headers = {"Host": "www.runninghub.cn", "Content-Type": "application/json"}

        response = requests.request("POST", f"{self.api_base}/task/openapi/create", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["data"]["taskId"]

    # ------------- 对外：整图 -------------
    def run_task(self, image_path: str) -> str:
        file_name = self.upload_image_path(image_path)
        node_info_list = [
            {"nodeId": 200, "fieldName": "image", "fieldValue": file_name}
        ]
        return self._create_task(self.workflow_single, node_info)

    # ------------- 对外：单 patch（与整图同构）-------------
    def run_batch_task(self, image_bytes: bytes, prompt: str) -> str:
        file_name = self._upload_image_byte(image_bytes)
        node_info = [
            {"nodeId": "18",  "fieldName": "self.prompt",  "fieldValue": prompt},
            {"nodeId": "19", "fieldName": "image", "fieldValue": file_name},
        ]
        return self._create_task(self.workflow_batch, node_info)

    # ------------- 回调下载（保存到目录）-------------
    def download_from_callback(self, url: str):
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.content

# pipeline.py
from __future__ import annotations
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import os

from ai_client import AIClient
from image_utils import crop_patches_aabb_from_paths, save_image  # -> [{'part_id','prompt','bbox','patch_bytes'}, ...]

# --------- 元数据 ---------
@dataclass
class TaskMeta:
    task_id: str
    output: str
    job_id: str
    feather: bool = False

@dataclass
class JobTracker:
    expected: int
    done: int = 0
    parts_seen: set = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

# --------- 主类（极简两入口 + 信号量 + 回调聚合） ---------
class AIPictureProcessor:
    def __init__(self,
                 *,
                 max_inflight: int = 30,          # 服务器并发上限（每次调用AI前 acquire）
                 submit_workers: int = 8,         # 提交/裁剪等工作线程
                 callback_workers: int = 8,       # 回调下载/落盘线程
                 on_job_complete: Optional[callable] = None,  # 全部完成后回调(DB等)
                 ):
        self.ai = AIClient()
        self.inflight_sem = threading.BoundedSemaphore(max_inflight)

        self.submit_pool = ThreadPoolExecutor(max_workers=submit_workers)
        self.callback_pool = ThreadPoolExecutor(max_workers=callback_workers)

        self._lock = threading.Lock()
        self.task_map: Dict[str, TaskMeta] = {}   # taskId -> TaskMeta
        self._released: set[str] = set()          # 已释放过信号量的 taskId
        self.jobs: Dict[str, JobTracker] = {}     # job_id -> JobTracker

        self._on_job_complete = on_job_complete or (lambda job_id, tracker: None)

    # --------- 工具 ---------
    def _start_tracker(self, expected: int):
        with self._lock:
            job = f"{uuid.uuid4().hex}"
            self.jobs[job] = JobTracker(expected=expected)
            return job

    # --------- 对外入口：单图 ---------
    def start_job(self, img: str):
        self.submit_pool.submit(self.run_job, img)

    # 在线程池里实际执行：单张图 -> 调AI
    def run_job(self, img: str):
        # 创建任务id
        job_id = self._start_tracker(expected=1)

        # 严格限流：每次真正调用AI前 acquire
        self.inflight_sem.acquire()
        try:
            task_id = self.ai.run_task(img)
            #获取img的文件格式
            out = os.path.dirname(img) + "proto" +  os.path.splitext(img)[-1]
            with self._lock:
                self.task_map[task_id] = TaskMeta(task_id=task_id, output=out)
            return {"job_id": job_id, "task_id": task_id}
        except Exception:
            # 创建失败必须归还
            try: self.inflight_sem.release()
            except Exception: pass
            raise

    # --------- 对外入口：批处理（大图 + 配置）---------
    def start_batch_job(self, img: str, config: str):
        self.submit_pool.submit(self.run_batch_job, img, config)

    # 在线程池里实际执行：裁剪 -> 逐patch调用AI（简单顺序版，每次都 acquire）
    def run_batch_job(self, img: str, config: str,
                      coord_origin: str = "top-left") -> Dict[str, Any]:

        # 创建任务id
        job_id = self._start_tracker(expected=1)

        # 分片，并截取出所有的小图像
        patches = crop_patches_aabb_from_paths(img, config, origin=coord_origin)
        n = len(patches)
        self._start_tracker(expected=n)
        if n == 0:
            # 没有可用区域
            return {"job_id": job_id, "parts": []}

        parts_meta: List[Dict[str, Any]] = []

        for p in patches:
            self.inflight_sem.acquire()
            try:
                task = self.ai.run_batch_task(p["patch_bytes"], p.get("prompt", ""))
                out = os.path.dirname(img) + "region" +  p["part_id"] + ".png"
                with self._lock:
                    self.task_map[task] = TaskMeta(task_id=task, output=out, part_id=p["part_id"], feather=True)
                parts_meta.append({
                    "part_id": p["part_id"],
                    "task_id": task,
                    "bbox": p["bbox"],
                })
            except Exception:
                try: self.inflight_sem.release()
                except Exception: pass
                raise

    # --------- 回调入口：释放并发 + 下载 + 聚合完成 ---------
    def on_callback(self, payload: Dict[str, Any]):
        """
        服务器回调到达时调用：
          - 释放一次 inflight_sem（只释放一次）
          - 丢到 callback_exec 下载结果
          - 聚合进度，全部完成后触发 on_job_complete
        """
        task_id = str(payload.get("taskId") or "")
        print(f"[callback] task_id={task_id}")

        if task_id:
            with self._lock:
                meta = self.task_map.get(task_id)
                if task_id not in self._released:
                    self._released.add(task_id)
                    try: self.inflight_sem.release()
                    except Exception: pass

        # 下载/落盘放回回调线程池，避免阻塞 HTTP 回调线程
        self.callback_exec.submit(self._run_callback, payload, meta)

    def _run_callback(self, payload: Dict[str, Any], meta: Optional[TaskMeta]):
        if not meta:
            return
        
        job_id = meta.job_id
        task_id = meta.task_id
        out = meta.output
        feather = meta.feather
        with self._lock:
            self.task_map.pop(task_id, None)

        # 1) 获取图片内容
        ed = payload.get("eventData")
        if not ed or ed is None:
            return
        if ed.get("code") != 0:
            return
        url = ed.get("data").get("fileUrl")

        try:
            piece = self.ai.download_from_callback(url)
        except Exception as e:
            print(f"[callback_io] error: {e}")

        save_image(piece, out, feather)

        # 2) 聚合（需要 job_id 才能统计）
        with self._lock:
            tracker = self.jobs.get(job_id)
        if not tracker:
            return

        with tracker.lock:
            # 去重（防回调重放）
            if task_id is not None and task_id in tracker.parts_seen:
                return
            if task_id is not None:
                tracker.parts_seen.add(task_id)

            tracker.done += 1

            done, expected = tracker.done, tracker.expected

        # 3) 全部完成 -> 触发 on_job_complete 一次，并清理
        if done >= expected:
            with self._lock:
                t2 = self.jobs.pop(job_id, None)
            if t2 is not None:
                try:
                    self._on_job_complete(job_id, t2)
                except Exception as e:
                    print(f"[on_job_complete] error: {e}")

    # --------- 关闭线程池 ---------
    def shutdown(self, wait: bool = True):
        self.submit_exec.shutdown(wait=wait)
        self.callback_exec.shutdown(wait=wait)

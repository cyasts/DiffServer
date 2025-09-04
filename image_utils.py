# image_ops.py
import os
import json
from typing import List, Dict, Any, Literal
import numpy as np
import cv2

CoordOrigin = Literal["top-left", "bottom-left"]

def decode_image_from_path(image_path: str) -> np.ndarray:
    """path -> np.ndarray(BGR[A])"""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"decode image failed: {image_path}")
    return img

def parse_config_from_path(config_path: str) -> List[Dict[str, Any]]:
    """读取 JSON 配置文件，返回 differences 数组"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("differences", [])

def _norm_points_to_pixels(points, W: int, H: int, origin: CoordOrigin) -> np.ndarray:
    """归一化坐标→像素；支持左上/左下原点"""
    out = []
    for p in points:
        x = float(p["x"]) * (W - 1)
        y_norm = float(p["y"])
        if origin == "bottom-left":
            y = (1.0 - y_norm) * (H - 1)
        else:
            y = y_norm * (H - 1)
        out.append([x, y])
    return np.array(out, dtype=np.float32)

def crop_patches_aabb_from_paths(image_path: str, config_path: str,
                                 origin: CoordOrigin = "bottom-left") -> List[Dict[str, Any]]:
    """
    以“最小包围矩形(AABB)”裁剪各区域，返回列表：
    [{
      "part_id": "0",
      "prompt": "换成惊讶的表情",
      "bbox": [x_min, y_min, x_max, y_max],
      "patch_bytes": PNG字节
    }, ...]
    """
    img = decode_image_from_path(image_path)
    H, W = img.shape[:2]
    diffs = parse_config_from_path(config_path)
    out: List[Dict[str, Any]] = []

    for idx, d in enumerate(diffs):
        pts = d.get("points", [])
        if len(pts) < 4:
            continue

        quad = _norm_points_to_pixels(pts, W, H, origin)
        x_min = max(0, int(np.floor(np.min(quad[:, 0]))))
        x_max = min(W, int(np.ceil (np.max(quad[:, 0]))))
        y_min = max(0, int(np.floor(np.min(quad[:, 1]))))
        y_max = min(H, int(np.ceil (np.max(quad[:, 1]))))
        if x_max <= x_min or y_max <= y_min:
            continue

        patch = img[y_min:y_max, x_min:x_max]
        ok, buf = cv2.imencode(".png", patch)  # 用PNG保留透明度
        if not ok:
            continue

        out.append({
            "part_id": str(idx),
            "prompt": d.get("text", ""),
            "bbox": [x_min, y_min, x_max, y_max],
            "patch_bytes": buf,
        })

    return out

def feather_image(  img: np.ndarray,
                    feather: int = 8,
                    shrink: int = 0,
                    gamma: float = 1.0) -> np.ndarray:
    """
    对整张小图四边向内羽化
    - img: np.ndarray，支持 GRAY/BGR/BGRA
    - feather: 羽化半径（像素）
    - shrink: 先整体向内收缩若干像素（腐蚀），可减轻边缘硬痕
    - gamma: 透明度曲线，<1更软、>1更硬，1为线性
    """
    if img is None or img.size == 0:
        raise ValueError("empty image")
    H, W = img.shape[:2]

    # --- 构造有边界的 mask：边界为0，内部为255 ---
    mask = np.zeros((H, W), np.uint8)
    if H > 2 and W > 2:
        mask[1:H-1, 1:W-1] = 255
    else:
        mask[:, :] = 255  # 图太小，直接全不羽化

    # 可选整体收缩（向内腐蚀）
    if shrink > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*shrink+1, 2*shrink+1))
        mask = cv2.erode(mask, k)

    # --- 距离变换生成 0..1 的 alpha ---
    if feather > 0:
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(dist / float(feather), 0.0, 1.0).astype(np.float32)
    else:
        alpha = (mask > 0).astype(np.float32)

    if gamma is not None and gamma > 0 and gamma != 1.0:
        alpha = np.power(alpha, gamma)

    # --- 组 BGRA，并与原 alpha 相乘（若存在） ---
    if img.ndim == 2:
        bgra = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        a0 = np.ones((H, W), np.float32)
    elif img.shape[2] == 3:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        a0 = np.ones((H, W), np.float32)
    elif img.shape[2] == 4:
        bgra = img.copy()
        a0 = bgra[:, :, 3].astype(np.float32) / 255.0
    else:
        raise ValueError("Unsupported image shape/channels")

    a = np.clip(alpha * a0, 0.0, 1.0)
    out = bgra.copy()
    out[:, :, 3] = (a * 255.0 + 0.5).astype(np.uint8)

    return out

def save_image(path: str, img: np.ndarray, feature: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if feature:
        img = feather_image(img, feather=feature)

    if not cv2.imwrite(path, img):
        raise RuntimeError(f"failed to save image: {path}")


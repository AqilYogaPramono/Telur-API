from pathlib import Path
from typing import Any

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_YOLO_MODEL_PATH = _REPO_ROOT / "models" / "detection_egg_v1.pt"
_yolo_model = None


def _load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not _YOLO_MODEL_PATH.is_file():
            raise ValueError("YOLO model file was not found at models/detection_egg_v1.pt.")
        from ultralytics import YOLO

        _yolo_model = YOLO(str(_YOLO_MODEL_PATH))
    return _yolo_model


def detect_egg_boxes_xyxy(image_bgr: np.ndarray) -> np.ndarray:
    model = _load_yolo_model()
    result = model(image_bgr)[0]
    if result.boxes is None or len(result.boxes.xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return result.boxes.xyxy.cpu().numpy()


def _rects_overlap(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    return not (rect_a[2] < rect_b[0] or rect_a[0] > rect_b[2] or rect_a[3] < rect_b[1] or rect_a[1] > rect_b[3])


def render_detection_overlay_jpeg(
    image_bgr: np.ndarray,
    boxes_xyxy: list[tuple[int, int, int, int]],
    per_egg: list[dict[str, Any]],
) -> bytes | None:
    overlay = image_bgr.copy()
    height_img, width_img = overlay.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = 1.0
    min_scale = 0.45
    thickness = 2
    pad = 8
    gap = 6
    placed: list[tuple[int, int, int, int]] = []

    for (x1, y1, x2, y2), item in zip(boxes_xyxy, per_egg):
        if item.get("is_mati"):
            color = (0, 140, 255)
        elif item["is_fertile"]:
            color = (0, 200, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        egg_number = item["egg_number"]
        text = f"Egg {egg_number}"
        scale = base_scale
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        max_text_width = max(20, width_img - (pad * 2))
        while text_width > max_text_width and scale > min_scale:
            scale = max(min_scale, scale - 0.05)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

        def text_rect(text_x: int, baseline_y: int) -> tuple[int, int, int, int]:
            top = baseline_y - text_height - pad
            left = text_x - pad
            right = text_x + text_width + pad
            bottom = baseline_y + baseline + pad
            return (left, top, right, bottom)

        def fits_image(rect: tuple[int, int, int, int]) -> bool:
            left, top, right, bottom = rect
            return top >= 0 and left >= 0 and right <= width_img and bottom <= height_img

        candidates: list[tuple[int, int]] = []
        text_x0 = max(pad, min(x1, width_img - text_width - pad))
        candidates.append((text_x0, y1 - gap))
        candidates.append((text_x0, y2 + text_height + baseline + gap))
        for delta_x in (40, 80, 120, -40, -80, 160, -120):
            text_x = max(pad, min(x1 + delta_x, width_img - text_width - pad))
            candidates.append((text_x, y1 - gap))
            candidates.append((text_x, y2 + text_height + baseline + gap))
        for step in range(1, 8):
            candidates.append((text_x0, y1 - gap - step * (text_height + baseline + pad * 2)))
            candidates.append((text_x0, y2 + text_height + baseline + gap + step * (text_height + baseline + pad * 2)))

        chosen: tuple[int, int, tuple[int, int, int, int]] | None = None
        for text_x, baseline_y in candidates:
            rect = text_rect(text_x, baseline_y)
            if not fits_image(rect):
                continue
            if any(_rects_overlap(rect, prev) for prev in placed):
                continue
            chosen = (text_x, baseline_y, rect)
            break

        if chosen is None:
            text_x = max(pad, min(x1, width_img - text_width - pad))
            baseline_y = min(y2 + text_height + baseline + gap, height_img - baseline - pad)
            rect = text_rect(text_x, baseline_y)
        else:
            text_x, baseline_y, rect = chosen

        placed.append(rect)
        cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
        cv2.putText(
            overlay,
            text,
            (text_x, baseline_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    ok, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    return buffer.tobytes()

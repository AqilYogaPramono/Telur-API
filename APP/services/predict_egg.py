from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import cv2
import numpy as np

from APP.services.crop_image import crop_bgr_region
from APP.services.egg_classification import EggCNNClassification, classify_egg_from_image_bytes
from APP.services.egg_detection import detect_egg_boxes_xyxy, render_detection_overlay_jpeg
from APP.services.encode_png import rgba_uint8_to_png_bytes
from APP.services.remove_background import remove_background


@dataclass(frozen=True)
class EggCropPreview:
    egg_index: int
    classification: EggCNNClassification
    png_bytes: bytes


@dataclass(frozen=True)
class YoloCropAnalysisResult:
    overlay_filename: str
    overlay_jpeg_bytes: bytes | None
    egg_count: int
    fertile_count: int
    infertile_count: int
    dead_count: int
    crop_previews: tuple[EggCropPreview, ...]


def analyze_egg_yolo_crop_sync(image_bytes: bytes) -> YoloCropAnalysisResult:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Invalid image file.")

    boxes = detect_egg_boxes_xyxy(image_bgr)
    valid_boxes: list[tuple[int, int, int, int]] = []
    overlay_items: list[dict[str, object]] = []
    previews: list[EggCropPreview] = []
    fertile_count = 0
    infertile_count = 0
    dead_count = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = crop_bgr_region(image_bgr, x1, y1, x2, y2)
        if crop.size == 0:
            continue
        rgba_cutout = remove_background(crop)
        png_bytes = rgba_uint8_to_png_bytes(rgba_cutout)
        if png_bytes is None:
            continue
        egg_index = len(previews) + 1
        cnn = classify_egg_from_image_bytes(png_bytes)
        if cnn.is_fertile:
            fertile_count += 1
        elif cnn.is_mati:
            dead_count += 1
        else:
            infertile_count += 1
        previews.append(EggCropPreview(egg_index=egg_index, classification=cnn, png_bytes=png_bytes))
        valid_boxes.append((x1, y1, x2, y2))
        overlay_items.append({"egg_number": egg_index, "is_fertile": cnn.is_fertile, "is_mati": cnn.is_mati})

    overlay_filename = f"{uuid4().hex}.jpg"
    overlay_bytes = render_detection_overlay_jpeg(image_bgr, valid_boxes, overlay_items)
    return YoloCropAnalysisResult(
        overlay_filename=overlay_filename,
        overlay_jpeg_bytes=overlay_bytes,
        egg_count=len(previews),
        fertile_count=fertile_count,
        infertile_count=infertile_count,
        dead_count=dead_count,
        crop_previews=tuple(previews),
    )


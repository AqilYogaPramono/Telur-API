from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from APP.services.crop_image import crop_bgr_region
from APP.services.egg_classification_experiment import classify_egg_from_image_bytes
from APP.services.egg_detection_experiment import detect_egg_boxes_xyxy, render_detection_overlay_jpeg
from APP.services.encode_png import rgba_uint8_to_png_bytes
from APP.services.remove_background import remove_background
from APP.utils.upload_validation import safe_label_for_filename


@dataclass(frozen=True)
class EggClassificationRow:
    egg_index: int
    classification_label: str
    confidence_score: float


@dataclass(frozen=True)
class YoloCropExperimentResult:
    images_detection: str
    egg_count: int
    fertile_count: int
    infertile_count: int
    dead_count: int
    egg_classifications: tuple[EggClassificationRow, ...]


def run_yolo_crop_experiment_sync(
    image_bytes: bytes,
    _upload_filename: str,
    public_dir: Path,
) -> YoloCropExperimentResult:
    public_dir.mkdir(parents=True, exist_ok=True)
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Invalid image file.")

    overlay_name = f"{uuid4().hex}.jpg"

    boxes = detect_egg_boxes_xyxy(image_bgr)
    valid_boxes: list[tuple[int, int, int, int]] = []
    overlay_items: list[dict[str, object]] = []
    classification_rows: list[EggClassificationRow] = []
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
        egg_index = len(classification_rows) + 1
        cnn = classify_egg_from_image_bytes(png_bytes)
        if cnn.is_fertile:
            fertile_count += 1
        elif cnn.is_mati:
            dead_count += 1
        else:
            infertile_count += 1
        classification_rows.append(
            EggClassificationRow(
                egg_index=egg_index,
                classification_label=cnn.classification_label,
                confidence_score=cnn.confidence_score,
            )
        )
        valid_boxes.append((x1, y1, x2, y2))
        overlay_items.append(
            {
                "egg_number": egg_index,
                "is_fertile": cnn.is_fertile,
                "is_mati": cnn.is_mati,
            }
        )
        crop_filename = f"{safe_label_for_filename(cnn.classification_label)} - egg {egg_index}.png"
        (public_dir / crop_filename).write_bytes(png_bytes)

    overlay_bytes = render_detection_overlay_jpeg(image_bgr, valid_boxes, overlay_items)
    if overlay_bytes is not None:
        (public_dir / overlay_name).write_bytes(overlay_bytes)

    return YoloCropExperimentResult(
        images_detection=overlay_name,
        egg_count=len(classification_rows),
        fertile_count=fertile_count,
        infertile_count=infertile_count,
        dead_count=dead_count,
        egg_classifications=tuple(classification_rows),
    )

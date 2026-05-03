import os
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEMPLATE_ROOT = _REPO_ROOT / "models" / "Dataset"

_templates_by_day: dict[int, list[np.ndarray]] | None = None


def _resolve_template_root() -> Path:
    custom = os.environ.get("EGG_TEMPLATE_MATCHING_ROOT", "").strip()
    if custom:
        return Path(custom)
    return _DEFAULT_TEMPLATE_ROOT


def _normalize_grayscale(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def _load_templates_per_day(base_root: Path) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    for day in range(1, 11):
        collected: list[np.ndarray] = []
        day_dir = base_root / f"hari ke-{day}" / "B525"
        if not day_dir.is_dir():
            continue
        for root, _, files in os.walk(day_dir):
            for name in files:
                lower = name.lower()
                if not lower.endswith((".jpg", ".jpeg", ".png")):
                    continue
                path = Path(root) / name
                img = cv2.imread(str(path))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = _normalize_grayscale(gray)
                collected.append(gray)
        if collected:
            out[day] = collected
    return out


def ensure_templates_loaded() -> None:
    global _templates_by_day
    if _templates_by_day is not None:
        return
    root = _resolve_template_root()
    if not root.is_dir():
        print(f"[egg templates] Directory not found at {root}. Template matching disabled.")
        _templates_by_day = {}
        return
    _templates_by_day = _load_templates_per_day(root)
    total = sum(len(v) for v in _templates_by_day.values())
    day_count = len(_templates_by_day)
    if total == 0:
        print(f"[egg templates] No images under {root}. Template matching disabled.")
    else:
        print(
            f"[egg templates] All templates loaded: {total} images across {day_count} day folders (root {root})."
        )


def get_templates_by_day() -> dict[int, list[np.ndarray]]:
    if _templates_by_day is None:
        ensure_templates_loaded()
    return _templates_by_day if _templates_by_day is not None else {}


def templates_available() -> bool:
    return bool(get_templates_by_day())


def match_best_template_in_crop(crop_bgr: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float]:
    templates = get_templates_by_day()
    if not templates or crop_bgr.size == 0:
        return None, -1.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = _normalize_grayscale(gray)
    best_score = -1.0
    best_box: tuple[int, int, int, int] | None = None
    for temps in templates.values():
        for temp in temps:
            th, tw = temp.shape[:2]
            if gray.shape[0] < th or gray.shape[1] < tw:
                continue
            res = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                x, y = max_loc
                best_score = max_val
                best_box = (x, y, x + tw, y + th)
    return best_box, best_score

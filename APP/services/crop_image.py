import numpy as np


def crop_bgr_region(image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return image_bgr[y1:y2, x1:x2].copy()

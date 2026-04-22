import cv2
import numpy as np
from rembg import remove


def remove_background(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output = remove(image_rgb)
    if not isinstance(output, np.ndarray):
        output = np.asarray(output)
    if output.ndim == 3 and output.shape[2] == 4:
        return output.copy()
    if output.ndim == 3 and output.shape[2] == 3:
        height, width = output.shape[:2]
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        return np.concatenate([output, alpha], axis=2)
    raise ValueError("Unexpected rembg output shape.")

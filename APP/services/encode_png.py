import cv2
import numpy as np


def rgba_uint8_to_png_bytes(rgba: np.ndarray) -> bytes | None:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected an RGBA image with shape HxWx4.")
    array = np.asarray(rgba)
    red, green, blue, alpha = cv2.split(array)
    bgra = cv2.merge([blue, green, red, alpha])
    ok, buffer = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        return None
    return buffer.tobytes()

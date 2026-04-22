import cv2
import numpy as np
from rembg import remove


def _refine_alpha_mask(
    alpha: np.ndarray,
    *,
    alpha_threshold: int = 20,
    erode_iterations: int = 1,
    blur_kernel_size: int = 3,
) -> np.ndarray:
    mask = np.asarray(alpha).astype(np.uint8, copy=False)
    if alpha_threshold > 0:
        mask = np.where(mask < alpha_threshold, 0, mask).astype(np.uint8)
    if erode_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_iterations)
    if blur_kernel_size and blur_kernel_size > 1:
        k = int(blur_kernel_size)
        if k % 2 == 0:
            k += 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def remove_background(
    image_bgr: np.ndarray,
    *,
    refine_edges: bool = True,
    alpha_threshold: int = 20,
    erode_iterations: int = 1,
    blur_kernel_size: int = 3,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output = remove(image_rgb)
    if not isinstance(output, np.ndarray):
        output = np.asarray(output)
    if output.ndim == 3 and output.shape[2] == 4:
        rgba = output.copy()
        if refine_edges:
            rgba[:, :, 3] = _refine_alpha_mask(
                rgba[:, :, 3],
                alpha_threshold=alpha_threshold,
                erode_iterations=erode_iterations,
                blur_kernel_size=blur_kernel_size,
            )
        return rgba
    if output.ndim == 3 and output.shape[2] == 3:
        height, width = output.shape[:2]
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        return np.concatenate([output, alpha], axis=2)
    raise ValueError("Unexpected rembg output shape.")

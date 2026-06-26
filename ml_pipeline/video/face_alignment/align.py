"""Face alignment utilities."""

from __future__ import annotations

import numpy as np


def align_face(image: np.ndarray, bbox: tuple[float, float, float, float], size: int = 256) -> np.ndarray:
    """Crop and resize face region."""
    import cv2

    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w_img, int(x + w)), min(h_img, int(y + h))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size))

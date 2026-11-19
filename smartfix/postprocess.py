"""Q4 - From anomaly map to defect zones (mask, red contours, % of surface)."""
import cv2
import numpy as np


def clean_mask(mask: np.ndarray, kernel_size: int = 5, min_area: int = 30) -> np.ndarray:
    """Morphological OPEN (removes specks) then CLOSE (fills holes), then drop
    connected components smaller than min_area pixels."""
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def defect_mask(amap: np.ndarray, pixel_threshold: float, **clean_kwargs) -> np.ndarray:
    return clean_mask(amap > pixel_threshold, **clean_kwargs)


def find_contours(mask: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def draw_contours(image_bgr: np.ndarray, contours: list, mask_size: int,
                  thickness: int = 3) -> np.ndarray:
    """Draw contours found on a mask_size x mask_size mask onto the ORIGINAL
    image (any size) in red. Coordinates are rescaled."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    sx, sy = w / mask_size, h / mask_size
    scaled = [(c.astype(np.float32) * [sx, sy]).astype(np.int32) for c in contours]
    cv2.drawContours(out, scaled, -1, (0, 0, 255), thickness)
    return out


def object_mask_from_mean(mean_img: np.ndarray, thresh: float = 0.08) -> np.ndarray:
    """Rough mask of the part itself from the mean normal image (part brighter
    than the dark MVTec background). Used so % is relative to the part."""
    gray = mean_img.max(axis=2)
    m = (gray > thresh).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def defect_percentage(mask: np.ndarray, object_mask: np.ndarray | None = None) -> float:
    if object_mask is None or object_mask.sum() == 0:
        return float(mask.mean() * 100)
    return float((mask & object_mask).sum() / object_mask.sum() * 100)


def heatmap_overlay(image_bgr: np.ndarray, amap: np.ndarray, vmax: float,
                    alpha: float = 0.5) -> np.ndarray:
    """Colour anomaly map (JET) blended on the image, resized to the image."""
    h, w = image_bgr.shape[:2]
    a = np.clip(amap / max(vmax, 1e-6), 0, 1)
    a = cv2.resize((a * 255).astype(np.uint8), (w, h))
    heat = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heat, alpha, 0)

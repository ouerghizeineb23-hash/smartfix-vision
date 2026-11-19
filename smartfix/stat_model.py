"""Q2 - Method A: per-pixel statistical reference.

For each pixel and channel we learn the mean and standard deviation over the
normal (defect-free) training images. A test pixel is anomalous when it is far
from that mean, measured in standard deviations (a z-score):

    z(x, y, c) = |I(x, y, c) - mean(x, y, c)| / (std(x, y, c) + eps)
    amap(x, y) = GaussianBlur( max_c z(x, y, c) )

The image score is the mean of the top 1 % of the map (more robust than the max).
Thresholds are chosen on the VALIDATION normal images, never on the test set.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StatModel:
    mean: np.ndarray                    # H x W x 3
    std: np.ndarray                     # H x W x 3
    eps: float = 0.01                   # avoids dividing by ~0 on a flat background
    blur_sigma: float = 4.0             # smooths isolated noisy pixels
    image_threshold: float | None = None
    pixel_threshold: float | None = None

    # ---------- training ----------
    @classmethod
    def fit(cls, train: np.ndarray, **kwargs) -> "StatModel":
        """train: N x H x W x 3 float images of NORMAL parts."""
        return cls(mean=train.mean(axis=0), std=train.std(axis=0), **kwargs)

    # ---------- inference ----------
    def anomaly_map(self, img: np.ndarray) -> np.ndarray:
        z = np.abs(img - self.mean) / (self.std + self.eps)
        amap = z.max(axis=2).astype(np.float32)
        if self.blur_sigma > 0:
            amap = cv2.GaussianBlur(amap, (0, 0), self.blur_sigma)
        return amap

    def anomaly_maps(self, images: np.ndarray) -> np.ndarray:
        return np.stack([self.anomaly_map(im) for im in images])

    @staticmethod
    def image_score(amap: np.ndarray, top_ratio: float = 0.01) -> float:
        flat = amap.ravel()
        k = max(1, int(len(flat) * top_ratio))
        return float(np.partition(flat, -k)[-k:].mean())

    # ---------- thresholds (on validation normals only) ----------
    def calibrate(self, val: np.ndarray, image_pct: float = 99.0,
                  pixel_pct: float = 99.9) -> None:
        """image_pct: percentile of the val image scores -> about (100 - pct) %
        false alarms on normal parts. pixel_pct: same idea for pixels."""
        maps = self.anomaly_maps(val)
        scores = [self.image_score(m) for m in maps]
        self.image_threshold = float(np.percentile(scores, image_pct))
        self.pixel_threshold = float(np.percentile(maps, pixel_pct))

    # ---------- save / load ----------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, mean=self.mean, std=self.std, eps=self.eps,
            blur_sigma=self.blur_sigma,
            image_threshold=np.nan if self.image_threshold is None else self.image_threshold,
            pixel_threshold=np.nan if self.pixel_threshold is None else self.pixel_threshold,
        )

    @classmethod
    def load(cls, path: Path) -> "StatModel":
        d = np.load(path)

        def opt(v):
            v = float(v)
            return None if np.isnan(v) else v

        return cls(mean=d["mean"], std=d["std"], eps=float(d["eps"]),
                   blur_sigma=float(d["blur_sigma"]),
                   image_threshold=opt(d["image_threshold"]),
                   pixel_threshold=opt(d["pixel_threshold"]))

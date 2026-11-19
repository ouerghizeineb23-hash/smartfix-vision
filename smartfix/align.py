"""Q1 - Alignment check and (optional) registration.

TODO (week 2):
  1. mean_image(train) -> look at it: sharp = aligned, blurry = not aligned.
  2. measure_shifts(train, ref) -> (dx, dy) per image; plot a histogram
     for the report.
  3. If shifts > ~2-3 px: implement register(img, ref) with
     cv2.findTransformECC (cv2.MOTION_EUCLIDEAN) + cv2.warpAffine,
     and apply it to train AND test images.
"""
import cv2
import numpy as np


def mean_image(images: np.ndarray) -> np.ndarray:
    return images.mean(axis=0)


def measure_shifts(images: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref_g = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_RGB2GRAY)
    shifts = []
    for img in images:
        g = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        (dx, dy), _ = cv2.phaseCorrelate(ref_g, g)
        shifts.append((dx, dy))
    return np.array(shifts)


def register(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO week 2 - only if measure_shifts shows misalignment")

"""Q1 - Loading, preprocessing and train/val/test split.

Every image (train, val, test, app upload) MUST go through `preprocess`,
so that the model always sees images prepared the same way.
"""
from pathlib import Path

import cv2
import numpy as np

from .config import CATEGORY, DATA_DIR, IMG_SIZE, SEED, VAL_RATIO

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def category_dir(category: str = CATEGORY) -> Path:
    d = DATA_DIR / category
    if not d.exists():
        raise FileNotFoundError(
            f"{d} not found. Download '{category}.tar.xz' from the MVTec AD "
            f"website and extract it into {DATA_DIR}."
        )
    return d


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def preprocess(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """BGR uint8 (any size) -> RGB float32 in [0, 1], size x size."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def load_image(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {path}")
    return preprocess(img, size)


def load_mask(path: Path | None, size: int = IMG_SIZE) -> np.ndarray:
    """Ground-truth mask as bool (size x size). None -> all False (good image)."""
    if path is None:
        return np.zeros((size, size), dtype=bool)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Cannot read mask {path}")
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return m > 127


def load_train_val(category: str = CATEGORY, val_ratio: float = VAL_RATIO,
                   seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Normal images only, split into (train, val).

    The val set is used to choose thresholds - never the test set.
    """
    paths = list_images(category_dir(category) / "train" / "good")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    n_val = int(round(len(paths) * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = np.stack([load_image(paths[i]) for i in sorted(train_idx)])
    val = np.stack([load_image(paths[i]) for i in sorted(val_idx)])
    return train, val


def load_test(category: str = CATEGORY):
    """Returns (images N x H x W x 3, labels N [0=good, 1=defect],
    masks N x H x W bool, defect_types list[str], paths list[Path])."""
    root = category_dir(category)
    images, labels, masks, types, paths = [], [], [], [], []
    for defect_dir in sorted((root / "test").iterdir()):
        dtype = defect_dir.name
        for p in list_images(defect_dir):
            gt = None
            if dtype != "good":
                gt = root / "ground_truth" / dtype / f"{p.stem}_mask.png"
            images.append(load_image(p))
            masks.append(load_mask(gt))
            labels.append(0 if dtype == "good" else 1)
            types.append(dtype)
            paths.append(p)
    return np.stack(images), np.array(labels), np.stack(masks), types, paths

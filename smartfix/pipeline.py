"""The single entry point used by BOTH the evaluation script and the app.

    inspector = Inspector.load()
    result = inspector.inspect_part(image_bgr)
    result["verdict"]  -> "conforme" | "anomalie détectée"
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import postprocess as pp
from .config import CATEGORY, IMG_SIZE, MODELS_DIR
from .data import preprocess
from .stat_model import StatModel

OK, KO = "conforme", "anomalie détectée"


def model_path(category: str = CATEGORY) -> Path:
    return MODELS_DIR / f"{category}_stat.npz"


@dataclass
class Inspector:
    model: StatModel
    object_mask: np.ndarray | None = None
    min_area: int = 30

    @classmethod
    def load(cls, category: str = CATEGORY) -> "Inspector":
        path = model_path(category)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run: python scripts/train_stat.py")
        return cls.from_model(StatModel.load(path))

    @classmethod
    def from_model(cls, model: StatModel) -> "Inspector":
        obj = pp.object_mask_from_mean(model.mean)
        if not 0.05 < obj.mean() < 0.95:   # mask not meaningful -> whole image
            obj = None
        return cls(model=model, object_mask=obj)

    def analyse(self, img: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Preprocessed image -> (anomaly map, score, cleaned defect mask)."""
        amap = self.model.anomaly_map(img)
        score = self.model.image_score(amap)
        mask = pp.defect_mask(amap, self.model.pixel_threshold, min_area=self.min_area)
        return amap, score, mask

    def inspect_part(self, image_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        img = preprocess(image_bgr, IMG_SIZE)
        amap, score, mask = self.analyse(img)
        is_defect = score > self.model.image_threshold
        if not is_defect:
            mask = np.zeros_like(mask)   # verdict OK -> no zone drawn
        contours = pp.find_contours(mask)
        overlay = pp.draw_contours(image_bgr, contours, IMG_SIZE)
        heat = pp.heatmap_overlay(image_bgr, amap, vmax=3 * self.model.pixel_threshold)
        pct = pp.defect_percentage(mask, self.object_mask)
        return {
            "verdict": KO if is_defect else OK,
            "is_defect": bool(is_defect),
            "score": score,
            "threshold": self.model.image_threshold,
            "amap": amap,
            "mask": mask,
            "contours": contours,
            "n_zones": len(contours),
            "overlay": overlay,
            "heatmap": heat,
            "defect_pct": pct,
            "time_ms": (time.perf_counter() - t0) * 1000,
        }

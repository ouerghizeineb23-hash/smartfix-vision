"""Fit Method A on normal train images and calibrate thresholds on val images.

Run: python scripts/train_stat.py [--category bottle]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartfix.config import CATEGORY  # noqa: E402
from smartfix.data import load_train_val  # noqa: E402
from smartfix.pipeline import model_path  # noqa: E402
from smartfix.stat_model import StatModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=CATEGORY)
    ap.add_argument("--image-pct", type=float, default=99.0)
    ap.add_argument("--pixel-pct", type=float, default=99.9)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train, val = load_train_val(args.category)
    print(f"Loaded {len(train)} train / {len(val)} val normal images")
    model = StatModel.fit(train)
    model.calibrate(val, image_pct=args.image_pct, pixel_pct=args.pixel_pct)
    out = model_path(args.category)
    model.save(out)
    print(f"Image threshold = {model.image_threshold:.3f} | "
          f"pixel threshold = {model.pixel_threshold:.3f}")
    print(f"Saved {out} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()

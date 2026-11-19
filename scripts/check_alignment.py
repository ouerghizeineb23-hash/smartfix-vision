"""Q1 - Alignment check: mean image + histogram of shifts (figures for the report).

Run: python scripts/check_alignment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from smartfix.align import mean_image, measure_shifts  # noqa: E402
from smartfix.config import CATEGORY, RESULTS_DIR  # noqa: E402
from smartfix.data import load_train_val  # noqa: E402


def main():
    train, val = load_train_val()
    allimgs = np.concatenate([train, val])
    ref = mean_image(allimgs)
    shifts = measure_shifts(allimgs, ref)
    mag = np.hypot(shifts[:, 0], shifts[:, 1])
    out = RESULTS_DIR / CATEGORY
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(allimgs[0]); axes[0].set_title("Exemple d'image normale")
    axes[1].imshow(ref); axes[1].set_title(f"Image moyenne ({len(allimgs)} images)")
    axes[2].hist(mag, bins=20)
    axes[2].set(title="Décalage par rapport à la moyenne",
                xlabel="Décalage (pixels, à 256 px)", ylabel="Nombre d'images")
    for a in axes[:2]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "alignment.png", dpi=150)

    print(f"Shift: mean {mag.mean():.2f} px, median {np.median(mag):.2f} px, "
          f"max {mag.max():.2f} px")
    if np.percentile(mag, 95) <= 3:
        print("=> Images are well aligned: no registration needed.")
    else:
        print("=> Noticeable misalignment: consider registering images (align.register).")
    print(f"Figure saved to {out / 'alignment.png'}")


if __name__ == "__main__":
    main()

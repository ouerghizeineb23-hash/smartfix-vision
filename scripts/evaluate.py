"""Evaluate Method A on the test set: metrics table + figures for the report.

Run: python scripts/evaluate.py
Outputs in results/<category>/ : metrics.csv, per_image.csv, roc.png,
threshold_study.png, score_histogram.png, gallery_success.png, gallery_failures.png
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smartfix import metrics as M  # noqa: E402
from smartfix.config import CATEGORY, RESULTS_DIR  # noqa: E402
from smartfix.data import load_test  # noqa: E402
from smartfix.pipeline import Inspector  # noqa: E402


def main():
    insp = Inspector.load()
    model = insp.model
    images, labels, gt, types, paths = load_test()
    print(f"Test set: {len(images)} images ({(labels == 0).sum()} good, "
          f"{(labels == 1).sum()} defective)")

    amaps, scores, masks = [], [], []
    for img in images:
        a, s, m = insp.analyse(img)
        amaps.append(a); scores.append(s); masks.append(m)
    amaps, scores, masks = np.stack(amaps), np.array(scores), np.stack(masks)

    # Timing on the full app path (read original image -> verdict + overlay)
    times = []
    for p in paths:
        bgr = cv2.imread(str(p))
        t0 = time.perf_counter()
        insp.inspect_part(bgr)
        times.append((time.perf_counter() - t0) * 1000)

    res = M.image_metrics(labels, scores, model.image_threshold)
    res["pixel_auroc"] = M.pixel_auroc(gt, amaps)
    res["mean_iou"] = M.mean_iou(masks, gt, labels)
    res["ms_per_image"] = float(np.mean(times))
    res["image_threshold"] = model.image_threshold
    res["pixel_threshold"] = model.pixel_threshold

    out = RESULTS_DIR / CATEGORY
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).T.rename(columns={0: "Méthode A"}).to_csv(out / "metrics.csv")

    pred = (scores > model.image_threshold).astype(int)
    per = pd.DataFrame({
        "file": [f"{t}/{p.name}" for t, p in zip(types, paths)], "type": types,
        "label": labels, "score": scores, "pred": pred,
        "iou": [M.iou(m, g) if l else np.nan for m, g, l in zip(masks, gt, labels)],
    })
    per.to_csv(out / "per_image.csv", index=False)

    # Per defect type: detection rate
    by_type = per.groupby("type").apply(
        lambda d: pd.Series({"n": len(d), "correct": int((d.pred == d.label).sum()),
                             "mean_iou": d.iou.mean()}), include_groups=False)
    by_type.to_csv(out / "per_type.csv")

    M.plot_roc(labels, scores, out / "roc.png")
    M.plot_threshold_study(labels, scores, model.image_threshold, out / "threshold_study.png")
    M.plot_score_histogram(labels, scores, model.image_threshold, out / "score_histogram.png")

    vmax = 3 * model.pixel_threshold
    correct = np.where(pred == labels)[0]
    wrong = np.where(pred != labels)[0]
    # successes: best-IoU defective images + one good image
    defect_ok = [i for i in correct if labels[i] == 1]
    defect_ok = sorted(defect_ok, key=lambda i: -per.iou[i])[:4]
    good_ok = [i for i in correct if labels[i] == 0][:1]
    for name, idx in [("gallery_success", defect_ok + good_ok), ("gallery_failures", list(wrong[:6]))]:
        if len(idx):
            M.plot_gallery(images[idx], amaps[idx], masks[idx], gt[idx],
                           [f"{types[i]} | score {scores[i]:.2f}" for i in idx],
                           out / f"{name}.png", vmax)

    print("\n=== Results (Method A) ===")
    for k, v in res.items():
        print(f"{k:>16}: {v:.3f}" if isinstance(v, float) else f"{k:>16}: {v}")
    print("\nPer defect type:\n", by_type.to_string())
    print(f"\nFigures and tables saved to {out}")


if __name__ == "__main__":
    main()

"""Q4 - Evaluation metrics and figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score, roc_curve)


def image_metrics(labels, scores, threshold) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    pred = (scores > threshold).astype(int)
    return {
        "image_auroc": roc_auc_score(labels, scores),
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "false_alarms": int(((pred == 1) & (labels == 0)).sum()),
        "missed_defects": int(((pred == 0) & (labels == 1)).sum()),
    }


def pixel_auroc(gt_masks, amaps, max_pixels: int = 2_000_000, seed: int = 0) -> float:
    y = np.asarray(gt_masks).ravel().astype(np.uint8)
    s = np.asarray(amaps).ravel()
    if len(y) > max_pixels:  # subsample to keep memory low
        idx = np.random.default_rng(seed).choice(len(y), max_pixels, replace=False)
        y, s = y[idx], s[idx]
    return roc_auc_score(y, s)


def iou(pred_mask, gt_mask) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(inter / union) if union else 1.0


def mean_iou(pred_masks, gt_masks, labels) -> float:
    vals = [iou(p, g) for p, g, l in zip(pred_masks, gt_masks, labels) if l == 1]
    return float(np.mean(vals)) if vals else float("nan")


def threshold_sweep(labels, scores, n: int = 200):
    """False alarms (good parts rejected) and missed defects vs threshold."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    ts = np.linspace(scores.min(), scores.max(), n)
    fa = np.array([((scores > t) & (labels == 0)).sum() for t in ts])
    md = np.array([((scores <= t) & (labels == 1)).sum() for t in ts])
    return ts, fa, md


# ---------------- figures ----------------
def plot_roc(labels, scores, path, title="Courbe ROC (niveau image)"):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="Taux de fausses alarmes", ylabel="Taux de détection", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_threshold_study(labels, scores, chosen, path):
    ts, fa, md = threshold_sweep(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, fa, label="Fausses alarmes (pièces bonnes rejetées)")
    ax.plot(ts, md, label="Défauts manqués")
    ax.axvline(chosen, color="k", ls="--", label=f"Seuil choisi = {chosen:.2f}")
    ax.set(xlabel="Seuil du score d'anomalie", ylabel="Nombre d'images",
           title="Étude du seuil")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_score_histogram(labels, scores, chosen, path):
    labels, scores = np.asarray(labels), np.asarray(scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[labels == 0], bins=20, alpha=0.6, label="Conformes")
    ax.hist(scores[labels == 1], bins=20, alpha=0.6, label="Défectueuses")
    ax.axvline(chosen, color="k", ls="--", label="Seuil")
    ax.set(xlabel="Score d'anomalie", ylabel="Nombre d'images",
           title="Distribution des scores")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_gallery(images, amaps, pred_masks, gt_masks, titles, path, vmax):
    """One row per image: original | anomaly map | predicted mask | ground truth."""
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    cols = ["Image", "Carte d'anomalie", "Masque prédit", "Vérité terrain"]
    for i in range(n):
        panels = [images[i], amaps[i], pred_masks[i], gt_masks[i]]
        for j, p in enumerate(panels):
            ax = axes[i, j]
            if j == 1:
                ax.imshow(p, cmap="jet", vmin=0, vmax=vmax)
            elif j >= 2:
                ax.imshow(p, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(p)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j])
        axes[i, 0].set_ylabel(titles[i], fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

# SmartFix Vision: Full Project Plan

**Goal:** detect surface defects on industrial parts from a single photo.
**Input:** a photo of a part. **Output:** a verdict ("conforme" or "anomalie détectée"), the defective zones outlined in red, and the percentage of the surface that is affected.
**Team:** Ines Oueslati, Zeineb Ouerghi. **Length:** 6 weeks.

---

## 0. Key decisions to make first

| Decision | Recommendation | Why |
|---|---|---|
| MVTec AD category | **`bottle`** (backups: `transistor`, `pill`) | Method A compares each pixel to the *average normal image*, so the parts must sit in the same place in every photo. `bottle` is photographed from above, centred, with large, clear defects (broken_large, broken_small, contamination). Avoid `screw`, `metal_nut` and `hazelnut` (rotated parts) and the textures (carpet, wood, ...), where a per-pixel mean means nothing. |
| Image size | Resize to **256×256** | Fast on a laptop CPU, and it is the size most papers use, so results are comparable. |
| Colour | Process the 3 RGB channels, then take the **max z-score over channels** | Contamination is often a colour change rather than a brightness change. |
| Threshold | Choose it on a **validation set of normal images** (e.g. 20% of `train/good`) | You must not tune on the test set. That is the classic mistake that examiners look for. |
| Code layout | Python package + notebooks for exploration + Streamlit app | See section 2. |

The two people cover 4 roles (Q1–Q4), so each person owns 2 blocks. Suggested split:
- **Person 1:** Q1 (data, preprocessing, alignment) + Q2 (reference model, anomaly map, thresholds)
- **Person 2:** Q4 (contours, metrics, deep method) + Q3 (Streamlit app, report, slides)

---

## 1. Environment setup (day 1)

1. Install **Git** and **VS Code** if they are not already installed.
2. In `CV-PROJECT`:
   ```bash
   git init
   python -m venv .venv
   .venv\Scripts\activate
   pip install numpy opencv-python scikit-image scikit-learn matplotlib pandas pillow tqdm streamlit torch torchvision jupyter
   pip freeze > requirements.txt
   ```
3. Create a private GitHub repository and give both team members access.
4. Add a `.gitignore` containing `.venv/`, `data/`, `models/*.npz`, `__pycache__/` and `*.db`. The dataset must **never** be committed.

## 2. Repository structure

```
CV-PROJECT/
├── data/mvtec/bottle/          # dataset (not in git)
│   ├── train/good/
│   ├── test/{good,broken_large,broken_small,contamination}/
│   └── ground_truth/{broken_large,...}/   # *_mask.png
├── smartfix/
│   ├── data.py         # loading, resizing, train/val split          (Q1)
│   ├── align.py        # alignment check + ECC registration           (Q1)
│   ├── stat_model.py   # Method A: mean/std, z-score map              (Q2)
│   ├── postprocess.py  # threshold, morphology, contours, % defect     (Q4)
│   ├── deep_model.py   # Method B: ResNet18 patch features            (Q4)
│   ├── metrics.py      # AUROC, IoU, threshold study, timing          (Q4)
│   └── pipeline.py     # one function: image -> verdict, mask, %     (shared)
├── scripts/
│   ├── train_stat.py   # fit Method A, save models/bottle_stat.npz
│   ├── train_deep.py   # fit Method B, save models/bottle_deep.pt
│   └── evaluate.py     # all metrics + figures into results/
├── app/streamlit_app.py                                              (Q3)
├── notebooks/01_explore.ipynb
├── results/            # figures and metrics tables for the report
├── report/
├── requirements.txt
└── README.md
```

The key idea is that `smartfix/pipeline.py` exposes **a single function** used by both the evaluation script and the app (not named `inspect.py`, which would clash with Python's built-in `inspect` module):
```python
def inspect_part(image_bgr) -> dict:
    # returns {"verdict": "conforme"|"anomalie détectée", "score": float,
    #          "mask": HxW bool, "contours": [...], "defect_pct": float, "time_ms": float}
```

---

## 3. Weeks 1–2: Data, preprocessing, alignment (Q1)

**Step 3.1: Download the dataset**
- Go to the official MVTec AD page (mvtec.com → Company → Research → Datasets → MVTec AD), accept the licence (CC BY-NC-SA 4.0, non-commercial use, which is fine for a school project) and download **only `bottle.tar.xz`** (about 150 MB). You do not need the full 4.9 GB archive.
- Extract it into `data/mvtec/bottle/`.

**Step 3.2: Explore (notebook `01_explore.ipynb`)**
- Count the images: `train/good` has 209, `test` has 83 (20 good and 63 defective).
- Display about 10 normal images and 3 of each defect type, each with its mask.
- Write down the image size (900×900) and the average defect size in pixels (from the masks). This tells you what minimum contour area is reasonable later on.

**Step 3.3: Loading and preprocessing (`data.py`)**
- `load_image(path, size=256)` → RGB, float32 in [0,1], resized.
- Optional light Gaussian blur (σ≈1) to reduce sensor noise.
- `split_train_val(train_good, val_ratio=0.2, seed=42)` → about 167 images to fit the model and 42 normal images to set the threshold.
- `load_test()` → images, labels (0 = good, 1 = defect), masks (all zeros for good images).

**Step 3.4: Check alignment (`align.py`)**
- Compute the mean image of `train/good` and look at it. If it is sharp, the parts are aligned. If it is blurry, they are not.
- Quantify it: for each image, measure the translation relative to the mean image with `cv2.phaseCorrelate`. Plot a histogram of the shifts.
- If the shifts are more than about 2–3 px, add registration: `cv2.findTransformECC` (Euclidean motion) towards the mean image, then `cv2.warpAffine`. Apply the same registration at test time.
- **Deliverable:** a figure "mean image before/after alignment" for the report.

✅ **End of week 2:** the data loads in one line, the split is fixed, and alignment is verified and documented.

---

## 4. Weeks 3–4: Method A, the statistical reference (Q2 + Q4)

**Step 4.1: Fit (`stat_model.py`)**
```python
X = stack(train_images)            # N × H × W × 3
mean = X.mean(axis=0)
std  = X.std(axis=0) + eps          # eps ≈ 0.01, avoids dividing by ~0
np.savez("models/bottle_stat.npz", mean=mean, std=std)
```

**Step 4.2: Anomaly map**
```python
z = np.abs(img - mean) / std        # H × W × 3
amap = z.max(axis=2)                # max over channels
amap = cv2.GaussianBlur(amap, (0,0), sigmaX=4)   # smooths isolated noise
```
- **Image score** = `amap.max()` (or the mean of the top 1% of pixels, which is more robust).
- Optional object mask: threshold the mean image to keep only the bottle and ignore the background.

**Step 4.3: Threshold (Q2)**
- Compute the image scores on the **42 validation** normal images.
- Threshold = 99th percentile (or max) of those scores. This targets about 1% false alarms.
- Pixel threshold: same idea on the validation pixels.
- Save both thresholds alongside the model.

**Step 4.4: Post-processing (`postprocess.py`, Q4)**
1. `mask = amap > pixel_threshold`
2. Morphology: `cv2.morphologyEx(OPEN)` to remove specks, then `CLOSE` to fill holes (5×5 elliptical kernel).
3. Remove components smaller than `min_area` (found in step 3.2).
4. `cv2.findContours` → draw them in **red** on the original image (scale the coordinates back to the original size).
5. `defect_pct = mask.sum() / object_area * 100`. State in the report whether this is a % of the part or of the image.

**Step 4.5: Evaluation (`metrics.py`, Q4)**
| Metric | How |
|---|---|
| Image AUROC | `roc_auc_score(labels, image_scores)` |
| Verdict accuracy / precision / recall / F1 | at the chosen threshold |
| Pixel AUROC | `roc_auc_score(all_masks.ravel(), all_amaps.ravel())` |
| IoU | `|pred ∩ gt| / |pred ∪ gt|` on the defective images |
| Threshold study | sweep the threshold, plot **false alarms vs missed defects**, mark the chosen point |
| Time per image | `time.perf_counter()` around `inspect_part()`, average over the test set |

Also produce a **gallery of successes and failures** (one row per image: original | map | predicted mask | ground truth). Examiners value an honest error analysis.

✅ **End of week 4:** `python scripts/evaluate.py --method stat` produces a metrics table and figures in `results/`.

> Expected level: Method A typically reaches an image AUROC of about 0.85–0.95 on `bottle`. Don't worry if it is lower than deep learning. That gap is exactly what Method B is there to measure.

---

## 5. Week 5: Streamlit application (Q3)

**`app/streamlit_app.py`**: two pages (sidebar).

**"Inspection" page**
1. `st.file_uploader` (png/jpg) + a "use a test example" button (very useful for the demo).
2. Call `inspect_part()`.
3. Show a large verdict: green "✅ Conforme" / red "❌ Anomalie détectée".
4. Two columns: the image with red contours | the anomaly map (heatmap).
5. Metrics: % of surface affected, anomaly score, processing time.
6. A method selector (A statistical / B deep) if B is ready.

**"Historique & statistiques" page**
- Each inspection is saved to SQLite (`history.db`): date, file name, verdict, score, %, method.
- Table with `st.dataframe`, CSV export.
- Simple statistics: number of inspections, compliance rate, histogram of the %, verdicts per day.

**Tips**
- Load the model with `@st.cache_resource`, so it is loaded once rather than on every click.
- Run it: `streamlit run app/streamlit_app.py`

✅ **End of week 5:** a complete demo works on a laptop without internet.

---

## 6. Method B (optional, in parallel during weeks 4–5, Q4)

A simplified PaDiM/PatchCore approach, to **measure the gap** with Method A:
1. Pre-trained `torchvision.models.resnet18(weights="DEFAULT")`, in eval mode, without gradients.
2. Extract `layer2` and `layer3` (forward hooks), resize `layer3` to the size of `layer2`, concatenate → one vector per patch (e.g. 32×32 patches × 384 dimensions).
3. **Memory bank:** all patches from the normal training images (randomly subsample to about 10% if RAM is tight).
4. Test: for each patch, take the distance to its **nearest normal neighbour** (`torch.cdist`, min) → 32×32 map → upsample to 256 + Gaussian blur.
5. Image score = max of the map. Same post-processing and same metrics as Method A.

Expected result: image AUROC of about 0.98–1.0 on `bottle`. Final comparison table for the report:

| Method | Image AUROC | Pixel AUROC | IoU | F1 | ms/image | Training |
|---|---|---|---|---|---|---|
| A: statistical | | | | | | seconds, no GPU |
| B: ResNet18 kNN | | | | | | minutes, CPU OK |

---

## 7. Week 6: Report, slides, README, rehearsal (Q3, everyone)

**Report (suggested outline)**
1. Introduction: the problem (slow, subjective, costly manual inspection)
2. Data: MVTec AD, why `bottle`, statistics
3. Method A: alignment, statistical model, post-processing (with equations)
4. Method B: features, kNN
5. Protocol: train/val/test split, threshold choice
6. Results: metrics table, ROC curves, threshold study, visual gallery, timing
7. Discussion: limitations (alignment, lighting, rotated parts), industrial outlook
8. Conclusion

**README:** description, installation, dataset download, commands (`train`, `evaluate`, `streamlit run`), app screenshots, results table.

**Rehearsal:** a 10-minute run-through + a live demo + a **backup video** of the demo, in case the laptop fails on the day.

---

## 8. Week-by-week summary

| Week | Person 1 (Q1+Q2) | Person 2 (Q4+Q3) | Checkpoint |
|---|---|---|---|
| 1 | Setup, download, exploration | Setup, repo, `metrics.py` skeleton | Data visible in the notebook |
| 2 | Preprocessing, split, alignment | Contours + % on ground-truth masks (test the drawing code) | Alignment verified |
| 3 | Fit Method A, anomaly map | Evaluation script, ROC curves | First AUROC |
| 4 | Threshold tuning on val, morphology | Threshold study, gallery; start Method B | Complete Method A results |
| 5 | Help with the app, `inspect_part()` | Streamlit app + history; finish B | Demo working |
| 6 | Report sections 2–4 | Report 5–7, slides, README | Rehearsal + video |

## 9. Common pitfalls

- ❌ Choosing the threshold on the test set → use the validation split.
- ❌ Forgetting the `+ eps` on the std → division by ~0 in the uniform background, which floods the map with false alarms.
- ❌ Preprocessing the training and test images differently (size, blur, alignment) → a single `preprocess()` function used everywhere.
- ❌ Committing the dataset or the `.venv` to git.
- ❌ Contour coordinates computed at 256 px and drawn at 900 px without rescaling.
- ❌ Testing the app only on test images: also try a phone photo and explain in the report why it fails (no alignment, different lighting). That is a good discussion point.

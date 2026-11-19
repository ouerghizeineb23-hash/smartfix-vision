# SmartFix Vision

Surface defect detection on industrial parts using computer vision (MVTec AD dataset).
Input: a photo of a part. Output: a verdict (conforme / anomalie détectée), defective zones outlined in red, and the % of the surface affected.

Team: Ines Oueslati, Zeineb Ouerghi. Full plan: [PLAN.md](PLAN.md).

## Installation (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

## Dataset

1. Download `bottle.tar.xz` from the official MVTec AD page (accept the CC BY-NC-SA 4.0 licence).
2. Extract it so that you get `data/mvtec/bottle/train`, `test` and `ground_truth`.
3. Check: `python scripts/check_setup.py`

## Usage

| Command | Purpose |
|---|---|
| `python scripts/check_setup.py` | Check the libraries and the dataset |
| `streamlit run app/streamlit_app.py` | Launch the inspection app |

## Structure

- `smartfix/`: the library (data, alignment, models, post-processing, metrics, pipeline)
- `scripts/`: training and evaluation scripts
- `app/`: Streamlit application
- `notebooks/`: exploration
- `results/`: figures and metric tables
- `report/`: report and slides

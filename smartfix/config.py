"""Shared settings: change CATEGORY here to switch MVTec AD category."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mvtec"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CATEGORY = "bottle"
IMG_SIZE = 256
VAL_RATIO = 0.2
SEED = 42

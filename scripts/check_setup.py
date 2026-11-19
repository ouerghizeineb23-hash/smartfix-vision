"""Run: python scripts/check_setup.py  - verifies the libraries and the dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy  # noqa: E402
import sklearn  # noqa: E402,F401
import streamlit  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402,F401

from smartfix.config import CATEGORY, DATA_DIR  # noqa: E402

print(f"Python {sys.version.split()[0]} | numpy {numpy.__version__} | "
      f"opencv {cv2.__version__} | torch {torch.__version__} | "
      f"streamlit {streamlit.__version__}")

if not (DATA_DIR / CATEGORY).exists():
    print(f"[ ] Dataset missing: extract {CATEGORY}.tar.xz into {DATA_DIR}")
    sys.exit(0)

from smartfix.data import load_test, load_train_val  # noqa: E402

train, val = load_train_val()
x, y, m, types, _ = load_test()
print(f"[x] {CATEGORY}: train {len(train)}, val {len(val)}, "
      f"test {len(x)} ({(y == 0).sum()} good / {(y == 1).sum()} defect), "
      f"types {sorted(set(types))}")

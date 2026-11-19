"""Q3 - SmartFix Vision inspection app.

Run: streamlit run app/streamlit_app.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from smartfix import history  # noqa: E402
from smartfix.config import CATEGORY, DATA_DIR  # noqa: E402
from smartfix.pipeline import Inspector  # noqa: E402

st.set_page_config(page_title="SmartFix Vision", page_icon="🔍", layout="wide")


@st.cache_resource
def get_inspector():
    return Inspector.load()


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def test_examples():
    root = DATA_DIR / CATEGORY / "test"
    return sorted(root.glob("*/*.png")) if root.exists() else []


# ---------------- sidebar ----------------
st.sidebar.title("🔍 SmartFix Vision")
page = st.sidebar.radio("Navigation", ["Inspection", "Historique & statistiques"])
st.sidebar.caption(f"Catégorie : **{CATEGORY}** · Méthode A (référence statistique)")

try:
    inspector = get_inspector()
except FileNotFoundError as e:
    st.error(f"Modèle introuvable. Lancez d'abord l'entraînement.\n\n{e}")
    st.stop()

# ---------------- page: inspection ----------------
if page == "Inspection":
    st.title("Inspection d'une pièce")
    st.caption("Importer une photo → verdict, zones défectueuses en rouge, "
               "pourcentage de surface touchée.")

    col_up, col_ex = st.columns([3, 1])
    uploaded = col_up.file_uploader("Importer une photo de la pièce",
                                    type=["png", "jpg", "jpeg", "bmp"])
    examples = test_examples()
    if examples and col_ex.button("🎲 Exemple de test aléatoire", use_container_width=True):
        st.session_state["example"] = str(random.choice(examples))

    image_bgr, name = None, None
    if uploaded is not None:
        data = np.frombuffer(uploaded.getvalue(), np.uint8)
        image_bgr, name = cv2.imdecode(data, cv2.IMREAD_COLOR), uploaded.name
        st.session_state.pop("example", None)
    elif "example" in st.session_state:
        p = Path(st.session_state["example"])
        image_bgr, name = cv2.imread(str(p)), f"{p.parent.name}/{p.name}"

    if image_bgr is None:
        st.info("Importez une image ou cliquez sur « Exemple de test aléatoire ».")
        st.stop()

    r = inspector.inspect_part(image_bgr)
    key = (name, round(r["score"], 6))
    if st.session_state.get("last_saved") != key:   # avoid duplicates on rerun
        history.add(name, r)
        st.session_state["last_saved"] = key

    if r["is_defect"]:
        st.error(f"## ❌ Anomalie détectée\n{name}")
    else:
        st.success(f"## ✅ Conforme\n{name}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Surface touchée", f"{r['defect_pct']:.2f} %")
    m2.metric("Zones défectueuses", r["n_zones"])
    m3.metric("Score d'anomalie", f"{r['score']:.2f}", help=f"Seuil = {r['threshold']:.2f}")
    m4.metric("Temps de traitement", f"{r['time_ms']:.0f} ms")

    c1, c2, c3 = st.columns(3)
    c1.image(bgr_to_rgb(image_bgr), caption="Image originale", use_container_width=True)
    c2.image(bgr_to_rgb(r["overlay"]), caption="Zones défectueuses (rouge)",
             use_container_width=True)
    c3.image(bgr_to_rgb(r["heatmap"]), caption="Carte d'anomalie",
             use_container_width=True)

# ---------------- page: history ----------------
else:
    st.title("Historique & statistiques")
    df = history.load()
    if df.empty:
        st.info("Aucune inspection pour l'instant.")
        st.stop()

    n = len(df)
    n_ok = int((df.verdict == "conforme").sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Inspections", n)
    k2.metric("Taux de conformité", f"{100 * n_ok / n:.1f} %")
    k3.metric("Anomalies", n - n_ok)
    k4.metric("Temps moyen", f"{df.time_ms.mean():.0f} ms")

    s1, s2 = st.columns(2)
    s1.subheader("Verdicts")
    s1.bar_chart(df.verdict.value_counts())
    s2.subheader("Inspections par jour")
    per_day = (df.assign(jour=pd.to_datetime(df.timestamp).dt.date)
                 .groupby(["jour", "verdict"]).size().unstack(fill_value=0))
    s2.bar_chart(per_day)

    st.subheader("Toutes les inspections")
    st.dataframe(df, use_container_width=True, hide_index=True)
    b1, b2 = st.columns([1, 5])
    b1.download_button("⬇️ Export CSV", df.to_csv(index=False).encode("utf-8"),
                       "historique.csv", "text/csv")
    if b2.button("🗑️ Vider l'historique"):
        history.clear()
        st.rerun()

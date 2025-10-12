import os, sys
import streamlit as st

# ---------- Resolve project paths ----------
FILE_DIR  = os.path.dirname(os.path.abspath(__file__))          # .../src/app
PROJ_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", "..")) # repo root
ART_DIR   = os.path.join(PROJ_ROOT, "artifacts")

# Make repo importable (safe to add even if already present)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# ---------- Load model (cached) ----------
import joblib

@st.cache_resource
def load_pipe():
    path = os.path.join(ART_DIR, "final_sentiment_pipe_v2.pkl")
    return joblib.load(path)

pipe = load_pipe()

# ---------- UI ----------
st.title("Women Clothing - Sentiment Classifier")
st.write("Type a customer review and see the sentiment prediction")

txt = st.text_area("Review text:")

col1, col2 = st.columns(2)
with col1:
    if st.button("Predict"):
        if not txt.strip():
            st.warning("Please enter some text first.")
        else:
            try:
                pred = pipe.predict([txt])[0]   # expects -1/0/1 based on your training
                label_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
                label = label_map.get(int(pred), str(pred))
                st.success(f"Prediction: **{label}**")
            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)
with col2:
    if st.button("Show probabilities"):
        try:
            proba = getattr(pipe, "predict_proba", None)
            if proba is None:
                st.info("This model doesn't expose predict_proba.")
            else:
                p = proba([txt])[0]
                st.write({"neg(-1)": float(p[0]), "neu(0)": float(p[1]), "pos(1)": float(p[2])})
        except Exception as e:
            st.error("Could not compute probabilities.")
            st.exception(e)

# ---------- Diagnostics (keep for sanity while deploying) ----------
import sklearn, numpy as np, scipy, pathlib, hashlib
st.write({
    "sklearn_version": sklearn.__version__,
    "numpy_version":   np.__version__,
    "scipy_version":   scipy.__version__,
})

tfidf = pipe.named_steps.get("tfidf")
st.write("Has tfidf step:", tfidf is not None)
st.write("use_idf:", getattr(tfidf, "use_idf", None))
st.write("Has vocabulary_:", hasattr(tfidf, "vocabulary_"))
st.write("Has idf_:", hasattr(tfidf, "idf_"))

artifact_path = os.path.join(ART_DIR, "final_sentiment_pipe_v2.pkl")
st.write("Artifact Path", artifact_path)
p = pathlib.Path(artifact_path)
if p.exists():
    st.write("Artifact size (bytes):", p.stat().st_size)
    st.write("Artifact sha256 (first 16):", hashlib.sha256(p.read_bytes()).hexdigest()[:16])
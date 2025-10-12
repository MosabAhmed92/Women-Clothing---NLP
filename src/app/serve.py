import sys
import os
import streamlit as st  # <-- streamlit import is fine anywhere

# ---------- add repo root and src/ to sys.path BEFORE importing from src.* ----------
FILE_DIR  = os.path.dirname(os.path.abspath(__file__))          # .../src/app
PROJ_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", "..")) # repo root

if FILE_DIR in sys.path:
    sys.path.remove(FILE_DIR)

if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

ART_DIR = os.path.join(PROJ_ROOT, "artifacts")
# ------------------------------------------------------------------------------------

from src.models.persist import load_model

@st.cache_resource
def get_model():
    return load_model(ART_DIR, "final_sentiment_pipe_v2")

st.title("Women Clothing - Sentiment Classifier")
st.write("Type a customer review and see the sentiment prediction")

user_input = st.text_area("Review text:")

pipe = get_model()
pipe = load_model(ART_DIR, "final_sentiment_pipe")  

import sklearn, numpy as np, scipy
st.write({
    "sklearn_version": sklearn.__version__,
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
})

tfidf = pipe.named_steps.get("tfidf")
st.write("Has tfidf step:", tfidf is not None)
st.write("use_idf:", getattr(tfidf, "use_idf", None))
st.write("Has vocabulary_:", hasattr(tfidf, "vocabulary_"))
st.write("Has idf_:", hasattr(tfidf, "idf_"))
st.write("Artifact Path", os.path.join(ART_DIR, "final_sentiment_pipe.pkl"))
import hashlib, pathlib
p = pathlib.Path(ART_DIR) / "final_sentiment_pipe.pkl"
if p.exists():
    st.write("Artifact size (bytes):", p.stat().st_size)
    st.write("Artifact sha256 (first 16):",
             hashlib.sha256(p.read_bytes()).hexdigest()[:16])
    





if st.button("Predict", type="primary"):
    txt = (user_input or "").strip()
    if not txt:
        st.warning("Please type a review")
    else:
        try:
            y = pipe.predict([txt])[0]              # <-- list[str]
            st.write("raw prediction:", y)          # always show something

            label_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
            label = label_map.get(int(y), str(y))

            # show a friendly message
            if label == "Positive":
                st.success("Positive Review")
            elif label == "Neutral":
                st.info("Neutral Review")
            else:
                st.error("Negative Review")
        except Exception as e:
            st.error("Prediction failed:")
            st.exception(e)  # <-- surfaces any hidden errors
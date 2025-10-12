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
    return load_model(ART_DIR, "final_sentiment_pipe")

st.title("Women Clothing - Sentiment Classifier")
st.write("Type a customer review and see the sentiment prediction")

user_input = st.text_area("Review text:")

model = get_model()
tfidf = model.named_steps.get("tfidf", None)  # preferred

has_vocab = bool(tfidf is not None and hasattr(tfidf, "vocabulary_") and tfidf.vocabulary_)
has_idf   = bool(tfidf is not None and hasattr(tfidf, "idf_")        and tfidf.idf_ is not None)

st.write("Has tfidf step:", tfidf is not None)
st.write("Has vocabulary_:", has_vocab)
st.write("Has idf_:", has_idf)
    

if st.button("Predict"):
    if user_input.strip():
        pred = model.predict([user_input])[0]
        if pred == 1:
            st.success("Positive Review")
        elif pred == 0:
            st.info("Neutral Review")
        else:
            st.error("Negative Review")
    else:
        st.warning("Please enter some text first")
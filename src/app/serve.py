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
    

pipe = get_model()  # your cached pipeline

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
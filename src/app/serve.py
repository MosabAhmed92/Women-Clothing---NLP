import streamlit as st
import joblib
import sklearn
import numpy
from pathlib import Path

# -----------------------------
# Load fitted pipeline once
# -----------------------------

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "sentiment_pipe.joblib"

@st.cache_resource
def load_pipeline():
        pipe = joblib.load(MODEL_PATH)
        vectorizer = pipe.named_steps['tfidf']
        return pipe

pipe = load_pipeline()


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🧵 Sentiment Prediction App")
st.write("Enter any product review or sentence to analyze its sentiment.")

user_text = st.text_area("Input text:", height=150)

if st.button("Predict"):
    if user_text.strip():
        pred = pipe.predict([user_text])[0]

        st.subheader("Result:")
        if pred == 1:
            st.success("✅ Positive Sentiment")
        elif pred == 0:
            st.info("😐 Neutral Sentiment")
        else:
            st.error("❌ Negative Sentiment")

    else:
        st.warning("Please enter some text first.")
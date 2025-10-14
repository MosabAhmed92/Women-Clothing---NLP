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

        if hasattr(vectorizer, 'idf_'):
            st.success(f"✅ TfidfVectorizer loaded successfully. idf_ shape: {vectorizer.idf_.shape}")
        else:
            st.error("❌ TfidfVectorizer found, but 'idf_' attribute is missing after loading.")

            try:
                vectorizer_path = Path(__file__).resolve().parents[2] / "artifacts" / "tfidf_vectorizer.joblib"
                backup_vectorizer = joblib.load(vectorizer_path)
                pipe.named_steps['tfidf'] = backup_vectorizer
                st.success("✅ Backup vectorizer loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load backup vectorizer: {e}")

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
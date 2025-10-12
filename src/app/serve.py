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
    try:
        loaded_pipe = joblib.load(MODEL_PATH)
        # --- New Diagnostic Code ---
        tfidf_vectorizer = loaded_pipe.named_steps.get('tfidf')
        if tfidf_vectorizer and hasattr(tfidf_vectorizer, 'idf_'):
            st.success(f"TfidfVectorizer loaded successfully. idf_ shape: {tfidf_vectorizer.idf_.shape}")
        elif tfidf_vectorizer:
            st.error("TfidfVectorizer found, but 'idf_' attribute is missing after loading.")
        else:
            st.error("TfidfVectorizer step not found in the pipeline.")
        # --- End New Diagnostic Code ---
        return loaded_pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- New Diagnostic Code for Library Versions ---
st.write(f"Joblib version: {joblib.__version__}")
st.write(f"Scikit-learn version: {sklearn.__version__}")
st.write(f"Numpy version: {numpy.__version__}")
# --- End New Diagnostic Code ---

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
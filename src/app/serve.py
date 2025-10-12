import streamlit as st
import joblib
import os # Import the os module
from pathlib import Path

# -----------------------------
# Load fitted pipeline once
# -----------------------------
MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "sentiment_pipe.joblib"

# Diagnostic checks for MODEL_PATH
if not MODEL_PATH.exists():
    st.error(f"Error: Model file not found at {MODEL_PATH}")
    st.stop() # Stop the app if the model is not found
elif not MODEL_PATH.is_file():
    st.error(f"Error: MODEL_PATH is not a file: {MODEL_PATH}")
    st.stop()
else:
    st.success(f"Model file found at: {MODEL_PATH}") # Confirm path is correct and file exists


@st.cache_resource
def load_pipeline():
    try:
        return joblib.load(MODEL_PATH)   # adjust path if needed
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

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
# src/app/serve.py
import streamlit as st
import joblib
from pathlib import Path
import os

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "sentiment_pipe.joblib"

@st.cache_resource
def load_pipeline_checked():
    st.write("Model path:", str(MODEL_PATH))
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file at {MODEL_PATH}")

    # Show file size to ensure we didn't load a tiny/unfitted artifact
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    st.write(f"Model file size: {size_mb:.2f} MB")

    pipe = joblib.load(MODEL_PATH)

    # Prove the vectorizer is fitted
    try:
        tfidf = pipe.named_steps["tfidf"]
    except Exception as e:
        raise RuntimeError("Loaded object is not a Pipeline with a 'tfidf' step.") from e

    if not hasattr(tfidf, "idf_"):
        raise RuntimeError("Loaded pipeline's TF-IDF is NOT fitted (missing idf_). "
                           "Ensure you saved the *fitted* pipeline after .fit().")

    # quick smoke transform
    _ = pipe.predict(["smoke test OK"])
    return pipe

pipe = load_pipeline_checked()


# import streamlit as st
# import joblib
# from pathlib import Path

# # -----------------------------
# # Load fitted pipeline once
# # -----------------------------
# MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "sentiment_pipe.joblib"


# @st.cache_resource
# def load_pipeline():
#     return joblib.load(MODEL_PATH)   # adjust path if needed

# pipe = load_pipeline()

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.title("🧵 Sentiment Prediction App")
# st.write("Enter any product review or sentence to analyze its sentiment.")

# user_text = st.text_area("Input text:", height=150)

# if st.button("Predict"):
#     if user_text.strip():
#         pred = pipe.predict([user_text])[0]

#         st.subheader("Result:")
#         if pred == 1:
#             st.success("✅ Positive Sentiment")
#         elif pred == 0:
#             st.info("😐 Neutral Sentiment")
#         else:
#             st.error("❌ Negative Sentiment")

#     else:
#         st.warning("Please enter some text first.")
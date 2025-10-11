import sys
import os
import site
import streamlit as st  # <-- streamlit import is fine anywhere

# ---------- add repo root and src/ to sys.path BEFORE importing from src.* ----------
FILE_DIR  = os.path.dirname(os.path.abspath(__file__))          # .../src/app
PROJ_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", "..")) # repo root
SRC_DIR   = os.path.join(PROJ_ROOT, "src")

site.addsitedir(PROJ_ROOT)

print(f"DEBUG: Final sys.path: {sys.path}")
try:
    print(f"DEBUG: Contents of PROJ_ROOT ({PROJ_ROOT}): {os.listdir(PROJ_ROOT)}")
except Exception as e:
    print(f"DEBUG: Error listing PROJ_ROOT: {e}")
try:
    print(f"DEBUG: Contents of SRC_DIR ({SRC_DIR}): {os.listdir(SRC_DIR)}")
except Exception as e:
    print(f"DEBUG: Error listing SRC_DIR: {e}")

ART_DIR = os.path.join(PROJ_ROOT, "artifacts")
# ------------------------------------------------------------------------------------

try:
    from src.models.persist import load_model
except ModuleNotFoundError as e:
    print(f"DEBUG: ModuleNotFoundError caught: {e}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG: sys.path at error: {sys.path}")
    raise e # Re-raise the exception after printing debug info

@st.cache_resource
def get_model():
    return load_model(ART_DIR, "final_sentiment_pipe")

st.title("Women Clothing - Sentiment Classifier")
st.write("Type a customer review and see the sentiment prediction")

user_input = st.text_area("Review text:")

model = get_model()

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
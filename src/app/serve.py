import sys
import os

FILE_DIR  = os.path.dirname(os.path.abspath(__file__))       # .../src/app
PROJ_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))  # repo root
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

SRC_DIR = os.path.join(PROJ_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ART_DIR = os.path.join(PROJ_ROOT, "artifacts")

from src.models.persist import load_model
import streamlit as st
# decorate the function so that it runs once per process 

@st.cache_resource
def get_model():
    model = load_model(ART_DIR, 'final_sentiment_pipe')
    return model 



st.title("Women Colthing - Sentiment Classifier")

st.write('Type a customer Review and see the sentiment prediction')

user_input = st.text_area('Review text:')

model = get_model()

if st.button('Predict'):
    if user_input.strip():
        pred = model.predict([user_input])[0]
        if pred == 1:
            st.success('Positive Review')

        elif pred == 0:
            st.info('Neutral Review')

        else:
            st.error('Negative Review')
    else:
        st.warning('Please Enter Some text First')


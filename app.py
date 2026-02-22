import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Text Emotion Detection",
    layout="wide",
    initial_sidebar_state="auto"
)

# Background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFDAB9;  /* Peach background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model_path = "model/text_emotion.pkl"
if os.path.exists(model_path):
    pipe_lr = joblib.load(open(model_path, "rb"))
else:
    pipe_lr = None
    st.warning("Model not found! Please check model path.")

# Emotion to emoji mapping
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# Prediction functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# App header
st.markdown("<h2 style='font-size:24px'>😊 Text Emotion Detection</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='font-size:16px'>Detect Emotions in Text</h4>", unsafe_allow_html=True)

# Layout: two columns
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("<h4 style='font-size:14px'>Enter Text</h4>", unsafe_allow_html=True)
    raw_text = st.text_area("", height=80, max_chars=300)
    submit_text = st.button("Submit")

with col2:
    if submit_text:
        if not raw_text.strip():
            st.error("Please enter some text!")
        elif pipe_lr:
            # Prediction
            prediction = predict_emotions(raw_text)
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.markdown(f"<h4 style='font-size:14px'>Prediction: {prediction} {emoji_icon}</h4>", unsafe_allow_html=True)

            # Confidence
            probability = get_prediction_proba(raw_text)
            st.markdown(f"<p style='font-size:12px'>Confidence: {np.max(probability)*100:.2f}%</p>", unsafe_allow_html=True)

            # Prediction probability chart
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions',
                y='probability',
                color='emotions'
            ).properties(height=150)  # smaller height for one-page fit
            st.altair_chart(fig, use_container_width=True)
        else:
            st.error("Model not loaded!")
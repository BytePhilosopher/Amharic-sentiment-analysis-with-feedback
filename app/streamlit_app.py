# streamlit_app.py

import streamlit as st
import pickle
import numpy as np

# --------------------------
# Load Model and Vectorizer
# --------------------------
@st.cache_resource
def load_model_and_vectorizer():
    with open("../models/logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("../models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --------------------------
# App Title and Intro
# --------------------------
st.set_page_config(page_title="QalAnalyzer | ቃል Analyzer", layout="centered")
st.title("🌍 QalAnalyzer | ቃል Analyzer")
st.subheader("🔍 Amharic Sentiment Classification")
st.write("Paste or type any Amharic sentence and get its **sentiment prediction**!")

# --------------------------
# Input Box
# --------------------------
amharic_input = st.text_area("✍️ Enter Amharic text here", height=150)

# --------------------------
# Predict Button
# --------------------------
if st.button("🔎 Analyze Sentiment"):
    if amharic_input.strip() == "":
        st.warning("Please enter some Amharic text to analyze.")
    else:
        # Preprocess input
        transformed_text = vectorizer.transform([amharic_input])
        prediction = model.predict(transformed_text)[0]

        # Display result
        st.success(f"🧠 **Predicted Sentiment:** `{prediction}`")

        # Optional emoji or comment
        if prediction == "positive":
            st.markdown("🎉 That sounds uplifting!")
        elif prediction == "negative":
            st.markdown("😟 That doesn't sound great.")
        elif prediction == "neutral":
            st.markdown("😐 A neutral tone detected.")
        else:
            st.markdown("🤔 Mixed or unclear sentiment.")

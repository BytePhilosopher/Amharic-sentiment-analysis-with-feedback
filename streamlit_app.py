# # streamlit_app.py

# import streamlit as st
# import pickle
# import numpy as np

# # --------------------------
# # Load Model and Vectorizer
# # --------------------------
# @st.cache_resource
# def load_model_and_vectorizer():
#     with open("models/logistic_model.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("models/tfidf_vectorizer.pkl", "rb") as f:
#         vectorizer = pickle.load(f)
#     return model, vectorizer

# model, vectorizer = load_model_and_vectorizer()

# # --------------------------
# # Page Config and Styling
# # --------------------------
# st.set_page_config(page_title="QalAnalyzer | ቃል Analyzer", layout="centered")

# st.markdown("""
#     <style>
#         .main {
#             background-color: #f5f7fa;
#         }
#         .stTextArea textarea {
#             font-size: 18px;
#             font-family: "Noto Sans Ethiopic", sans-serif;
#         }
#         .stButton button {
#             background-color: #1a73e0;
#             color: white;
#             font-size: 18px;
#             padding: 10px 20px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # --------------------------
# # App Title and Intro
# # --------------------------
# st.markdown("<h1 style='text-align: center; color: #1a73e0;'>🌍 QalAnalyzer | ቃል Analyzer</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color: #fff;'>🔍 Amharic Sentiment Classification using Machine Learning</h3>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>Paste or type an Amharic sentence below to detect its sentiment!</p>", unsafe_allow_html=True)

# # --------------------------
# # Input Box
# # --------------------------
# amharic_input = st.text_area("✍️ **Enter Amharic text here:**", height=150)

# # --------------------------
# # Predict Button and Result
# # --------------------------
# if st.button("🔎 Analyze Sentiment"):
#     if amharic_input.strip() == "":
#         st.warning("⚠️ Please enter some Amharic text to analyze.")
#     else:
#         with st.spinner("Analyzing sentiment..."):
#             transformed_text = vectorizer.transform([amharic_input])
#             prediction = model.predict(transformed_text)[0]

#         st.markdown("---")
#         # Map numerical prediction to sentiment label
#         label_map = {0: "negative", 1: "positive", 2: "neutral"}
#         sentiment_text = label_map.get(prediction, "unknown")

#         # Display prediction
#         st.markdown(f"<h2 style='text-align: center;'>🧠 Predicted Sentiment: <span style='color: #0b8043;'>`{sentiment_text.upper()}`</span></h2>", unsafe_allow_html=True)

#         # Show contextual message
#         if sentiment_text == "positive":
#             st.success("🎉 That sounds uplifting!")
#         elif sentiment_text == "negative":
#             st.error("😟 That doesn't sound great.")
#         elif sentiment_text == "neutral":
#             st.info("😐 A neutral tone detected.")
#         else:
#             st.warning("🤔 Mixed or unclear sentiment.")

# streamlit_app.py

# import streamlit as st
# import fasttext
# import numpy as np

# # --------------------------
# # Load FastText Model
# # --------------------------
# @st.cache_resource
# def load_fasttext_model():
#     return fasttext.load_model("models/amharic_fasttext_model.ftz")

# model = load_fasttext_model()

# # --------------------------
# # Page Config and Styling
# # --------------------------
# st.set_page_config(page_title="QalAnalyzer | ቃል Analyzer", layout="centered")

# st.markdown("""
#     <style>
#         .main {
#             background-color: #f5f7fa;
#         }
#         .stTextArea textarea {
#             font-size: 18px;
#             font-family: "Noto Sans Ethiopic", sans-serif;
#         }
#         .stButton button {
#             background-color: #1a73e0;
#             color: white;
#             font-size: 18px;
#             padding: 10px 20px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # --------------------------
# # App Title and Intro
# # --------------------------
# st.markdown("<h1 style='text-align: center; color: #1a73e0;'>🌍 QalAnalyzer | ቃል Analyzer</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color: #666;'>🔍 Amharic Sentiment Classification using FastText</h3>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>Paste or type an Amharic sentence below to detect its sentiment!</p>", unsafe_allow_html=True)

# # --------------------------
# # Input Box
# # --------------------------
# amharic_input = st.text_area("✍️ **Enter Amharic text here:**", height=150)

# # --------------------------
# # Predict Button and Result
# # --------------------------
# if st.button("🔎 Analyze Sentiment"):
#     if amharic_input.strip() == "":
#         st.warning("⚠️ Please enter some Amharic text to analyze.")
#     else:
#         with st.spinner("Analyzing sentiment..."):
#             label, prob = model.predict(amharic_input, k=1)
#             sentiment_text = label[0].replace("__label__", "").lower()
#             confidence = round(prob[0] * 100, 2)

#         st.markdown("---")
#         st.markdown(f"<h2 style='text-align: center;'>🧠 Predicted Sentiment: <span style='color: #0b8043;'>`{sentiment_text.upper()}`</span></h2>", unsafe_allow_html=True)
#         st.markdown(f"<p style='text-align: center;'>Confidence: {confidence}%</p>", unsafe_allow_html=True)

#         if sentiment_text == "positive":
#             st.success("🎉 That sounds uplifting!")
#         elif sentiment_text == "negative":
#             st.error("😟 That doesn't sound great.")
#         elif sentiment_text == "neutral":
#             st.info("😐 A neutral tone detected.")
#         else:
#             st.warning("🤔 Mixed or unclear sentiment.")

import streamlit as st
import pandas as pd
import fasttext
import os
import re

st.set_page_config(page_title="QalAnalyzer: ቃል Sentiment Analysis", layout="centered")

# Load FastText model
@st.cache_resource
def load_model():
    return fasttext.load_model("models/amharic_fasttext_model.ftz")

model = load_model()

# Clean Amharic text (make sure it's same as used during training)
def clean_amharic(text):
    text = re.sub(r"[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF ]", "", text)
    return text.strip()

# Predict sentiment
def predict_sentiment(text):
    cleaned = clean_amharic(text)
    prediction = model.predict(cleaned)
    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]
    return label, confidence

# App header
st.title("📊 QalAnalyzer - Amharic Sentiment Classifier")
st.markdown("Analyze sentiment of Amharic text and improve the model via your feedback! 🚀")

# Input text
amharic_input = st.text_area("📝 Enter Amharic text here:")

if amharic_input:
    label, confidence = predict_sentiment(amharic_input)
    st.markdown(f"### Prediction: **{label.upper()}** ({confidence:.2%} confidence)")

    # Feedback section
    st.markdown("---")
    st.subheader("🤔 Was this prediction correct?")
    feedback = st.radio("Please give your feedback:", ["Yes", "No"], horizontal=True)

    if feedback == "No":
        correct_label = st.radio("What should the correct label be?", ["positive", "negative"], horizontal=True)

        if st.button("✅ Submit Feedback"):
            # Clean input
            cleaned_text = clean_amharic(amharic_input)
            correct_label_binary = 1 if correct_label == "positive" else 0
            feedback_data = pd.DataFrame([[cleaned_text, correct_label_binary]],
                                         columns=["cleaned_tweet", "label"])

            # Append to dataset
            DATA_PATH = "data/processed/amharic_sentiment_cleaned.csv"
            if os.path.exists(DATA_PATH):
                existing = pd.read_csv(DATA_PATH)
                combined = pd.concat([existing, feedback_data], ignore_index=True)
                combined.drop_duplicates(subset=["cleaned_tweet"], keep="last", inplace=True)
            else:
                combined = feedback_data

            combined.to_csv(DATA_PATH, index=False, encoding="utf-8")
            st.success("🎉 Thank you! Your feedback was added to the dataset.")

    elif feedback == "Yes":
        st.success("🙌 Awesome! Glad the prediction was correct.")

st.markdown("---")
st.caption("Created with ❤️ for Amharic NLP")


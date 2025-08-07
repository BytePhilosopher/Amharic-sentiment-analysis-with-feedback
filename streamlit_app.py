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
import numpy as np
import pickle
import re
import os
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from supabase import create_client, Client

# -------------------------------
# Supabase Credentials
# -------------------------------
SUPABASE_URL = "https://mtwqcqdshygmdtdtqfav.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im10d3FjcWRzaHlnbWR0ZHRxZmF2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ1ODU1ODUsImV4cCI6MjA3MDE2MTU4NX0.FjLLbxvrfB3VrzR-nfQcoM1THVJhzWJAxWOhl0WYKM4"

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

# -------------------------------
# Clean Amharic Text
# -------------------------------
def clean_amharic(text):
    text = re.sub(r"[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF ]", "", text)
    return text.strip()

# -------------------------------
# Load Vectorizer & Model
# -------------------------------
@st.cache_resource
def load_vectorizer():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    with open("models/sgd_model.pkl", "rb") as f:
        return pickle.load(f)

vectorizer = load_vectorizer()
model = load_model()

# -------------------------------
# Predict Sentiment
# -------------------------------
def predict_sentiment(text):
    cleaned = clean_amharic(text)
    features = vectorizer.transform([cleaned])
    label = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features))
    return "positive" if label == 1 else "negative", confidence

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="QalAnalyzer | ቃል Analyzer", layout="centered")
st.title("📊 QalAnalyzer - Amharic Sentiment Classifier")
st.markdown("Analyze Amharic sentiment and help us improve the model with your feedback!")

text_input = st.text_area("📝 Enter Amharic text here:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter Amharic text.")
    else:
        label, confidence = predict_sentiment(text_input)
        st.session_state.prediction = label
        st.session_state.confidence = confidence
        st.session_state.analyzed = True
        st.session_state.text_input = text_input

# -------------------------------
# Results + Feedback
# -------------------------------
if st.session_state.get("analyzed", False):
    st.markdown(f"### 🧠 Prediction: **{st.session_state.prediction.upper()}** ({st.session_state.confidence:.2%} confidence)")
    st.markdown("---")

    st.subheader("🤔 Was this prediction correct?")
    feedback = st.radio("Give your feedback:", ["Yes", "No"], horizontal=True)

    if feedback == "No":
        correct_label = st.radio("Select the correct label:", ["positive", "negative"], horizontal=True)

        if st.button("✅ Submit Feedback"):
            cleaned_text = clean_amharic(st.session_state.text_input)
            label_int = 1 if correct_label == "positive" else 0

            try:
                supabase.table("qal-analyzer").insert({
                    "cleaned_tweet": cleaned_text,
                    "label": label_int,
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.success("✅ Feedback saved to Supabase!")
            except Exception as e:
                st.error(f"❌ Error saving feedback: {e}")

            st.session_state.analyzed = False
            st.session_state.text_input = ""

    elif feedback == "Yes":
        st.success("🙌 Awesome! Glad it worked well.")

st.markdown("---")
st.caption("Created with ❤️ for Amharic NLP")

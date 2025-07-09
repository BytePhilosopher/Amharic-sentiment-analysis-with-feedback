import streamlit as st
import pickle

st.title("QalAnalyzer - ቃል Analyzer")

with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('../models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

text = st.text_area("Enter Amharic text")

if st.button("Analyze"):
    if text:
        clean_text = text.strip()
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)
        st.success(f"Sentiment: {prediction[0]}")
    else:
        st.warning("Please enter some text.")

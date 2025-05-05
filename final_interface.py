import pandas as pd
import re
import joblib
import streamlit as st
from streamlit.components.v1 import html

# Load the model and vectorizer
model_path = 'best_machine_model.joblib'
vectorizer_path = 'tfidf_vectorizer.joblib'  # Save your vectorizer during training if needed

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetical characters
    return text.lower()

# Function to predict if a tweet is depressed or not
def predict_depression(tweet):
    cleaned_tweet = clean_text(tweet)  # Clean the input tweet
    tweet_vector = vectorizer.transform([cleaned_tweet])  # Transform to TF-IDF
    prediction = model.predict(tweet_vector)  # Make prediction
    return "Depressed" if prediction[0] == 1 else "Not Depressed"

# Streamlit interface with custom HTML and CSS
st.set_page_config(page_title="Depression Detection", page_icon=":sparkles:", layout="centered")

# CSS styles
custom_css = """
<style>
html, body {
    font-family: 'Arial', sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 0;
}
.main-container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
h1 {
    color: #333;
    text-align: center;
    margin-bottom: 20px;
}
textarea {
    width: 100%;
    padding: 10px;
    font-size: 16px;
    border-radius: 4px;
    border: 1px solid #ddd;
    box-sizing: border-box;
    margin-bottom: 10px;
    background-color: #fafafa;
    color: #333;
}
textarea:focus {
    outline: none;
    border-color: #007BFF;
}
button {
    display: block;
    width: 100%;
    padding: 10px;
    font-size: 16px;
    color: white;
    background-color: #007BFF;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
}
button:hover {
    background-color: #0056b3;
}
.result {
    margin-top: 20px;
    padding: 10px;
    font-size: 18px;
    text-align: center;
    border-radius: 4px;
}
.result-depressed {
    color: white;
    background-color: #ff4d4d;
}
.result-not-depressed {
    color: white;
    background-color: #28a745;
}
.warning {
    color: #856404;
    background-color: #fff3cd;
    padding: 10px;
    border-radius: 4px;
    margin-top: 10px;
    text-align: center;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Main interface
#st.markdown('<div class="main-container">', unsafe_allow_html=True)
#st.markdown("<h1>Depression Detection from Tweets</h1>", unsafe_allow_html=True)
st.header("Depression Detection from Tweets")
st.write("Enter a tweet below to predict if it indicates signs of depression.")

# Input text box
input_tweet = st.text_area("Enter the tweet:")

if st.button("Predict"):
    if input_tweet.strip():
        result = predict_depression(input_tweet)
        result_class = "result-depressed" if result == "Depressed" else "result-not-depressed"
        st.markdown(f'<div class="result {result_class}">The tweet is classified as: <b>{result}</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning">Please enter a tweet to analyze.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

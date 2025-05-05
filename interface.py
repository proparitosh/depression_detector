import pandas as pd
import re
import joblib
import streamlit as st

# Load the model and vectorizer
model_path = 'best_model.joblib'
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

# Example usage
st.header("__Depression Detection__")
input_tweet = input("Enter your message:")
result = predict_depression(input_tweet)
print(f"The tweet is classified as: {result}")

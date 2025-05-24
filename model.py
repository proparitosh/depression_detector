import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

import streamlit as st  # Add this

@st.cache_resource
def train_model():
    df = pd.read_csv("depression_dataset.csv", encoding='utf-8')

    df['label'] = df['class'].map({'non-suicide': 0, 'suicide': 1})
    df = df.dropna(subset=['label'])  # ✅ Drop rows with NaN label

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', str(text))
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stopwords.words('english')]
        return " ".join(words)

    df['clean_text'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, vectorizer


def predict_depression(text, model, vectorizer):
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', str(text))
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stopwords.words('english')]
        return " ".join(words)

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "🚨 Depressed" if prediction[0] == 1 else "✅ Not Depressed"

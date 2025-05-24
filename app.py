import streamlit as st
from model import train_model, predict_depression

st.title("💬 Depression Detection App")

# Add training message
with st.spinner("⏳ Wait, model is training..."):
    model, vectorizer = train_model()

st.write("Training Done!")

# User input
user_input = st.text_input("Enter your message here:")

if st.button("Analyze"):
    result = predict_depression(user_input, model, vectorizer)
    st.write("Prediction:", result)

**Depression Detection on Social Media - Project Report**

** 1. Introduction**
 This project focuses on detecting signs of depression in social media text using Machine Learning. 
The application is built with Python and Streamlit, using NLTK for natural language preprocessing and
 Scikit-learn for modeling.
** 2. Technologies Used**
- Python- Streamlit (for UI)- NLTK (for text preprocessing)- Scikit-learn (for ML model)- Pandas (for data handling)
** 3. Project Files-** 
 app.py: Main Streamlit application for UI and input- model.py: Preprocessing and model training logic- depression_dataset.csv: Contains labeled text (0 - not depressed, 1 - depressed)- requirements.txt: List of required packages- README.md: Project overview and instructions
** 4. How It Works**
Depression Detection on Social Media - Project Report
 1. The dataset is cleaned and preprocessed using NLTK stopwords and punctuation removal.
 2. The data is vectorized using CountVectorizer.
 3. A Naive Bayes model is trained on this data.
 4. A web interface allows real-time prediction of user input text.
 5. Installation & Execution
   i. Install required packages: pip install -r requirements.txt
   ii. Run the application: streamlit run app.py
 6. Sample Code Snippet
 #app.py
 import streamlit as st
 from model import train_model, predict_depression
 model, vectorizer = train_model()
 st.title("Depression Detection from Social Media Text")
 text = st.text_area("Enter your message:")
 if st.button("Analyze"):
    prediction = predict_depression(text, model, vectorizer)

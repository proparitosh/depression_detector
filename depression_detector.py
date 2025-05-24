# Step 1: Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Step 2: Download NLTK Stopwords
nltk.download('stopwords')

# Step 3: Load Dataset
df = pd.read_csv("depression_dataset.csv", encoding='utf-8')
df = df[['text', 'class']]  # Only use relevant columns
df.dropna(inplace=True)     # Drop empty rows if any
print("✅ Dataset Loaded Successfully!")


# Step 4: Encode Labels
df['label'] = df['class'].map({'non-suicide': 0, 'suicide': 1})

# Step 5: Preprocess Text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))  # Remove non-letters
    text = text.lower()                          # Convert to lowercase
    words = text.split()                         # Split into words
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)
print("✅ Text Preprocessing Done")

# Step 6: Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['label']

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data Split")

# Step 8: Train the Model
model = MultinomialNB()
model.fit(X_train, y_train)
print("✅ Model Training Complete")

# Step 9: Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Predict Function
def predict_depression(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "🚨 Depressed" if prediction[0] == 1 else "✅ Not Depressed"

# Step 11: Looping User Input
print("\n🔍 Ready to analyze your messages.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a message: ")
    if user_input.lower() == "exit":
        print("👋 Exiting... Stay safe!")
        break
    try:
        print("🎯 Prediction:", predict_depression(user_input), "\n")
    except Exception as e:
        print("⚠️ Error with input:", e)

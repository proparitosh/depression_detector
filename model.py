import pandas as pd
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
data_path = 'Mental-Health-Twitter.csv'
df = pd.read_csv(data_path)

# Display first few rows of the dataset to understand its structure
print(df.head())

# Clean text function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetical characters
    return text.lower()

df['cleaned_text'] = df['post_text'].apply(clean_text)

# Check for null values and remove if any
df.dropna(subset=['cleaned_text', 'post_text','label'], inplace=True)

# Convert text to numerical data
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Path to save the trained model
model_path = 'depression_detection_model.joblib'
vectorizer_path = 'tfidf_vectorizer.joblib'

# Check if the model already exists
if os.path.exists(model_path):
    # Load the model from the file
    model = joblib.load(model_path)
    print("Model loaded from file, skipping training.")
else:
    # Train the model and save it
    model = LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print("Model trained and saved.")

joblib.dump(vectorizer, vectorizer_path)
print("Vectorizer saved.")
# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

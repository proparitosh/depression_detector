import pandas as pd
import sklearn
import xgboost
import re
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Paths for saving models and vectorizer
best_model_path = 'best_model.joblib'
vectorizer_path = 'tfidf_vectorizer.joblib'

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

# Feature engineering
def extract_features(df):
    df['text_length'] = df['post_text'].apply(len)  # Length of the post
    df['word_count'] = df['post_text'].apply(lambda x: len(x.split()))  # Number of words
    df['hashtag_count'] = df['post_text'].apply(lambda x: len(re.findall(r'#\w+', x)))  # Number of hashtags
    df['mention_count'] = df['post_text'].apply(lambda x: len(re.findall(r'@\w+', x)))  # Number of mentions
    if 'likes' in df.columns:
        df['likes'] = df['likes']  # Include likes if available
    if 'retweets' in df.columns:
        df['retweets'] = df['retweets']  # Include retweets if available
    if 'comments' in df.columns:
        df['comments'] = df['comments']  # Include comments if available
    return df

df = extract_features(df)

# Check for null values and remove if any
df.dropna(subset=['cleaned_text', 'post_text', 'label'], inplace=True)

# Convert text to numerical data with updated vectorizer settings
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
X_text = vectorizer.fit_transform(df['cleaned_text'])

# Additional numerical features
numerical_features = ['text_length', 'word_count', 'hashtag_count', 'mention_count']
if 'likes' in df.columns:
    numerical_features.append('likes')
if 'retweets' in df.columns:
    numerical_features.append('retweets')
if 'comments' in df.columns:
    numerical_features.append('comments')

additional_features = df[numerical_features]
scaler = StandardScaler()
X_additional = scaler.fit_transform(additional_features)

# Combine text and numerical features
X = hstack([X_text, X_additional])
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the vectorizer if not already saved
if not os.path.exists(vectorizer_path):
    joblib.dump(vectorizer, vectorizer_path)
    print("Vectorizer saved.")

# Check if the best model already exists
if os.path.exists(best_model_path):
    print("Loading the best model from disk...")
    best_model = joblib.load(best_model_path)
    print("Best model loaded.")
else:
    # Dictionary to hold models and their hyperparameter grids
    param_grids = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga'),
            "params": {
                "C": [0.1, 1, 10, 100]
            }
        },
        "Support Vector Machine": {
            "model": SVC(kernel='linear', class_weight='balanced', probability=True),
            "params": {
                "C": [0.1, 1, 10, 100]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        },
        "Neural Network (MLP)": {
            "model": MLPClassifier(max_iter=500, random_state=42),
            "params": {
                "hidden_layer_sizes": [(128,), (128, 64), (256, 128)],
                "learning_rate_init": [0.001, 0.01, 0.1]
            }
        }
    }

    # Train and evaluate models using GridSearchCV
    best_model = None
    best_accuracy = 0
    
    print("\nHyperparameter Tuning Results:")
    for model_name, model_info in param_grids.items():
        print(f"\nTuning {model_name}...")
        grid_search = GridSearchCV(estimator=model_info['model'], param_grid=model_info['params'],
                                   cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Best {model_name} Accuracy: {acc:.4f} with Params: {grid_search.best_params_}")

        # Update and save only the best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = best_estimator

    # Save the best model after identifying it
    if best_model is not None:
        joblib.dump(best_model, best_model_path)
        print(f"\nBest Model: {type(best_model).__name__} saved with Accuracy: {best_accuracy:.4f}")

# Use the loaded or newly trained best model for predictions
print("\nUsing the best model for predictions...")
y_pred = best_model.predict(X_test)
print(f"Accuracy of the loaded/trained best model: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

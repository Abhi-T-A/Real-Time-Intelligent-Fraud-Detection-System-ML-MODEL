import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading text data...")
df = pd.read_csv("data/raw/upi_fraud_dataset.csv")

# Use only text
df['text'] = df['text'].fillna("")  # avoid null error

X_text = df['text']
y = df['is_fraud']

print("Converting text to features...")
vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(X_text)

print("Training NLP model...")
model = LogisticRegression(max_iter=200)

model.fit(X, y)

print("Saving model and vectorizer...")
joblib.dump(model, "models/model_nlp.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

print("Model 4 trained successfully!")
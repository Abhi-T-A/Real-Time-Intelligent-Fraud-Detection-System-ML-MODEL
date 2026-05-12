import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading data...")
df = pd.read_csv("data/raw/upi_fraud_dataset.csv")

mapping = {
    'bill payment': 0,
    'transfer': 1,
    'recharge': 2,
    'merchant': 3
}

df['transaction_type'] = df['transaction_type'].map(mapping)
df['transaction_type'] = df['transaction_type'].fillna(0)

X = df[['amount', 'time_of_day', 'transaction_type']]
y = df['is_fraud']

print("Training model...")
model = RandomForestClassifier(n_estimators=100)

model.fit(X, y)

joblib.dump(model, "models/model_transaction.pkl")

print("Model 1 trained successfully!")
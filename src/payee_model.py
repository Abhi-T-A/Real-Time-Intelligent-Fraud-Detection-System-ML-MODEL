import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading data...")
df = pd.read_csv("data/raw/upi_fraud_dataset.csv")

#  Encode transaction_type (keep consistency)
mapping = {
    'bill payment': 0,
    'transfer': 1,
    'recharge': 2,
    'merchant': 3
}

df['transaction_type'] = df['transaction_type'].map(mapping)
df['transaction_type'] = df['transaction_type'].fillna(0)

# Payee features
features = ['new_payee', 'payee_age', 'payee_blacklisted', 'tx_to_payee']

# keep only existing columns (safe)
features = [col for col in features if col in df.columns]

X = df[features]
y = df['is_fraud']

print("Using features:", features)

print("Training Payee Risk Model...")
model = LogisticRegression(max_iter=200)

model.fit(X, y)

joblib.dump(model, "models/model_payee.pkl")

print("Model 3 trained successfully!")
from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib

print("Loading data...")
df = pd.read_csv("data/raw/upi_fraud_dataset.csv")

# Encode transaction_type
mapping = {
    'bill payment': 0,
    'transfer': 1,
    'recharge': 2,
    'merchant': 3
}

df['transaction_type'] = df['transaction_type'].map(mapping)
df['transaction_type'] = df['transaction_type'].fillna(0)

# SAME features as Model 1
X = df[['amount', 'time_of_day', 'transaction_type']]

print("Training Isolation Forest...")
model = IsolationForest(contamination=0.1, random_state=42)

model.fit(X)

joblib.dump(model, "models/model_anomaly.pkl")

print("Model 2 trained successfully!")
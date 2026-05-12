import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")
df = pd.read_csv("data/raw/upi_fraud_dataset.csv")

# -------- Add behavior features --------
# simulate user behavior (since dataset doesn't have it)
df["user_avg_amount"] = df["amount"].rolling(5, min_periods=1).mean()
df["user_tx_count_24h"] = np.random.randint(1, 10, size=len(df))

# -------- Feature engineering --------
df["amount_ratio"] = df["amount"] / (df["user_avg_amount"] + 1)

X = df[["amount_ratio", "user_tx_count_24h"]]
y = df["is_fraud"]

print("Training behavior model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# -------- Save --------
joblib.dump(model, "models/model_behavior.pkl")

print("Model 6 (Behavior) trained successfully!")
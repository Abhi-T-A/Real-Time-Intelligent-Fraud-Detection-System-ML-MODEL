import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading data...")
df = pd.read_csv("data/raw/upi_fraud_dataset.csv")

#Device / location features
features = ['device_new', 'device_change_freq', 'ip_mismatch', 'geo_distance']

# keep only available columns
features = [col for col in features if col in df.columns]

X = df[features]
y = df['is_fraud']

print("Using features:", features)

print("Training Device Risk Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X, y)

print("Saving model...")
joblib.dump(model, "models/model_device.pkl")

print("Model 5 trained successfully!")
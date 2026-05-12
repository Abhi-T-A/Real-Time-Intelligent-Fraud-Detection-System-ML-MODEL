import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -------- Load Models --------
print("Loading models...")

model1 = joblib.load("models/model_transaction.pkl")
model2 = joblib.load("models/model_anomaly.pkl")
model3 = joblib.load("models/model_payee.pkl")
model4 = joblib.load("models/model_nlp.pkl")
model5 = joblib.load("models/model_device.pkl")
model6 = joblib.load("models/model_behavior.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

# -------- Load Dataset --------
print("Loading dataset...")
df = pd.read_csv("data/raw/upi_fraud_dataset.csv")

# -------- Fix transaction type --------
mapping = {
    'bill payment': 0,
    'transfer': 1,
    'recharge': 2,
    'merchant': 3
}
df['transaction_type'] = df['transaction_type'].map(mapping).fillna(0)

# -------- Fill missing columns --------
required_cols = [
    'new_payee', 'payee_age', 'payee_blacklisted', 'tx_to_payee',
    'device_new', 'device_change_freq', 'ip_mismatch', 'geo_distance'
]

for col in required_cols:
    if col not in df.columns:
        df[col] = 0

# -------- Behavior Features --------
df["user_avg_amount"] = df["amount"].rolling(5, min_periods=1).mean()
df["user_tx_count_24h"] = np.random.randint(1, 10, len(df))

behavior = np.column_stack([
    df["amount"] / (df["user_avg_amount"] + 1),
    df["user_tx_count_24h"]
])

# -------- Graph --------
def build_graph(df):
    graph = {}
    for _, row in df.iterrows():
        payee = f"payee_{row['tx_to_payee']}"
        if payee not in graph:
            graph[payee] = {"transactions": 0, "fraud_count": 0}

        graph[payee]["transactions"] += 1
        if row["is_fraud"] == 1:
            graph[payee]["fraud_count"] += 1
    return graph

def compute_graph_risk(graph, payee_id):
    payee = f"payee_{payee_id}"
    if payee not in graph:
        return 0.1
    fraud_ratio = graph[payee]["fraud_count"] / (graph[payee]["transactions"] + 1)
    return min(fraud_ratio, 1.0)

print("Building graph...")
graph = build_graph(df)
graph_scores = df["tx_to_payee"].apply(lambda x: compute_graph_risk(graph, x)).values

# -------- Features --------
X_tx = df[['amount', 'time_of_day', 'transaction_type']]
X_payee = df[['new_payee', 'payee_age', 'payee_blacklisted', 'tx_to_payee']]
X_device = df[['device_new', 'device_change_freq', 'ip_mismatch', 'geo_distance']]

df['text'] = df['text'].fillna("")
X_text = vectorizer.transform(df['text'])

y = df['is_fraud']

print("Generating model predictions...")

# Model outputs
r1 = model1.predict_proba(X_tx)[:, 1]

r2_raw = model2.decision_function(X_tx)
r2 = (r2_raw - r2_raw.min()) / (r2_raw.max() - r2_raw.min() + 1e-6)

r3 = model3.predict_proba(X_payee)[:, 1]
r4 = model4.predict_proba(X_text)[:, 1]
r5 = model5.predict_proba(X_device)[:, 1]
r6 = model6.predict_proba(behavior)[:, 1]

# -------- Meta Input --------
meta_X = np.column_stack((r1, r2, r3, r4, r5, r6, graph_scores))

# -------- Train --------
print("Training Meta Model...")

meta_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

meta_model.fit(meta_X, y)

# -------- Save --------
print("Saving meta model...")
joblib.dump(meta_model, "models/model_meta.pkl")

print("Meta Model trained successfully!")
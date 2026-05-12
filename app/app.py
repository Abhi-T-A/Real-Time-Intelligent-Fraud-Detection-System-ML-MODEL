from flask import Flask, request, jsonify
from app.otp_fraud_model import otp_fraud_model  # type: ignore
import joblib
import numpy as np
import pandas as pd
import os
import time

app = Flask(__name__)

# -------- User History -------- 
user_history = {}

def update_user_history(user_id, data):
    if user_id not in user_history:
        user_history[user_id] = {
            "total_tx": 0,
            "total_amount": 0,
            "avg_amount": 0,
            "last_location": 0,
            "devices": set(),
            "payees": set(),
            "last_tx_time": time.time(),
            "tx_times": []
        }

    user = user_history[user_id]

    user["total_tx"] += 1
    user["total_amount"] += data.get("amount", 0)
    user["avg_amount"] = user["total_amount"] / user["total_tx"]

    user["last_location"] = data.get("geo_distance", 0)
    user["devices"].add(data.get("device_new", 0))
    user["payees"].add(data.get("tx_to_payee", 0))

    current_time = time.time()
    user["tx_times"].append(current_time)
    user["last_tx_time"] = current_time


# -------- Normalize --------
def normalize(x):
    x = np.array(x)
    if x.max() == x.min():
        return np.zeros_like(x) + 0.5
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


# -------- Load Models --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model1 = joblib.load(os.path.join(BASE_DIR, "models/model_transaction.pkl"))
model2 = joblib.load(os.path.join(BASE_DIR, "models/model_anomaly.pkl"))
model3 = joblib.load(os.path.join(BASE_DIR, "models/model_payee.pkl"))
model4 = joblib.load(os.path.join(BASE_DIR, "models/model_nlp.pkl"))
model5 = joblib.load(os.path.join(BASE_DIR, "models/model_device.pkl"))
model6 = joblib.load(os.path.join(BASE_DIR, "models/model_behavior.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models/tfidf.pkl"))


# -------- Graph --------
df_graph = pd.read_csv("data/raw/upi_fraud_dataset.csv")

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

graph = build_graph(df_graph)


def compute_graph_risk(graph, payee_id):
    payee = f"payee_{payee_id}"
    if payee not in graph:
        return 0.1

    fraud_ratio = graph[payee]["fraud_count"] / (graph[payee]["transactions"] + 1)
    return min(fraud_ratio, 1.0)


# -------- LOCATION ANOMALY --------
def location_risk(user_id, data):
    if user_id not in user_history:
        return 0.1, None

    user = user_history[user_id]

    current_loc = data.get("geo_distance", 0)
    last_loc = user.get("last_location", 0)

    distance_change = abs(current_loc - last_loc)

    if distance_change > 1000:
        return 0.5, "Location changed drastically (>1000 km)"
    elif distance_change > 500:
        return 0.3, "Unusual location change (>500 km)"
    elif distance_change > 200:
        return 0.2, "Moderate location change"
    
    return 0, None


# -------- ALL ML MODELS --------
def run_all_models(data):
    type_map = {"bill payment":0,"transfer":1,"recharge":2,"merchant":3}

    tx = np.array([[data.get("amount",0),
                    data.get("time_of_day",0),
                    type_map.get(data.get("transaction_type",""),0)]])

    payee = np.array([[data.get("new_payee",0),
                       data.get("payee_age",0),
                       data.get("payee_blacklisted",0),
                       data.get("tx_to_payee",0)]])

    device = np.array([[data.get("device_new",0),
                        data.get("device_change_freq",0),
                        data.get("ip_mismatch",0),
                        data.get("geo_distance",0)]])

    behavior = np.array([[data.get("amount",0)/(data.get("user_avg_amount",2000)+1),
                          data.get("user_tx_count_24h",1)]])

    text_vec = vectorizer.transform([data.get("text","")])

    r1 = model1.predict_proba(tx)[:,1][0]
    r2 = normalize(model2.decision_function(tx))[0]
    r3 = model3.predict_proba(payee)[:,1][0]
    r4 = model4.predict_proba(text_vec)[:,1][0]
    r5 = model5.predict_proba(device)[:,1][0]
    r6 = model6.predict_proba(behavior)[:,1][0]

    return {
        "transaction": r1,
        "anomaly": r2,
        "payee": r3,
        "nlp": r4,
        "device": r5,
        "behavior": r6
    }

def final_decision(data):
    user_id = data.get("user_id", "default_user")

    # 1. LOCATION CHECK (BEFORE UPDATE)
    loc_score, loc_reason = location_risk(user_id, data)

    # 2. UPDATE HISTORY
    update_user_history(user_id, data)

    # 3. RUN MODELS
    scores = run_all_models(data)
    otp_result = otp_fraud_model(data)

    # 4. ML SCORE
    ml_score = sum(scores.values()) / len(scores)

    #  5. FINAL SCORE
    final_score = (
        0.4 * ml_score +
        0.4 * otp_result["otp_risk_score"] +
        0.2 * loc_score
    )

    # 6. REASONS (START FROM OTP)
    reasons = otp_result["otp_reasons"].copy()

    actions="BLOCK" if final_score>0.7 else "ALLOW"

    # ADD LOCATION REASON
    if loc_reason:
        reasons.append(loc_reason)

    # ADD COMBINED LOGIC (YOUR LINE)
    if loc_score > 0 and data.get("device_new") == 1:
        reasons.append("New device + new location detected")

def final_decision(data):
    user_id = data.get("user_id", "default_user")

    # 1. LOCATION CHECK (BEFORE UPDATE)
    loc_score, loc_reason = location_risk(user_id, data)

    #  2. UPDATE HISTORY
    update_user_history(user_id, data)

    #  3. RUN MODELS
    scores = run_all_models(data)
    otp_result = otp_fraud_model(data)

    #  4. ML SCORE
    ml_score = sum(scores.values()) / len(scores)

    #  5. FINAL SCORE
    final_score = (
        0.4 * ml_score +
        0.4 * otp_result["otp_risk_score"] +
        0.2 * loc_score
    )

    #  6. REASONS
    reasons = otp_result["otp_reasons"].copy()

    #  LOCATION REASON
    if loc_reason:
        reasons.append(loc_reason)

    #  COMBINED LOGIC
    if loc_score > 0 and data.get("device_new") == 1:
        reasons.append("New device + new location detected")

    #  7. RISK LEVEL
    if final_score > 0.7:
        level = "HIGH"
    elif final_score > 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"

    #  8. ACTION
    action = "BLOCK" if final_score > 0.7 else "ALLOW"

    #  9. RETURN RESPONSE
    return {
        "risk_score": round(final_score, 2),
        "risk_level": level,
        "action": action,
        "otp_score": otp_result["otp_risk_score"],
        "location_score": loc_score,
        "reasons": reasons,
        "model_scores": scores
    }
# -------- API --------
@app.route("/")
def home():
    return "Fraud Detection API Running"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data"}), 400

    result = final_decision(data)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
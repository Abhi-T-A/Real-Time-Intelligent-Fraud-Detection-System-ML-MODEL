import pandas as pd
import numpy as np
import random

n = 50000  # number of rows (increase to 100k if needed)

data = []

scam_texts = [
    "urgent send money",
    "share otp now",
    "verify account immediately",
    "click this link to claim prize",
    "your account blocked send money",
    "lottery winner send fees"
]

normal_texts = [
    "payment done",
    "received money",
    "grocery payment",
    "rent paid",
    "thanks",
    "bill payment"
]

for _ in range(n):

    amount = np.random.randint(10, 20000)
    time_of_day = np.random.randint(0, 24)
    transaction_type = np.random.randint(1, 4)

    new_payee = np.random.choice([0,1], p=[0.7,0.3])
    payee_age = np.random.randint(0, 60)
    payee_blacklisted = np.random.choice([0,1], p=[0.9,0.1])
    tx_to_payee = np.random.randint(1, 15)

    device_new = np.random.choice([0,1], p=[0.8,0.2])
    device_change_freq = np.random.randint(1, 10)
    ip_mismatch = np.random.choice([0,1], p=[0.85,0.15])
    geo_distance = np.random.randint(1, 1500)

    # Fraud logic (important)
    fraud = 0

    if (
        amount > 8000 and new_payee == 1
        or payee_blacklisted == 1
        or ip_mismatch == 1 and geo_distance > 500
    ):
        fraud = 1

    # Text generation
    if fraud == 1:
        text = random.choice(scam_texts)
    else:
        text = random.choice(normal_texts)

    data.append([
        amount, time_of_day, transaction_type,
        new_payee, payee_age, payee_blacklisted, tx_to_payee,
        device_new, device_change_freq, ip_mismatch, geo_distance,
        text, fraud
    ])

columns = [
    "amount","time_of_day","transaction_type",
    "new_payee","payee_age","payee_blacklisted","tx_to_payee",
    "device_new","device_change_freq","ip_mismatch","geo_distance",
    "text","is_fraud"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/raw/upi_fraud_dataset.csv", index=False)

print("UPI Fraud Dataset Generated!")
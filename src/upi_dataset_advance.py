"""
generate_upi_dataset_advanced.py
=================================
Advanced UPI Fraud Dataset Generator for Indian Payment Apps
Simulates realistic fraud patterns for PhonePe, Google Pay, Paytm ecosystems.

Fraud logic is rule-based — NOT random — to reflect real-world attack patterns.
"""

import os
import numpy as np
import pandas as pd
import random
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RANDOM_SEED = 42
N_ROWS = 100_000          # Configurable dataset size
FRAUD_RATIO = 0.25        # 25% fraud cases (within 20–30% target)
OUTPUT_PATH = "data/raw/upi_fraud_dataset.csv"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# TEXT TEMPLATES
# ─────────────────────────────────────────────

SCAM_TEXTS = [
    "URGENT: Your account will be blocked. Send ₹1 now to verify.",
    "Share your OTP immediately to avoid account suspension.",
    "Congratulations! You won ₹50,000. Send ₹500 processing fee.",
    "Your UPI ID is compromised. Transfer funds to safe account now.",
    "KYC pending — account blocked in 24 hrs. Click link to update.",
    "Bank official: send money to this number to unlock your account.",
    "Emergency! Your family member is in hospital. Send money ASAP.",
    "Refund processing failed. Share PIN to re-initiate your refund.",
    "Electricity bill unpaid — service will be cut in 2 hours. Pay now.",
    "Send ₹1 to verify your UPI and receive ₹10,000 cashback.",
    "Your account shows suspicious activity. Send OTP to confirm it's you.",
    "UPI limit exceeded. Pay ₹100 admin fee to restore transactions.",
    "Job offer confirmed. Pay registration fee to HR via UPI.",
    "Lottery winner! Send ₹200 tax to claim your ₹1,00,000 prize.",
    "Urgent: send money now — will return double in 10 minutes.",
    "Your Aadhaar is linked to fraud case. Pay to clear your name.",
    "SIM card will be blocked. Share details with telecom executive.",
    "NEFT failed — verify account by sending ₹1 to this UPI ID.",
    "Custom duty pending on your parcel. Pay ₹300 to release it.",
    "Your GPay account flagged. Send money to new ID to avoid freeze.",
]

NORMAL_TEXTS = [
    "Paid electricity bill for the month.",
    "Recharge done for Jio prepaid plan.",
    "Split the restaurant bill with friends.",
    "Transferred rent to landlord.",
    "Paid grocery at D-Mart via UPI.",
    "Sent money to mom for household expenses.",
    "Paid OLA cab fare.",
    "Swiggy order payment done.",
    "Google Play subscription renewed.",
    "Zomato food order paid.",
    "Petrol pump payment at HP station.",
    "School fee paid via net banking UPI.",
    "LIC premium payment for this quarter.",
    "Transferred money to friend for shared trip.",
    "Paid medical bill at Apollo Pharmacy.",
    "Bought train ticket via IRCTC UPI.",
    "Paid for gym membership renewal.",
    "Amazon order payment completed.",
    "DTH recharge done for Tata Play.",
    "Water bill paid to municipal corporation.",
    "Paid plumber for home repair work.",
    "Transferred salary advance to self.",
    "Paid for freelance project milestone.",
    "Coffee at Café Coffee Day.",
    "Transferred to savings wallet.",
]

TRANSACTION_TYPES = ["bill payment", "transfer", "recharge", "merchant"]


# ─────────────────────────────────────────────
# HELPER GENERATORS
# ─────────────────────────────────────────────

def random_time_of_day(is_night: bool = False) -> float:
    """Return hour as float (0–23.99). Bias toward night if is_night=True."""
    if is_night:
        # Night window: 22:00–05:00
        hour = random.choice(
            list(np.arange(22, 24, 0.1)) + list(np.arange(0, 5, 0.1))
        )
    else:
        # Daytime: 06:00–21:59
        hour = np.random.uniform(6, 22)
    return round(float(hour), 2)


def is_night(time_of_day: float) -> int:
    """Return 1 if hour is between 22:00 and 05:00."""
    return int(time_of_day >= 22 or time_of_day < 5)


def device_trust_score(device_new: int, device_change_freq: float, ip_mismatch: int) -> float:
    """
    Composite device trust score (0–1).
    Lower score = higher device risk.
    """
    score = 1.0
    if device_new:
        score -= 0.35
    score -= min(device_change_freq / 10, 0.3)  # High freq → low trust
    if ip_mismatch:
        score -= 0.25
    return round(max(0.0, min(1.0, score + np.random.normal(0, 0.05))), 3)


# ─────────────────────────────────────────────
# FRAUD SCENARIO BUILDERS
# ─────────────────────────────────────────────

def build_scam_message_fraud() -> dict:
    """
    Scenario 1 — Scam Message Fraud
    Victim receives urgent/threatening text, transfers to new unknown payee.
    Signals: scam text + new_payee + high amount.
    """
    tod = random_time_of_day(is_night=False)
    amount = round(np.random.uniform(500, 25000), 2)
    d_new = 1
    ip_mis = random.choice([0, 1])
    geo = round(np.random.uniform(0, 50), 2)
    dcf = round(np.random.uniform(0, 3), 2)
    return {
        "amount": amount,
        "time_of_day": tod,
        "transaction_type": random.choice(["transfer", "bill payment"]),
        "new_payee": 1,
        "payee_age": round(np.random.uniform(0, 30), 1),      # Very new payee
        "payee_blacklisted": random.choice([0, 0, 1]),         # Often not yet blacklisted
        "tx_to_payee": random.randint(0, 2),
        "device_new": d_new,
        "device_change_freq": dcf,
        "ip_mismatch": ip_mis,
        "geo_distance": geo,
        "text": random.choice(SCAM_TEXTS),
        "is_night_txn": is_night(tod),
        "amount_spike": round(np.random.uniform(3.0, 10.0), 2),
        "txn_velocity": round(np.random.uniform(3, 15), 2),
        "location_change": random.choice([0, 1]),
        "device_trust_score": device_trust_score(d_new, dcf, ip_mis),
        "upi_id_age": round(np.random.uniform(0, 90), 1),
        "is_fraud": 1,
    }


def build_night_fraud() -> dict:
    """
    Scenario 2 — Night-Time Fraud
    Large transfer to young/unknown payee in late-night hours.
    Signals: time 22–05 + high amount + low payee_age.
    """
    tod = random_time_of_day(is_night=True)
    amount = round(np.random.uniform(5000, 100000), 2)
    d_new = random.choice([0, 1])
    ip_mis = random.choice([0, 1])
    dcf = round(np.random.uniform(0, 4), 2)
    return {
        "amount": amount,
        "time_of_day": tod,
        "transaction_type": random.choice(["transfer", "merchant"]),
        "new_payee": random.choice([0, 1]),
        "payee_age": round(np.random.uniform(0, 60), 1),       # Often young payee
        "payee_blacklisted": random.choice([0, 1]),
        "tx_to_payee": random.randint(0, 3),
        "device_new": d_new,
        "device_change_freq": dcf,
        "ip_mismatch": ip_mis,
        "geo_distance": round(np.random.uniform(10, 200), 2),
        "text": random.choice(SCAM_TEXTS + NORMAL_TEXTS),
        "is_night_txn": 1,
        "amount_spike": round(np.random.uniform(2.5, 8.0), 2),
        "txn_velocity": round(np.random.uniform(1, 8), 2),
        "location_change": random.choice([0, 1]),
        "device_trust_score": device_trust_score(d_new, dcf, ip_mis),
        "upi_id_age": round(np.random.uniform(0, 180), 1),
        "is_fraud": 1,
    }


def build_device_takeover_fraud() -> dict:
    """
    Scenario 3 — Device Takeover / SIM Swap Fraud
    Attacker uses a new device with mismatched IP and geographic jump.
    Signals: device_new=1 + ip_mismatch=1 + large geo_distance.
    """
    tod = random_time_of_day()
    amount = round(np.random.uniform(1000, 80000), 2)
    d_new = 1
    ip_mis = 1
    dcf = round(np.random.uniform(3, 10), 2)     # Rapid device changes
    geo = round(np.random.uniform(100, 2000), 2)  # Large geographic jump
    return {
        "amount": amount,
        "time_of_day": tod,
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "new_payee": random.choice([0, 1]),
        "payee_age": round(np.random.uniform(0, 365), 1),
        "payee_blacklisted": random.choice([0, 1]),
        "tx_to_payee": random.randint(0, 5),
        "device_new": d_new,
        "device_change_freq": dcf,
        "ip_mismatch": ip_mis,
        "geo_distance": geo,
        "text": random.choice(NORMAL_TEXTS + SCAM_TEXTS),
        "is_night_txn": is_night(tod),
        "amount_spike": round(np.random.uniform(1.5, 6.0), 2),
        "txn_velocity": round(np.random.uniform(2, 12), 2),
        "location_change": 1,
        "device_trust_score": device_trust_score(d_new, dcf, ip_mis),
        "upi_id_age": round(np.random.uniform(0, 365), 1),
        "is_fraud": 1,
    }


def build_small_to_large_fraud() -> dict:
    """
    Scenario 4 — Small-to-Large Escalation Pattern
    Attacker tests with micro-transactions, then executes large transfer.
    Signals: high txn_velocity + high amount_spike + high amount.
    """
    tod = random_time_of_day()
    amount = round(np.random.uniform(10000, 150000), 2)
    d_new = random.choice([0, 1])
    ip_mis = random.choice([0, 1])
    dcf = round(np.random.uniform(0, 3), 2)
    return {
        "amount": amount,
        "time_of_day": tod,
        "transaction_type": "transfer",
        "new_payee": random.choice([0, 1]),
        "payee_age": round(np.random.uniform(0, 180), 1),
        "payee_blacklisted": random.choice([0, 1]),
        "tx_to_payee": random.randint(3, 20),       # Many prior txns (testing phase)
        "device_new": d_new,
        "device_change_freq": dcf,
        "ip_mismatch": ip_mis,
        "geo_distance": round(np.random.uniform(0, 100), 2),
        "text": random.choice(SCAM_TEXTS + NORMAL_TEXTS),
        "is_night_txn": is_night(tod),
        "amount_spike": round(np.random.uniform(5.0, 20.0), 2),   # Huge spike vs average
        "txn_velocity": round(np.random.uniform(8, 30), 2),        # High velocity
        "location_change": random.choice([0, 1]),
        "device_trust_score": device_trust_score(d_new, dcf, ip_mis),
        "upi_id_age": round(np.random.uniform(0, 730), 1),
        "is_fraud": 1,
    }


def build_blacklisted_payee_fraud() -> dict:
    """
    Scenario 5 — Blacklisted Payee Fraud
    Transaction directed to a known fraudulent UPI ID.
    Signals: payee_blacklisted=1 (primary signal).
    """
    tod = random_time_of_day()
    amount = round(np.random.uniform(100, 50000), 2)
    d_new = random.choice([0, 1])
    ip_mis = random.choice([0, 1])
    dcf = round(np.random.uniform(0, 5), 2)
    return {
        "amount": amount,
        "time_of_day": tod,
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "new_payee": random.choice([0, 1]),
        "payee_age": round(np.random.uniform(0, 365), 1),
        "payee_blacklisted": 1,                                   # Definitive fraud signal
        "tx_to_payee": random.randint(0, 10),
        "device_new": d_new,
        "device_change_freq": dcf,
        "ip_mismatch": ip_mis,
        "geo_distance": round(np.random.uniform(0, 500), 2),
        "text": random.choice(SCAM_TEXTS + NORMAL_TEXTS),
        "is_night_txn": is_night(tod),
        "amount_spike": round(np.random.uniform(0.5, 5.0), 2),
        "txn_velocity": round(np.random.uniform(0, 10), 2),
        "location_change": random.choice([0, 1]),
        "device_trust_score": device_trust_score(d_new, dcf, ip_mis),
        "upi_id_age": round(np.random.uniform(0, 1095), 1),
        "is_fraud": 1,
    }


FRAUD_SCENARIOS = [
    build_scam_message_fraud,
    build_night_fraud,
    build_device_takeover_fraud,
    build_small_to_large_fraud,
    build_blacklisted_payee_fraud,
]


# ─────────────────────────────────────────────
# LEGITIMATE TRANSACTION BUILDER
# ─────────────────────────────────────────────

def build_legitimate_transaction() -> dict:
    """
    Generate a realistic, non-fraudulent UPI transaction.
    All signals should be within normal ranges.
    """
    tod = random_time_of_day(is_night=False)
    amount = round(np.random.lognormal(mean=6.5, sigma=1.2), 2)  # Log-normal ≈ real spend
    amount = round(min(amount, 50000), 2)
    d_new = random.choices([0, 1], weights=[85, 15])[0]
    ip_mis = random.choices([0, 1], weights=[92, 8])[0]
    dcf = round(abs(np.random.normal(0.5, 0.5)), 2)
    return {
        "amount": amount,
        "time_of_day": tod,
        "transaction_type": random.choices(
            TRANSACTION_TYPES, weights=[30, 35, 20, 15]
        )[0],
        "new_payee": random.choices([0, 1], weights=[75, 25])[0],
        "payee_age": round(np.random.uniform(90, 1825), 1),     # Established payees
        "payee_blacklisted": 0,
        "tx_to_payee": random.randint(1, 50),
        "device_new": d_new,
        "device_change_freq": dcf,
        "ip_mismatch": ip_mis,
        "geo_distance": round(abs(np.random.normal(5, 10)), 2),
        "text": random.choice(NORMAL_TEXTS),
        "is_night_txn": is_night(tod),
        "amount_spike": round(abs(np.random.normal(1.0, 0.4)), 2),
        "txn_velocity": round(abs(np.random.normal(1.5, 1.0)), 2),
        "location_change": random.choices([0, 1], weights=[90, 10])[0],
        "device_trust_score": device_trust_score(d_new, dcf, ip_mis),
        "upi_id_age": round(np.random.uniform(180, 3650), 1),
        "is_fraud": 0,
    }


# ─────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────

def generate_dataset(n_rows: int = N_ROWS, fraud_ratio: float = FRAUD_RATIO) -> pd.DataFrame:
    """
    Generate the full UPI fraud dataset.

    Strategy:
    - Generate fraud_ratio% fraud rows using weighted random scenario selection.
    - Generate remainder as legitimate transactions.
    - Shuffle and reset index.
    """
    n_fraud = int(n_rows * fraud_ratio)
    n_legit = n_rows - n_fraud

    print(f"[INFO] Generating {n_rows:,} rows | Fraud: {n_fraud:,} | Legit: {n_legit:,}")

    # --- Fraud records (equal distribution across 5 scenarios) ---
    fraud_records = []
    scenario_weights = [0.25, 0.20, 0.20, 0.20, 0.15]  # Weighted by real-world prevalence

    for _ in range(n_fraud):
        scenario_fn = random.choices(FRAUD_SCENARIOS, weights=scenario_weights, k=1)[0]
        fraud_records.append(scenario_fn())

    # --- Legitimate records ---
    legit_records = [build_legitimate_transaction() for _ in range(n_legit)]

    # --- Combine and shuffle ---
    all_records = fraud_records + legit_records
    random.shuffle(all_records)

    df = pd.DataFrame(all_records)

    # ── Post-processing: enforce data types and clamp ranges ──
    df["amount"] = df["amount"].clip(lower=1.0).round(2)
    df["time_of_day"] = df["time_of_day"].clip(0, 23.99).round(2)
    df["payee_age"] = df["payee_age"].clip(lower=0).round(1)
    df["tx_to_payee"] = df["tx_to_payee"].clip(lower=0).astype(int)
    df["device_change_freq"] = df["device_change_freq"].clip(lower=0).round(2)
    df["geo_distance"] = df["geo_distance"].clip(lower=0).round(2)
    df["amount_spike"] = df["amount_spike"].clip(lower=0).round(2)
    df["txn_velocity"] = df["txn_velocity"].clip(lower=0).round(2)
    df["upi_id_age"] = df["upi_id_age"].clip(lower=0).round(1)
    df["device_trust_score"] = df["device_trust_score"].clip(0, 1).round(3)

    # Ensure boolean/binary columns are integer 0/1
    for col in ["new_payee", "payee_blacklisted", "device_new",
                "ip_mismatch", "is_night_txn", "location_change", "is_fraud"]:
        df[col] = df[col].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> None:
    """Run basic quality checks and print a summary report."""
    print("\n" + "═" * 55)
    print("  DATASET VALIDATION REPORT")
    print("═" * 55)
    print(f"  Total rows        : {len(df):,}")
    print(f"  Total columns     : {len(df.columns)}")

    fraud_pct = df["is_fraud"].mean() * 100
    print(f"  Fraud rows        : {df['is_fraud'].sum():,}  ({fraud_pct:.1f}%)")
    print(f"  Legitimate rows   : {(df['is_fraud'] == 0).sum():,}  ({100 - fraud_pct:.1f}%)")

    missing = df.isnull().sum().sum()
    print(f"  Missing values    : {missing}")

    print(f"\n  Amount range      : ₹{df['amount'].min():.2f} – ₹{df['amount'].max():,.2f}")
    print(f"  Avg amount (fraud): ₹{df.loc[df.is_fraud==1,'amount'].mean():,.2f}")
    print(f"  Avg amount (legit): ₹{df.loc[df.is_fraud==0,'amount'].mean():,.2f}")

    print(f"\n  Fraud scenario mix (approx):")
    print(f"    Blacklisted payee: {df.loc[df.is_fraud==1,'payee_blacklisted'].mean()*100:.1f}%")
    print(f"    Night txns (fraud): {df.loc[df.is_fraud==1,'is_night_txn'].mean()*100:.1f}%")
    print(f"    Device takeover  : {df.loc[df.is_fraud==1,'device_new'].mean()*100:.1f}%")

    print(f"\n  Transaction type distribution:")
    print(df["transaction_type"].value_counts(normalize=True).mul(100).round(1).to_string())
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Generate
    df = generate_dataset(n_rows=N_ROWS, fraud_ratio=FRAUD_RATIO)

    # Validate
    validate_dataset(df)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Dataset saved → {OUTPUT_PATH}")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")


if __name__ == "__main__":
    main()
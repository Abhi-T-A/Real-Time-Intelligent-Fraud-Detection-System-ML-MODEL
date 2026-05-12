def otp_fraud_model(data):
    risk = 0
    reasons = []

    # Call + OTP
    if data.get("call_active") and data.get("is_otp"):
        risk += 0.4
        reasons.append("User likely shared OTP during active call")

    #  Device
    if data.get("device_new"):
        risk += 0.2
        reasons.append("Transaction from new device")

    if data.get("is_otp_verified") and data.get("device_new"):
        risk += 0.3
        reasons.append("OTP used on different device")

    # New payee
    if data.get("new_payee"):
        risk += 0.1
        reasons.append("New payee added")

    #  High amount
    if data.get("amount", 0) > 50000:
        risk += 0.1
        reasons.append("High transaction amount")

    return {
        "otp_risk_score": min(risk, 1.0),
        "otp_flag": risk > 0.5,
        "otp_reasons": reasons
    }
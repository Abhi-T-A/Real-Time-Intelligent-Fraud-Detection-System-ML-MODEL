# Real-Time-Intelligent-Fraud-Detection-System-ML-Model

A machine learning based real-time fraud detection system designed to detect suspicious UPI/payment transactions using multiple intelligent models and behavioral analysis techniques.

---

## Features

- Real-time fraud prediction
- OTP fraud detection
- Device behavior analysis
- Transaction anomaly detection
- NLP-based suspicious text analysis
- Payee risk analysis
- Graph-based fraud risk evaluation
- Multiple ML models integration
- Dataset preprocessing and training pipeline

---

## Tech Stack

### Backend / ML
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy

### Machine Learning
- Anomaly Detection
- Behavioral Analysis
- NLP Classification
- Transaction Risk Scoring

### Dataset
- UPI Fraud Dataset (CSV)

---

## Project Structure

```bash
Real-Time-Intelligent-Fraud-Detection-System-ML-MODEL/
│
├── app/
│   ├── app.py
│   ├── graph_risk.py
│   └── otp_fraud_model.py
│
├── data/
│   └── raw/
│       └── upi_fraud_dataset.csv
│
├── src/
│   ├── anomaly_model.py
│   ├── bahavior_model.py
│   ├── device_model.py
│   ├── generate_upi_dataset.py
│   ├── nlp_model.py
│   ├── payee_model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train_meta.py
│   ├── transaction_model.py
│   └── upi_dataset_advance.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Real-Time-Intelligent-Fraud-Detection-System-ML-MODEL.git
```

### Navigate to Project

```bash
cd Real-Time-Intelligent-Fraud-Detection-System-ML-MODEL
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python app/app.py
```

---

## Machine Learning Modules

| Module | Purpose |
|---|---|
| anomaly_model.py | Detect abnormal transactions |
| behavior_model.py | Analyze user behavior |
| device_model.py | Device fingerprint/risk analysis |
| nlp_model.py | NLP fraud keyword analysis |
| payee_model.py | Suspicious payee detection |
| transaction_model.py | Transaction fraud scoring |
| otp_fraud_model.py | OTP scam prediction |
| graph_risk.py | Fraud network relationship analysis |

---

## Future Enhancements

- Real-time dashboard
- Deep learning fraud detection
- Mobile app integration
- API deployment
- Blockchain verification
- AI-powered adaptive risk engine

---

## Author

Abhi T A

---

## License

This project is developed for educational and research purposes.

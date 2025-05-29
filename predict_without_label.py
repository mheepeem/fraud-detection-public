import pandas as pd
import joblib

# === Load test data ===
df = pd.read_parquet("datas/test_transaction_for_prediction.parquet")

# === Load model and threshold ===
model = joblib.load("model/catboost_fraud_model.pkl")
with open("model/best_threshold.txt") as f:
    threshold = float(f.read().strip())

# === Predict probabilities
proba = model.predict_proba(df)[:, 1]

# === Apply threshold
df["fraud_probability"] = proba
df["predicted_isFraud"] = (proba >= threshold).astype(int)

# === Print top predictions
print("🔍 Sample Predictions:")
print(df[["fraud_probability", "predicted_isFraud"]].head(10))

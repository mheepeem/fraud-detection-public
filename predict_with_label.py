import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Load Data & Model ===
df = pd.read_parquet("datas/test_transaction_cleaned.parquet")
model = joblib.load("model/catboost_fraud_model.pkl")

# === Load threshold ===
with open("model/best_threshold.txt") as f:
    threshold = float(f.read().strip())

# === Separate features and target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# === Predict probabilities
proba = model.predict_proba(X)[:, 1]

# === Apply threshold
y_pred = (proba >= threshold).astype(int)

# === Metrics
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# === Show Results
print(f"🔎 Evaluation at threshold = {threshold:.4f}")
print(f"✔️ Accuracy:  {acc:.4f}")
print(f"✔️ Precision: {prec:.4f}")
print(f"✔️ Recall:    {rec:.4f}")
print(f"✔️ F1 Score:  {f1:.4f}")

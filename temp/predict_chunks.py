import pandas as pd
from catboost import CatBoostClassifier

# === Define paths based on Processing Job mount points ===
INPUT_DATA_PATH = "/opt/ml/processing/input/test.parquet"
MODEL_PATH = "/opt/ml/processing/input/catboost_fraud_model.cbm"
OUTPUT_PATH = "/opt/ml/processing/output/predicted_output.parquet"

# === Load test data ===
print("🔹 Loading test data...")
df = pd.read_parquet(INPUT_DATA_PATH)

# === Load model ===
print("🔹 Loading CatBoost model...")
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# === Run prediction ===
print("🔹 Predicting...")
threshold = 0.8523727044216667
proba = model.predict_proba(df)[:, 1]
df["fraud_probability"] = proba
df["predicted_isFraud"] = (proba >= threshold).astype(int)

# === Save prediction ===
print("🔹 Saving results...")
df.to_parquet(OUTPUT_PATH, index=False)
print("✅ Prediction saved to:", OUTPUT_PATH)

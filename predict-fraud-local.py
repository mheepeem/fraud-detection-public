### SAGE MAKER NOTEBOOOK CODE ###

### !pip install catboost -v

import pandas as pd
from catboost import CatBoostClassifier

# === Configuration ===
BATCH_SIZE = 1000
THRESHOLD = 0.8420037372646131
MODEL_LOCAL_PATH = "code/catboost_fraud_model.cbm"

# === Load CatBoost Model ===
print("🔹 Loading CatBoost model...")
model = CatBoostClassifier()
model.load_model(MODEL_LOCAL_PATH)

# === Load Actual Data ===
print("🔹 Loading training actual data...")
actual_df = pd.read_parquet("s3://ml-dataflow-bucket/raw/train_transaction/train_transaction_for_vis.parquet")

# === Chunk & Upload Actual Data ===
chunk_count = 0
print("🔹 Uploading actual chunks...")
for start in range(0, len(actual_df), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(actual_df))
    chunk = actual_df.iloc[start:end].copy()
    s3_path = f"s3://ml-dataflow-bucket/predicted/chunks/chunk_{chunk_count:03d}.parquet"
    chunk.to_parquet(s3_path, index=False, storage_options={"anon": False})
    print(f"✅ Uploaded actual chunk {chunk_count} → {s3_path}")
    chunk_count += 1

# === Load Test Data ===
print("🔹 Loading test features and datetime...")
test_df = pd.read_parquet("s3://ml-dataflow-bucket/raw/test_transaction/test_transaction_for_prediction.parquet")
dt_df = pd.read_parquet("s3://ml-dataflow-bucket/raw/test_transaction/test_transaction_dt.parquet")
assert len(test_df) == len(dt_df), "❌ Test and datetime must match!"

# === Predict & Upload Test Chunks ===
print("🔹 Predicting and uploading future chunks...")
for start in range(0, len(test_df), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(test_df))
    chunk = test_df.iloc[start:end].copy()
    dt_chunk = dt_df.iloc[start:end].copy()

    # Predict
    proba = model.predict_proba(chunk)[:, 1]
    is_fraud = (proba >= THRESHOLD).astype(int)

    # Use "future" as the value label
    result = pd.DataFrame({
        "isFraud": is_fraud,
        "TransactionDT": dt_chunk["TransactionDT"].values,
        "value": "future"
    })

    s3_path = f"s3://ml-dataflow-bucket/predicted/chunks/chunk_{chunk_count:03d}.parquet"
    result.to_parquet(s3_path, index=False, storage_options={"anon": False})
    print(f"✅ Uploaded predicted chunk {chunk_count} → {s3_path}")
    chunk_count += 1

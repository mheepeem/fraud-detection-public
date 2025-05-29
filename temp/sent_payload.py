import json
import requests
import tracemalloc
import os
import pandas as pd
import base64

# === Config ===
file_configs = [
    {
        "name": "test_data",
        "path": "datas/test_transaction_cleaned.parquet",
        "send_as": "whole",        # "chunk" or "whole"
        "sample_frac": 0.001    # Only used if send_as is "chunk"
    },
]

api_url = "https://1xwq4ukc2m.execute-api.us-east-1.amazonaws.com/prod/ingest"
headers = {"Content-Type": "application/json"}
chunk_size = 400

# === Memory Tracker ===
tracemalloc.start()

def print_memory(prefix=""):
    current, peak = tracemalloc.get_traced_memory()
    print(f"📊 {prefix} Memory — Current: {current / 1024:.1f} KB | Peak: {peak / 1024:.1f} KB")

def send_parquet_file(dataset_name, dataframe, filename):
    dataframe.to_parquet(filename, index=False)
    with open(filename, "rb") as f:
        parquet_bytes = f.read()
        encoded_parquet = base64.b64encode(parquet_bytes).decode("utf-8")

    payload = {
        "dataset": dataset_name,
        "filename": filename,
        "data": encoded_parquet,
    }

    try:
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
        print(f"📤 Sent file '{filename}' | Status: {response.status_code}")
        print_memory(f"After sending {filename}")
    except Exception as e:
        print(f"❌ Error sending file '{filename}' from '{dataset_name}': {e}")

# === Process Each File ===
for config in file_configs:
    dataset_name = config["name"]
    file_path = config["path"]
    send_as = config.get("send_as", "chunk")
    sample_frac = config.get("sample_frac", None)

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue

    print(f"\n📄 Processing dataset: {file_path} ({dataset_name})")
    df = pd.read_parquet(file_path)
    print(f"✅ Loaded {len(df)} rows")

    if send_as == "whole":
        print_memory("Before sending whole file")
        filename = f"{dataset_name}.parquet"
        send_parquet_file(dataset_name, df, filename)

    elif send_as == "chunk":
        # Sample if sample_frac is defined
        if sample_frac is not None:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"🔎 Sampled {len(df)} rows with frac={sample_frac}")

        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            chunk_df = df.iloc[start:end]
            chunk_id = (start // chunk_size) + 1
            print_memory(f"Before sending chunk {chunk_id}")
            filename = f"{dataset_name}_chunk_{chunk_id}.parquet"
            send_parquet_file(dataset_name, chunk_df, filename)

    else:
        print(f"⚠️ Unknown 'send_as' mode: {send_as}")

# === Final Report ===
print_memory("Final")
tracemalloc.stop()

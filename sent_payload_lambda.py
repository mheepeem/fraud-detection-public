import requests

# === Config ===
lambda_url = "URL"  # Your Lambda URL

# === Files to Upload ===
files_to_upload = [
    {
        "file_path": "datas/test_transaction_for_prediction.parquet",
        "dataset_name": "test_transaction",
        "filename": "test_transaction_for_prediction.parquet"
    },
    {
        "file_path": "datas/test_transaction_dt.parquet",
        "dataset_name": "test_transaction",
        "filename": "test_transaction_dt.parquet"
    },
    {
        "file_path": "datas/train_transaction_for_vis.parquet",
        "dataset_name": "train_transaction",
        "filename": "train_transaction_for_vis.parquet"
    }
]

# === Upload Process ===
for file_info in files_to_upload:
    print(f"🚀 Uploading {file_info['filename']}...")

    # Step 1: Get pre-signed URL
    resp = requests.post(lambda_url, json={
        "dataset": file_info["dataset_name"],
        "filename": file_info["filename"]
    })
    result = resp.json()
    upload_url = result["upload_url"]
    print("✅ Got upload URL")

    # Step 2: Upload file
    with open(file_info["file_path"], "rb") as f:
        put_resp = requests.put(upload_url, data=f)
        print("📤 Upload status:", put_resp.status_code)

    if put_resp.ok:
        print(f"🎉 Upload complete to s3://{result['s3_key']}\n")
    else:
        print(f"❌ Upload failed: {put_resp.text}\n")

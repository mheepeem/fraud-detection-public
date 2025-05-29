import pandas as pd

# === Config ===
input_path = "datas/test_transaction_cleaned.parquet"
output_path = "datas/test_data_local.parquet"
fraction = 0.001  # Take first 0.1% of rows

# === Load full dataset
df = pd.read_parquet(input_path)
total_rows = len(df)
num_rows = int(total_rows * fraction)

# === Slice the first portion (non-random)
df_sampled = df.iloc[:num_rows]
print(f"📥 Taking first {num_rows} rows out of {total_rows}")

# === Save to file
df_sampled.to_parquet(output_path, index=False)
print(f"✅ Saved to {output_path}")

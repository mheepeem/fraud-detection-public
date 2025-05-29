import pandas as pd

# Load CSV
df = pd.read_csv("../datas/test_transaction.csv")

# Convert to Parquet
df.to_parquet("./datas/test_transaction.parquet", engine="pyarrow", index=False)

print("✅ Conversion complete!")

import pandas as pd
import numpy as np

# === Load both cleaned versions ===
df_v1 = pd.read_parquet('./datas/test_data_aws.parquet')
df_v2 = pd.read_parquet('./datas/test_data_local.parquet')

# === Shape comparison ===
print(f"📊 Shape v1: {df_v1.shape}")
print(f"📊 Shape v2: {df_v2.shape}")

# === Column comparison ===
v1_cols = set(df_v1.columns)
v2_cols = set(df_v2.columns)

only_in_v1 = sorted(v1_cols - v2_cols)
only_in_v2 = sorted(v2_cols - v1_cols)
shared = sorted(v1_cols & v2_cols)

print(f"\n✅ Shared columns: {len(shared)}")
print(f"➖ Only in v1: {len(only_in_v1)}")
print(f"➕ Only in v2: {len(only_in_v2)}")

# === Compare column dtypes ===
dtype_diff = {
    col: (df_v1[col].dtype, df_v2[col].dtype)
    for col in shared if df_v1[col].dtype != df_v2[col].dtype
}
print(f"\n🔍 Columns with different dtypes: {len(dtype_diff)}")
if dtype_diff:
    for col, (t1, t2) in list(dtype_diff.items())[:10]:
        print(f"  - {col}: v1={t1}, v2={t2}")

# === Sort both DataFrames to ensure row alignment ===
sort_cols = ["TransactionID"] if "TransactionID" in shared else shared[:5]
df_v1 = df_v1.sort_values(by=sort_cols).reset_index(drop=True)
df_v2 = df_v2.sort_values(by=sort_cols).reset_index(drop=True)

# === Deep equality check with tolerance for numeric columns ===
print("\n🔍 Checking data equality (NaN-aware + numeric tolerance)...")
mismatch_columns = []
mismatch_counts = {}

for col in shared:
    s1 = df_v1[col]
    s2 = df_v2[col]

    if np.issubdtype(s1.dtype, np.number):
        is_equal = np.isclose(s1.fillna(0), s2.fillna(0), atol=1e-6, equal_nan=True)
    else:
        is_equal = s1.fillna("___NULL___") == s2.fillna("___NULL___")

    if not np.all(is_equal):
        mismatch_columns.append(col)
        mismatch_counts[col] = (~is_equal).sum()

# === Summary ===
if not mismatch_columns:
    print("✅ DataFrames are identical (with numeric tolerance and NaN handling).")
else:
    print(f"❌ Mismatched columns: {len(mismatch_columns)}")
    top_diffs = sorted(mismatch_counts.items(), key=lambda x: x[1], reverse=True)

    print("\n🔝 Top 10 mismatched columns:")
    for col, count in top_diffs[:10]:
        print(f"  - {col}: {count} mismatches")

    mismatch_rows = df_v1[shared].ne(df_v2[shared]).any(axis=1)
    print(f"\n🔢 Total rows with at least one mismatch: {mismatch_rows.sum()}")
    print(f"🧪 First few mismatched row indices: {mismatch_rows[mismatch_rows].index.tolist()[:5]}")

    # === Show sample mismatches
    print("\n🔬 Sample mismatched values for top columns:")

    max_cols_to_show = 5
    max_rows_to_show = 5

    for col, _ in top_diffs[:max_cols_to_show]:
        s1 = df_v1[col]
        s2 = df_v2[col]

        if np.issubdtype(s1.dtype, np.number):
            mismatch_mask = ~np.isclose(s1.fillna(0), s2.fillna(0), atol=1e-6, equal_nan=True)
        else:
            mismatch_mask = (s1.fillna("___NULL___") != s2.fillna("___NULL___"))

        sample = pd.DataFrame({
            "index": s1[mismatch_mask].index,
            "v1": s1[mismatch_mask].values,
            "v2": s2[mismatch_mask].values
        }).head(max_rows_to_show)

        print(f"\n📌 Column: {col}")
        print(sample.to_string(index=False))

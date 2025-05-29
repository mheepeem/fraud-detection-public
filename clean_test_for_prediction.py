import pandas as pd

# === Load raw data ===
df = pd.read_csv('./datas/test_transaction.csv')

# === Load final feature list ===
with open('./datas/final_features.txt', 'r') as f:
    final_features = [line.strip() for line in f.readlines()]

# === Step 1: Drop columns with ≥ 90% missing
missing_percent = df.isnull().mean() * 100
df = df.drop(columns=missing_percent[missing_percent >= 90].index)

# === Step 2: Create `_missing` flags for 60–90% missing columns
mask = (missing_percent >= 60) & (missing_percent < 90)
cols_to_flag = missing_percent[mask].index.tolist()

missing_flags = {
    f"{col}_missing": df[col].isnull().astype(int)
    for col in cols_to_flag
}
df = pd.concat([df, pd.DataFrame(missing_flags)], axis=1)

# === Step 3: Impute missing values safely
num_cols = df.select_dtypes(include='number').columns
obj_cols = df.select_dtypes(include='object').columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[obj_cols] = df[obj_cols].fillna("Unknown")

# === Save TransactionDT separately before dropping
transaction_dt = df[['TransactionDT']]
transaction_dt.to_parquet('./datas/test_transaction_dt.parquet', index=False)
print("💾 Saved TransactionDT separately.")


# === Step 4: Drop unnecessary columns
drop_cols = ['TransactionID', 'TransactionDT','P_emaildomain', 'R_emaildomain', 'addr1', 'addr2', 'isFraud']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# === Step 5: One-hot encode categorical columns
categorical_cols = [
    'ProductCD', 'card2', 'card3', 'card4', 'card5', 'card6',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]
cols_to_encode = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

# === Step 6: Align with final features
current_cols = set(df.columns)
final_cols = set(final_features)

missing_cols = list(final_cols - current_cols)
extra_cols = list(current_cols - final_cols)

# Fix fragmentation: add all missing at once
df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)

# Drop extras
df = df.drop(columns=extra_cols)

# Reorder
df = df[final_features]

# === Summary
print(f"✅ Final cleaned data shape: {df.shape}")
print(f"➕ Added {len(missing_cols)} missing columns")
print(f"➖ Dropped {len(extra_cols)} unexpected columns")

# === Step 7: Save
df.to_parquet('./datas/test_transaction_for_prediction.parquet', index=False)
print("💾 Saved cleaned and aligned data to Parquet.")

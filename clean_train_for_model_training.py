import pandas as pd

# Read data
df = pd.read_csv('./datas/train_transaction.csv')

# Step 0: Calculate missing summary
missing_counts = df.isnull().sum()
missing_percent = df.isnull().mean() * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_percent
}).sort_values(by='Missing %', ascending=False)

# Step 1: Drop columns with >= 90% missing
missing_threshold = 90
cols_to_drop = missing_summary[missing_summary['Missing %'] >= missing_threshold].index
df_cleaned = df.drop(cols_to_drop, axis=1)
print(f"Dropped {len(cols_to_drop)} columns with ≥ {missing_threshold}% missing data.")

# Step 2: Create missing-indicator flags for 60–90% missing columns
mask = (missing_summary['Missing %'] >= 60) & (missing_summary['Missing %'] < 90)
cols_to_flag = missing_summary[mask].index.tolist()

missing_flags = {
    f"{col}_missing": df_cleaned[col].isnull().astype(int)
    for col in cols_to_flag
}

df_cleaned = pd.concat([df_cleaned, pd.DataFrame(missing_flags)], axis=1)
print(f"Created {len(cols_to_flag)} '_missing' indicator features (optimized).")

# Step 3: Investigate correlation with isFraud
correlations = {}
for col in cols_to_flag:
    flag_col = f"{col}_missing"
    if 'isFraud' in df_cleaned.columns:
        corr = df_cleaned[flag_col].corr(df_cleaned['isFraud'])
        correlations[flag_col] = corr

# Create sorted dataframe of correlations
correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with isFraud'])
correlation_df = correlation_df.sort_values(by='Correlation with isFraud', ascending=False)

print("\nTop 10 missing-indicator correlations with isFraud:")
print(correlation_df.head(10))

print("\nBottom 10 (negative correlation):")
print(correlation_df.tail(10))

# Step 4: Keep only _missing flags with correlation < -0.15
important_flags = correlation_df[correlation_df['Correlation with isFraud'] < -0.15].index.tolist()
keep_originals = [col.replace('_missing', '') for col in important_flags]

# Drop 60–90% missing columns that aren't in important list
drop_candidates = list(set(cols_to_flag) - set(keep_originals))
drop_flags = [f"{col}_missing" for col in drop_candidates]

df_cleaned = df_cleaned.drop(columns=drop_candidates + drop_flags)

print(f"\nDropped {len(drop_candidates)} original columns and {len(drop_flags)} weak _missing indicators.")
print(f"Kept {len(important_flags)} strong _missing indicators.")
print(f"\nRemaining columns: {df_cleaned.shape[1]}")

# Step 5: Impute remaining missing values (<60%)
num_cols = df_cleaned.select_dtypes(include='number').columns
obj_cols = df_cleaned.select_dtypes(include='object').columns

# Impute numeric with median
df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].median())

# Impute object (categorical) with 'Unknown'
df_cleaned[obj_cols] = df_cleaned[obj_cols].fillna("Unknown")

# Confirm all missing values are now handled
total_missing = df_cleaned.isnull().sum().sum()
print(f"\nRemaining total missing values: {total_missing}")

# Step 6: Drop unnecessary columns
drop_cols = ['TransactionID', 'TransactionDT','P_emaildomain', 'R_emaildomain', 'addr1', 'addr2']
df_cleaned = df_cleaned.drop(columns=[col for col in drop_cols if col in df_cleaned.columns])
print(f"Dropped columns: {drop_cols}")

# Step 7: One-Hot Encode selected categorical features
categorical_cols = [
    'ProductCD', 'card2', 'card3', 'card4', 'card5', 'card6',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]
cols_to_encode = [col for col in categorical_cols if col in df_cleaned.columns]

df_cleaned = pd.get_dummies(df_cleaned, columns=cols_to_encode, drop_first=True)

print(f"One-hot encoded columns: {cols_to_encode}")
print(f"Data shape after encoding: {df_cleaned.shape}")

# Step 8: Save final column list
final_features = df_cleaned.columns.tolist()
with open("./datas/final_features.txt", "w") as f:
    f.write("\n".join(final_features))
print(f"Saved final feature list with {len(final_features)} columns.")

# Step 9: Save cleaned data to Parquet
df_cleaned.to_parquet('./datas/train_transaction_for_model_training.parquet', engine="pyarrow", index=False)
print("Saved cleaned dataset to Parquet.")

import pandas as pd

# Step 1: Load the CSV file
df = pd.read_csv('datas/train_transaction.csv')

# Step 2: Select only the required columns and make a copy
selected_df = df[['isFraud', 'TransactionDT']].copy()

# Step 3: Add a new column with the string 'current'
selected_df['value'] = 'current'

# Step 4: Save to Parquet
selected_df.to_parquet('datas/train_transaction_for_vis.parquet', index=False)

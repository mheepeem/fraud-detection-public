import pandas as pd
import math

def split_dataframe_into_chunks(df: pd.DataFrame, num_chunks: int) -> list[pd.DataFrame]:
    """
    Splits a DataFrame into `num_chunks` approximately equal parts.

    Args:
        df (pd.DataFrame): The input DataFrame.
        num_chunks (int): The number of chunks to split into.

    Returns:
        List[pd.DataFrame]: A list of DataFrame chunks.
    """
    total_rows = len(df)
    chunk_size = math.ceil(total_rows / num_chunks)
    chunks = [
        df.iloc[i * chunk_size : (i + 1) * chunk_size].reset_index(drop=True)
        for i in range(num_chunks)
    ]
    return chunks

df = pd.read_parquet("datas/test_data_local.parquet")
chunks = split_dataframe_into_chunks(df, 5)  # split into 5 parts

# Save each chunk
for i, chunk in enumerate(chunks, 1):
    chunk.to_parquet(f"datas/test_data_local_{i}.parquet", index=False)
    print(f"✅ Saved output_chunk_{i}.parquet with {len(chunk)} rows")


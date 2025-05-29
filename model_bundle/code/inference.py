import os
import pandas as pd
from catboost import CatBoostClassifier

THRESHOLD = 0.8420037372646131

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "catboost_fraud_model.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def input_fn(input_data, content_type):
    if content_type == "application/x-parquet":
        return pd.read_parquet(input_data)
    else:
        raise Exception(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    proba = model.predict_proba(input_data)[:, 1]
    input_data["fraud_probability"] = proba
    input_data["predicted_isFraud"] = (proba >= THRESHOLD).astype(int)
    return input_data

def output_fn(predictions, accept):
    if accept == "application/x-parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(predictions)
        buffer = pa.BufferOutputStream()
        pq.write_table(table, buffer)
        return buffer.getvalue().to_pybytes(), "application/x-parquet"
    else:
        raise Exception(f"Unsupported accept type: {accept}")

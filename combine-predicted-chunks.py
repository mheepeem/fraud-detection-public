### GLUE CODE ###

import sys
import boto3
from pyspark.context import SparkContext
from awsglue.context import GlueContext

# Initialize
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
s3 = boto3.client("s3")

# Define paths
bucket = "ml-dataflow-bucket"
input_path = "s3://ml-dataflow-bucket/predicted/chunks/"
temp_output_prefix = "predicted/full/temp_output/"
final_key = "predicted/full/predicted_fraud.csv"

# 1. Read all chunked predictions
df = spark.read.parquet(input_path)

# 2. Coalesce to a single file and write as CSV
df.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"s3://{bucket}/{temp_output_prefix}")

# 3. Find the single generated part file and rename it
response = s3.list_objects_v2(Bucket=bucket, Prefix=temp_output_prefix)

for obj in response.get("Contents", []):
    key = obj["Key"]
    if key.endswith(".csv"):
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": key},
            Key=final_key
        )
        print(f"✅ Copied as: s3://{bucket}/{final_key}")
        s3.delete_object(Bucket=bucket, Key=key)
        break

# 4. Clean up remaining part files and metadata (_SUCCESS, etc.)
for obj in response.get("Contents", []):
    if not obj["Key"].endswith("/"):
        s3.delete_object(Bucket=bucket, Key=obj["Key"])

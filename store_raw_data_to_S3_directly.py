### LAMBDA CODE ###

import json
import boto3
import os
from datetime import datetime

s3 = boto3.client("s3")
BUCKET_NAME = "BUCKET"  # ← update this

def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        dataset = body.get("dataset", "default")
        filename = body.get("filename", f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")

        s3_key = f"raw/{dataset}/{filename}"

        presigned_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": BUCKET_NAME, "Key": s3_key},
            ExpiresIn=900  # URL valid for 15 minutes
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "upload_url": presigned_url,
                "s3_key": s3_key
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

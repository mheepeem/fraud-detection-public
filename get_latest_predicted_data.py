### LAMBDA CODE ###

import boto3
import json

# === Config ===
BUCKET = "ml-dataflow-bucket"
KEY = "predicted/full/predicted_fraud.csv"

def lambda_handler(event, context):
    print("🎯 Generating pre-signed S3 URL")

    s3 = boto3.client("s3")

    try:
        # Generate a 1-hour valid URL
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': BUCKET, 'Key': KEY},
            ExpiresIn=3600  # 1 hour
        )

        print("✅ URL generated")
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"url": url})
        }

    except Exception as e:
        print(f"❌ Lambda Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": f"Lambda Error: {str(e)}"
        }

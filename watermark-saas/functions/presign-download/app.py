"""
POST /jobs/{jobId}/download Lambda handler.

Generates a presigned GET URL for the processed result file.
"""

import json
import os
import logging

import boto3
from botocore.config import Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
dynamodb = boto3.resource("dynamodb")

BUCKET = os.environ["BUCKET_NAME"]
TABLE_NAME = os.environ["TABLE_NAME"]
table = dynamodb.Table(TABLE_NAME)

DOWNLOAD_URL_EXPIRY = 3600  # 1 hour


def handler(event, context):
    try:
        job_id = event.get("pathParameters", {}).get("jobId")
        if not job_id:
            return response(400, {"error": "jobId is required"})

        # Read job record
        result = table.get_item(
            Key={"pk": f"JOB#{job_id}", "sk": "META"}
        )
        item = result.get("Item")

        if not item:
            return response(404, {"error": "Job not found"})

        if item.get("status") != "completed":
            return response(400, {
                "error": f"Job is not completed. Current status: {item.get('status')}"
            })

        result_key = item.get("resultKey")
        if not result_key:
            return response(500, {"error": "Result file not found"})

        # Generate presigned download URL
        download_url = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET,
                "Key": result_key,
            },
            ExpiresIn=DOWNLOAD_URL_EXPIRY,
        )

        return response(200, {
            "downloadUrl": download_url,
            "expiresIn": DOWNLOAD_URL_EXPIRY,
            "filename": item.get("filename"),
        })

    except Exception as e:
        logger.error(f"Download handler error: {e}")
        return response(500, {"error": "Internal server error"})


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body),
    }

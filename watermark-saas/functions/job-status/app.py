"""
GET /jobs/{jobId}/status Lambda handler.

Returns job status and preview URL if completed.
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

PREVIEW_URL_EXPIRY = 3600  # 1 hour


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

        status = item.get("status", "unknown")
        resp_body = {
            "jobId": job_id,
            "status": status,
            "filename": item.get("filename"),
            "createdAt": item.get("createdAt"),
        }

        # If completed, generate presigned URL for preview
        if status == "completed" and item.get("previewKey"):
            preview_url = s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": BUCKET,
                    "Key": item["previewKey"],
                },
                ExpiresIn=PREVIEW_URL_EXPIRY,
            )
            resp_body["previewUrl"] = preview_url

        if status == "failed":
            resp_body["error"] = item.get("errorMessage", "Unknown error")

        return response(200, resp_body)

    except Exception as e:
        logger.error(f"Status handler error: {e}")
        return response(500, {"error": "Internal server error"})


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body),
    }

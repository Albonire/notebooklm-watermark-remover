"""
POST /upload Lambda handler.

Flow:
1. Extract client IP, hash with SHA-256 for anonymous quota tracking
2. Check DynamoDB daily usage count for ANON#{ipHash}
3. If usage >= 3, return 429
4. Generate UUID jobId
5. Create S3 presigned PUT URL for uploads/{jobId}/{filename}
6. Create DynamoDB job record (status=pending)
7. Increment daily counter
8. Send SQS message for processing trigger (after upload completes - actually,
   we use S3 event notification in template.yaml, but we prepare the job record)
9. Return { jobId, uploadUrl, expiresIn }
"""

import json
import os
import uuid
import hashlib
import time
import logging
from datetime import datetime, timezone

import boto3
from botocore.config import Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
dynamodb = boto3.resource("dynamodb")
sqs = boto3.client("sqs")

BUCKET = os.environ["BUCKET_NAME"]
TABLE_NAME = os.environ["TABLE_NAME"]
QUEUE_URL = os.environ["QUEUE_URL"]
table = dynamodb.Table(TABLE_NAME)

MAX_ANONYMOUS_DAILY = 3
PRESIGN_EXPIRY = 900  # 15 minutes
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}


def get_daily_key():
    """Return today's date string in UTC for daily quota tracking."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_end_of_day_ttl():
    """Return Unix timestamp for end of current UTC day."""
    now = datetime.now(timezone.utc)
    end_of_day = now.replace(hour=23, minute=59, second=59)
    return int(end_of_day.timestamp())


def handler(event, context):
    try:
        # Parse request
        body = json.loads(event.get("body", "{}"))
        filename = body.get("filename", "")
        content_type = body.get("contentType", "")

        if not filename:
            return response(400, {"error": "filename is required"})

        if content_type not in ALLOWED_CONTENT_TYPES:
            return response(400, {
                "error": f"Unsupported content type: {content_type}. Allowed: PNG, JPG, WEBP"
            })

        # Extract client IP and hash it
        source_ip = event.get("requestContext", {}).get("http", {}).get("sourceIp", "unknown")
        ip_hash = hashlib.sha256(source_ip.encode()).hexdigest()[:16]

        # Check anonymous daily quota
        daily_key = get_daily_key()
        quota_pk = f"ANON#{ip_hash}"
        quota_sk = f"DAILY#{daily_key}"

        quota_item = table.get_item(
            Key={"pk": quota_pk, "sk": quota_sk}
        ).get("Item")

        current_usage = int(quota_item["count"]) if quota_item else 0

        if current_usage >= MAX_ANONYMOUS_DAILY:
            return response(429, {
                "error": "Daily free limit reached (3/3). Please try again tomorrow.",
                "usage": current_usage,
                "limit": MAX_ANONYMOUS_DAILY,
            })

        # Generate job ID
        job_id = str(uuid.uuid4())
        s3_key = f"uploads/{job_id}/{filename}"

        # Create presigned PUT URL
        upload_url = s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": BUCKET,
                "Key": s3_key,
                "ContentType": content_type,
            },
            ExpiresIn=PRESIGN_EXPIRY,
        )

        # Create job record in DynamoDB
        now = int(time.time())
        ttl = now + 86400  # 24 hours

        table.put_item(Item={
            "pk": f"JOB#{job_id}",
            "sk": "META",
            "status": "pending",
            "ipHash": ip_hash,
            "filename": filename,
            "s3Key": s3_key,
            "contentType": content_type,
            "createdAt": now,
            "ttl": ttl,
        })

        # Increment daily counter
        table.update_item(
            Key={"pk": quota_pk, "sk": quota_sk},
            UpdateExpression="SET #c = if_not_exists(#c, :zero) + :one, #t = :ttl",
            ExpressionAttributeNames={"#c": "count", "#t": "ttl"},
            ExpressionAttributeValues={
                ":zero": 0,
                ":one": 1,
                ":ttl": get_end_of_day_ttl(),
            },
        )

        # Send SQS message to trigger processing
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps({
                "jobId": job_id,
                "s3Key": s3_key,
                "filename": filename,
            }),
        )

        remaining = MAX_ANONYMOUS_DAILY - current_usage - 1

        return response(200, {
            "jobId": job_id,
            "uploadUrl": upload_url,
            "expiresIn": PRESIGN_EXPIRY,
            "remainingCredits": remaining,
        })

    except Exception as e:
        logger.error(f"Upload handler error: {e}")
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

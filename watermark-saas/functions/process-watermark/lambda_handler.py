"""
SQS-triggered Lambda: processes uploaded images to remove watermarks.

Flow:
1. Parse S3 key from SQS message body
2. Download image from S3 uploads/ prefix
3. Run watermark removal via WatermarkRemover.process_image()
4. Upload result to S3 results/ prefix
5. Generate low-res preview (max 400px, JPEG q=60) to S3 previews/ prefix
6. Update DynamoDB job record with status=completed
7. On error: update job status=failed
"""

import json
import os
import logging
import tempfile
import uuid

import boto3
import cv2

from watermark_core import WatermarkRemover, WatermarkConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

BUCKET = os.environ["BUCKET_NAME"]
TABLE_NAME = os.environ["TABLE_NAME"]
table = dynamodb.Table(TABLE_NAME)


def generate_preview(input_path: str, preview_path: str, max_dim: int = 400, quality: int = 60):
    """Generate a low-res JPEG preview of the processed image."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot read image for preview: {input_path}")

    h, w = img.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(preview_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def update_job_status(job_id: str, status: str, **extra_attrs):
    """Update the DynamoDB job record."""
    update_expr = "SET #s = :status"
    expr_names = {"#s": "status"}
    expr_values = {":status": status}

    for key, value in extra_attrs.items():
        update_expr += f", {key} = :{key}"
        expr_values[f":{key}"] = value

    table.update_item(
        Key={"pk": f"JOB#{job_id}", "sk": "META"},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
    )


def handler(event, context):
    """SQS event handler - processes each message (one image per message)."""
    for record in event["Records"]:
        body = json.loads(record["body"])
        job_id = body["jobId"]
        s3_key = body["s3Key"]
        filename = body.get("filename", "image.png")

        logger.info(f"Processing job {job_id}, key={s3_key}")

        # Determine file extension
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"

        try:
            update_job_status(job_id, "processing")

            # Download from S3
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
                s3.download_file(BUCKET, s3_key, tmp_in.name)
                input_path = tmp_in.name

            # Process watermark removal
            output_path = input_path.replace(ext, f"_cleaned{ext}")
            remover = WatermarkRemover(WatermarkConfig())
            success = remover.process_image(input_path, output_path)

            if not success:
                raise RuntimeError("Watermark removal returned False")

            # Upload result
            result_key = f"results/{job_id}/{filename}"
            s3.upload_file(output_path, BUCKET, result_key)

            # Generate and upload preview
            preview_path = input_path.replace(ext, "_preview.jpg")
            generate_preview(output_path, preview_path)
            preview_key = f"previews/{job_id}/preview.jpg"
            s3.upload_file(
                preview_path,
                BUCKET,
                preview_key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )

            # Update job record
            update_job_status(
                job_id,
                "completed",
                resultKey=result_key,
                previewKey=preview_key,
            )

            logger.info(f"Job {job_id} completed successfully")

            # Cleanup temp files
            for path in [input_path, output_path, preview_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            update_job_status(job_id, "failed", errorMessage=str(e))
            raise

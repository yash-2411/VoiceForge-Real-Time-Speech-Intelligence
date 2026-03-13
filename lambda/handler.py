import base64
import json
import os
import uuid
from typing import Any

import boto3
from botocore.exceptions import ClientError

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}


def _response(body: dict, status_code: int = 200) -> dict:
    return {
        "statusCode": status_code,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _upload_action(event: dict) -> dict:
    bucket = os.environ["S3_BUCKET"]
    region = os.environ.get("AWS_REGION", "us-east-1")
    job_id = str(uuid.uuid4())
    key = f"voiceforge/audio/{job_id}.wav"

    s3 = boto3.client("s3", region_name=region)
    upload_url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,
    )
    return _response({"upload_url": upload_url, "job_id": job_id})


def _transcribe_action(event: dict) -> dict:
    bucket = os.environ["S3_BUCKET"]
    region = os.environ.get("AWS_REGION", "us-east-1")
    endpoint_name = os.environ["SAGEMAKER_ENDPOINT_NAME"]
    job_id = event.get("job_id")
    if not job_id:
        return _response({"error": "Missing job_id"}, 400)

    s3 = boto3.client("s3", region_name=region)
    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

    audio_key = f"voiceforge/audio/{job_id}.wav"
    result_key = f"voiceforge/results/{job_id}.json"

    try:
        obj = s3.get_object(Bucket=bucket, Key=audio_key)
        audio_bytes = obj["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return _response({"error": f"Audio not found for job_id: {job_id}"}, 404)
        return _response({"error": str(e)}, 500)
    except Exception as e:
        return _response({"error": str(e)}, 500)

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    payload = json.dumps({"audio_b64": audio_b64})

    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        result_body = response["Body"].read().decode("utf-8")
        transcript = json.loads(result_body)
        if isinstance(transcript, dict) and "error" in transcript:
            return _response(transcript, 500)
    except Exception as e:
        return _response({"error": str(e)}, 500)

    try:
        s3.put_object(
            Bucket=bucket,
            Key=result_key,
            Body=json.dumps(transcript),
            ContentType="application/json",
        )
    except Exception:
        pass

    return _response({"job_id": job_id, "transcript": transcript})


def _get_result_action(event: dict) -> dict:
    bucket = os.environ["S3_BUCKET"]
    region = os.environ.get("AWS_REGION", "us-east-1")
    job_id = event.get("job_id")
    if not job_id:
        return _response({"error": "Missing job_id"}, 400)

    s3 = boto3.client("s3", region_name=region)
    result_key = f"voiceforge/results/{job_id}.json"

    try:
        obj = s3.get_object(Bucket=bucket, Key=result_key)
        result = json.loads(obj["Body"].read().decode("utf-8"))
        return _response({"job_id": job_id, "status": "ready", "transcript": result})
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return _response({"job_id": job_id, "status": "processing"})
        return _response({"error": str(e)}, 500)
    except Exception as e:
        return _response({"error": str(e)}, 500)


def handler(event: dict, context: Any) -> dict:
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return _response({}, 204)

    body = event.get("body", "{}")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = {}
    action = body.get("action") or event.get("action")

    if action == "upload":
        return _upload_action(event)
    if action == "transcribe":
        merged = {**body, **event}
        return _transcribe_action(merged)
    if action == "get_result":
        merged = {**body, **event}
        return _get_result_action(merged)

    return _response({"error": "Invalid action. Use: upload, transcribe, get_result"}, 400)

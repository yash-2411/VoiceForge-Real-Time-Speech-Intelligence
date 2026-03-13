# VoiceForge — Real-Time Speech Intelligence Platform

Production-ready Python project for speaker-attributed transcription using AWS SageMaker LMI (Whisper + pyannote), Lambda, and S3.

---

## Architecture Diagram (AWS Components)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VoiceForge Architecture                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐      ┌─────────────┐      ┌─────────┐      ┌──────────────────────┐
    │  Client  │ ───► │   Lambda    │ ───► │   S3   │ ───► │  SageMaker LMI       │
    │ (Dashboard)     │  (API)      │      │ (Audio) │      │  Whisper + Pyannote  │
    └──────────┘      └─────────────┘      └─────────┘      └──────────────────────┘
           ▲                   │                 │                    │
           │                   │                 │                    │
           │                   ▼                 ▼                    ▼
           │            ┌─────────────┐   ┌─────────────┐      ┌─────────────┐
           └────────────│  Transcript │   │  Results    │      │  ml.g4dn    │
                        │  (JSON)     │   │  (S3)       │      │  (T4 GPU)   │
                        └─────────────┘   └─────────────┘      └─────────────┘

  AWS Components:
  • Lambda     = Public API (upload URL, transcribe, get_result)
  • S3         = Audio staging + result storage
  • SageMaker  = LMI container with Whisper + pyannote (scale-to-zero)
```

---

## Code Flow

```
  User                    Lambda                    S3                 SageMaker
   │                        │                        │                      │
   │  action=upload         │                        │                      │
   │──────────────────────►│                        │                      │
   │                        │  presigned PUT URL     │                      │
   │◄──────────────────────│  + job_id              │                      │
   │                        │                        │                      │
   │  PUT audio (presigned)  │                        │                      │
   │───────────────────────────────────────────────►│                      │
   │                        │                        │                      │
   │  action=transcribe     │                        │                      │
   │  + job_id              │                        │                      │
   │──────────────────────►│  GET audio             │                      │
   │                        │───────────────────────►│                      │
   │                        │  audio bytes           │                      │
   │                        │◄───────────────────────│                      │
   │                        │  invoke_endpoint       │                      │
   │                        │─────────────────────────────────────────────►│
   │                        │  transcript (JSON)     │                      │
   │                        │◄─────────────────────────────────────────────│
   │                        │  PUT result            │                      │
   │                        │───────────────────────►│                      │
   │  transcript             │                        │                      │
   │◄──────────────────────│                        │                      │
```

---

## Project Structure

```
voiceforge/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── sagemaker_artifacts/
│   ├── serving.properties
│   ├── requirements.txt
│   └── model.py
├── lambda/
│   ├── handler.py
│   └── lambda_deployment.json
├── api/
│   └── app.py
├── dashboard/
│   └── app.py
├── utils/
│   ├── s3_utils.py
│   └── sagemaker_utils.py
├── 1_package_and_upload.py
├── 2_deploy_endpoint.py
├── 3_deploy_lambda.py
├── 4_setup_autoscaling.py
├── 5_test_endpoint.py
└── 6_delete_endpoint.py
```

---

## Prerequisites

- AWS account with CLI configured
- HuggingFace token (for pyannote.audio) — [accept pyannote terms](https://huggingface.co/pyannote/speaker-diarization-3.1)
- IAM roles: `SageMakerExecutionRole`, `LambdaVoiceForgeRole`

---

## Quickstart

1. Copy `.env.example` to `.env` and fill in values.
2. Create S3 bucket and IAM roles (see IAM section below).
3. Run in order:

```bash
python 1_package_and_upload.py
python 2_deploy_endpoint.py
python 3_deploy_lambda.py
python 4_setup_autoscaling.py
python 5_test_endpoint.py
```

---

## Cost Estimate

| Resource              | Price         | Notes                          |
|-----------------------|---------------|--------------------------------|
| ml.g4dn.xlarge        | $0.736/hr     | T4 GPU, scale-to-0 when idle   |
| Lambda (300s, 512MB)  | ~$0.001/invoke| Scale-to-zero                  |
| S3                    | ~$0.023/GB    | Audio + results                |
| **Idle cost**         | **$0.00**     | Scale-to-0 eliminates baseline |
| **Active (1 req/hr)** | **~$0.05/hr** | Endpoint warmup + Lambda       |

---

## IAM Roles

### SageMakerExecutionRole
- AmazonSageMakerFullAccess
- AmazonS3FullAccess (or scoped to bucket)
- ECR: GetAuthorizationToken, BatchGetImage, GetDownloadUrlForLayer

### LambdaVoiceForgeRole
- AWSLambdaBasicExecutionRole
- sagemaker:InvokeEndpoint, s3:GetObject, s3:PutObject (scoped to voiceforge/*)

---

## Teardown

```bash
python 6_delete_endpoint.py
```

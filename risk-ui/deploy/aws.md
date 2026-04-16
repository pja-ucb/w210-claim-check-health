# AWS Deployment Notes (MVP)

## Minimal AWS Setup

1. **Backend API**
   - Package FastAPI into a container.
   - Deploy to ECS Fargate behind an ALB.
   - Environment variables for:
     - `SAGEMAKER_ENDPOINT`
     - `RAG_ENDPOINT`
     - data source configs (Athena/RDS/S3)

2. **Model**
   - Deploy model to SageMaker real-time endpoint.
   - Backend calls `InvokeEndpoint`.

3. **RAG**
   - Host retrieval service on ECS or SageMaker.
   - Backend calls the endpoint for evidence.

4. **Frontend**
   - Upload static files to S3.
   - Serve via CloudFront.
   - Point API base URL to the ALB endpoint.

## IAM/Permissions

- Backend task role should allow:
  - `sagemaker:InvokeEndpoint`
  - `s3:GetObject` or `athena:StartQueryExecution`
  - `rds-db:connect` if using RDS

## Local Validation

- Start backend on `http://localhost:8000`
- Open `frontend/index.html` in browser
- Upload a CSV and verify scoring

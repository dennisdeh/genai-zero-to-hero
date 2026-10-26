#!/usr/bin/env bash
# Manual deployment of the iris API to Google Cloud Run (see book_cloud_deployment.md).
#
# Prerequisites: the gcloud CLI, Docker, a Google Cloud project with billing enabled, and
#   gcloud auth login && gcloud config set project <PROJECT_ID>
#
# Run from the repository root:
#   GCP_PROJECT_ID=my-project GCP_REGION=europe-west6 \
#     bash p01_containerisation_and_cloud_deployment/c02_cloud_deployment/deploy_cloud_run.sh
#
# Clean up afterwards (to avoid costs):
#   gcloud run services delete iris-api --region "$GCP_REGION"
#   gcloud artifacts repositories delete gzh --location "$GCP_REGION"
set -euo pipefail

# 1. Configuration (override with environment variables)
PROJECT_ID="${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-europe-west6}"
REPO="${GCP_ARTIFACT_REPO:-gzh}"
SERVICE="${SERVICE_NAME:-iris-api}"
TAG="${IMAGE_TAG:-$(git rev-parse --short=12 HEAD)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/gzh-iris-api:${TAG}"

# 2. Enable the required APIs and create the Docker repository in Artifact Registry (once)
gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project "$PROJECT_ID"
gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --repository-format=docker \
       --location "$REGION" --project "$PROJECT_ID" --description "genai-zero-to-hero images"

# 3. Let docker authenticate to the registry, then build for linux/amd64 (needed on Apple Silicon) and push
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker build --platform linux/amd64 \
  -f p01_containerisation_and_cloud_deployment/c00_docker/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"

# 4. Deploy (creates the service or a new revision of it)
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --port 5000 \
  --cpu 1 \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 40 \
  --set-env-vars LOG_LEVEL=INFO,MAX_BATCH_SIZE=1000 \
  --allow-unauthenticated

# 5. Smoke test
URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT_ID" --format 'value(status.url)')"
echo "Service URL: $URL"
curl -fsS "$URL/ready" && echo
echo "Call it with: API_URL=$URL python -m p00_fundamentals.c02_fastapi.main_client"

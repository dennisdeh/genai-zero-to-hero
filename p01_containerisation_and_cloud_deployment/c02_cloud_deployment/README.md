## Environment
- A cloud account with billing enabled (the example uses Google Cloud; Azure and AWS equivalents are described
  in `book_cloud_deployment.md`). Remember to delete the resources afterwards to avoid costs.
- CLI tools: Docker, the `gcloud` CLI (authenticated with `gcloud auth login`) and git.
- Python conda environment for the client: `p00_fundamentals/c02_fastapi/conda_env_gzh_api0.yml`
- Docker compose setup: none (see `../c00_docker` for local runs).
- Environment variables of `deploy_cloud_run.sh`:
  - `GCP_PROJECT_ID` (required): the Google Cloud project
  - `GCP_REGION` (default: `europe-west6`), `GCP_ARTIFACT_REPO` (default: `gzh`),
    `SERVICE_NAME` (default: `iris-api`), `IMAGE_TAG` (default: current git commit SHA)
- No credentials are stored in this repository; the GitHub Actions example uses Workload Identity Federation
  and repository variables (see the header of `github_actions_ci_cd_example.yml` and the one-off set-up below).

## How to run
Manual deployment to Google Cloud Run (from the repository root):
```bash
GCP_PROJECT_ID=my-project GCP_REGION=europe-west6 \
  bash p01_containerisation_and_cloud_deployment/c02_cloud_deployment/deploy_cloud_run.sh
API_URL=<printed service URL> python -m p00_fundamentals.c02_fastapi.main_client
```

Clean up:
```bash
gcloud run services delete iris-api --region europe-west6
gcloud artifacts repositories delete gzh --location europe-west6
```

### One-off set-up for the CI/CD workflow
The workflow authenticates with Workload Identity Federation (WIF), which must first be set up in the Google
Cloud project. Replace the placeholders (`my-project`, `OWNER/REPO`) and run (requires the project owner or
equivalent IAM permissions, and the Artifact Registry repository `gzh`, e.g. created by `deploy_cloud_run.sh`):
```bash
PROJECT_ID=my-project
REGION=europe-west6
GITHUB_REPO=OWNER/REPO                     # the GitHub repository that may deploy, e.g. octo-org/octo-repo
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
DEPLOY_SA=gh-deployer@${PROJECT_ID}.iam.gserviceaccount.com
RUNTIME_SA=iris-api-runtime@${PROJECT_ID}.iam.gserviceaccount.com

# 1. APIs used by WIF, a deployer service account (used by GitHub) and a runtime service account (used by Cloud Run)
gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project="$PROJECT_ID"
gcloud iam service-accounts create gh-deployer --project="$PROJECT_ID" --display-name="GitHub Actions deployer"
gcloud iam service-accounts create iris-api-runtime --project="$PROJECT_ID" --display-name="iris API runtime"

# 2. A workload identity pool with a GitHub OIDC provider. The attribute condition admits only tokens issued
#    for workflows of GITHUB_REPO; without it, any GitHub repository could present tokens to the pool.
gcloud iam workload-identity-pools create github --project="$PROJECT_ID" --location=global \
  --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc github-actions --project="$PROJECT_ID" \
  --location=global --workload-identity-pool=github --display-name="GitHub Actions OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository=='${GITHUB_REPO}'"

# 3. Allow workflows of GITHUB_REPO to impersonate the deployer service account
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" --project="$PROJECT_ID" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${GITHUB_REPO}"

# 4. Permissions of the deployer: push images to the repository, deploy Cloud Run services (roles/run.admin, as
#    --allow-unauthenticated sets the IAM policy of the service) and deploy services running as the runtime SA
gcloud artifacts repositories add-iam-policy-binding gzh --project="$PROJECT_ID" --location="$REGION" \
  --role=roles/artifactregistry.writer --member="serviceAccount:${DEPLOY_SA}"
gcloud projects add-iam-policy-binding "$PROJECT_ID" --role=roles/run.admin \
  --member="serviceAccount:${DEPLOY_SA}" --condition=None
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --project="$PROJECT_ID" \
  --role=roles/iam.serviceAccountUser --member="serviceAccount:${DEPLOY_SA}"

# 5. The value of the repository variable GCP_WIF_PROVIDER
gcloud iam workload-identity-pools providers describe github-actions --project="$PROJECT_ID" \
  --location=global --workload-identity-pool=github --format='value(name)'
```
The runtime service account needs no roles for this API; give it only what the service itself accesses (e.g.
`roles/secretmanager.secretAccessor` on a specific secret). Repository names can be renamed or re-registered, so
for stricter set-ups map and check the immutable `assertion.repository_id` instead, and additionally restrict
deployments to the `main` branch (`attribute.ref`).

Then copy `github_actions_ci_cd_example.yml` to `.github/workflows/iris_api.yml`, set the repository variables
listed in its header (`GCP_PROJECT_ID`, `GCP_REGION`, `GCP_ARTIFACT_REPO=gzh`, `GCP_WIF_PROVIDER` from step 5,
`GCP_DEPLOY_SA`, `GCP_RUNTIME_SA`), create a GitHub environment called `production`, and push to `main`.
Clean up the WIF set-up with `gcloud iam workload-identity-pools delete github --location=global` and
`gcloud iam service-accounts delete <email>`.

## Files
- `deploy_cloud_run.sh`: builds, pushes and deploys the image to Cloud Run with the gcloud CLI
- `github_actions_ci_cd_example.yml`: example CI/CD workflow (test, build, smoke test, push, deploy)

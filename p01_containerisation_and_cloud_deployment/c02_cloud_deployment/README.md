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
  and repository variables (see the header of `github_actions_ci_cd_example.yml`).

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

CI/CD: copy `github_actions_ci_cd_example.yml` to `.github/workflows/iris_api.yml`, set the repository variables
listed in its header, create a GitHub environment called `production`, and push to `main`.

## Files
- `deploy_cloud_run.sh`: builds, pushes and deploys the image to Cloud Run with the gcloud CLI
- `github_actions_ci_cd_example.yml`: example CI/CD workflow (test, build, smoke test, push, deploy)

## Deploying to the cloud
The container image from c00 runs anywhere a container runtime is available. This chapter covers the last mile:
storing the image in a registry, running it on a managed cloud service, and automating the path from a commit to
a deployed service with a CI/CD pipeline. The files are:
- `deploy_cloud_run.sh`: a manual deployment to Google Cloud Run with the `gcloud` CLI;
- `github_actions_ci_cd_example.yml`: an example GitHub Actions workflow (test, build, push, deploy).

### Choosing a deployment target
Cloud providers offer a spectrum of ways to run containers, from "you manage everything" to "you only provide the
image":

| Option | Examples | You manage | Typical use |
|---|---|---|---|
| Virtual machines | Compute Engine, EC2, Azure VMs | OS, Docker, scaling, load balancing | full control, GPUs, legacy |
| Managed Kubernetes | GKE, EKS, AKS | manifests, node pools (partly) | many services, complex workloads, platform teams |
| Serverless containers | Cloud Run, Azure Container Apps, Amazon ECS Express Mode / ECS on Fargate | the image and a few settings | stateless HTTP APIs such as model services |
| Managed ML endpoints | Vertex AI, SageMaker, Azure ML | the model (container optional) | MLOps integration, built-in monitoring |

For a small stateless model API, **serverless containers** are usually the best starting point: no cluster to
operate, HTTPS endpoints out of the box, automatic scaling (including to zero, so idle services cost little)
and billing by actual usage. The Kubernetes knowledge from c01 transfers directly, because these platforms use
the same concepts (images, environment variables, health checks, CPU/memory settings, revisions).

### Container registries
A registry stores and distributes images. Every cloud has one (Google Artifact Registry, Amazon ECR, Azure
Container Registry), and there are independent ones (Docker Hub, GitHub Container Registry `ghcr.io`).
Platforms pull most efficiently - and sometimes exclusively - from the registry of their own cloud, so the image
is pushed there. An image reference has the form
```
<registry host>/<namespace or project>/<repository>:<tag>       e.g.
europe-west6-docker.pkg.dev/my-project/gzh/gzh-iris-api:3f2c1a9b7d4e
```
Good practice for tags:
- tag images with the **git commit SHA** (and optionally a semantic version), never deploy `latest`;
- treat tags as immutable; for full reproducibility, deploy by **digest** (`...@sha256:...`), which identifies
  the exact image content;
- build for the target architecture: cloud machines are mostly `linux/amd64`, so on Apple Silicon build with
  `docker build --platform linux/amd64` (or multi-platform images with `docker buildx`).

### Example: Google Cloud Run
Cloud Run runs a container image as an HTTPS service and scales the number of instances with the number of
concurrent requests. The script `deploy_cloud_run.sh` performs these steps (run it from the repository root):
```bash
# one-off set-up: enable the APIs and create a Docker repository
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create gzh --repository-format=docker --location=europe-west6
gcloud auth configure-docker europe-west6-docker.pkg.dev

# build for linux/amd64 and push
IMAGE=europe-west6-docker.pkg.dev/$PROJECT_ID/gzh/gzh-iris-api:$(git rev-parse --short=12 HEAD)
docker build --platform linux/amd64 \
  -f p01_containerisation_and_cloud_deployment/c00_docker/Dockerfile -t $IMAGE .
docker push $IMAGE

# deploy a new revision
gcloud run deploy iris-api --image $IMAGE --region europe-west6 --port 5000 \
  --cpu 1 --memory 512Mi --min-instances 0 --max-instances 3 --concurrency 40 \
  --set-env-vars LOG_LEVEL=INFO --allow-unauthenticated
```
Notes on the settings:
- **Port**: Cloud Run sends traffic to the port given by `--port` and passes it to the container in the `PORT`
  environment variable. The Dockerfile's start command uses `${PORT}`, so the image works with any value.
- **Concurrency** is the maximum number of simultaneous requests per instance. CPU-bound inference benefits
  from a moderate value; more traffic leads to more instances, up to `--max-instances` (a cost guard).
- **Scale to zero and cold starts**: with `--min-instances 0` an idle service costs (almost) nothing, but the first
  request after a pause waits for a container to start and the model to load (the lifespan in p00). Small images
  and fast model loading reduce cold-start latency; `--min-instances 1` avoids it at a fixed cost. Large models
  (e.g. LLMs) are better served from persistent instances or GPU-enabled platforms.
- **Health checks**: Cloud Run checks that the container listens on the port at start-up; HTTP startup and
  liveness probes (e.g. on `/ready` and `/health`) can be configured in the service definition.
- **Access**: `--allow-unauthenticated` makes the endpoint public, which is fine for a demo (and may be blocked by
  organisation policies). Production APIs require authentication, e.g. Cloud Run IAM (callers send an identity
  token), an API gateway, or API keys checked by the application.

The service URL is printed by the script and can be used directly with the client from p00:
`API_URL=https://iris-api-....run.app python -m p00_fundamentals.c02_fastapi.main_client`.
Every deployment creates a new immutable *revision*; traffic can be split between revisions for canary releases
and shifted back for an instant rollback. Delete the service and repository afterwards to avoid costs.

### Equivalents on Azure and AWS
On **Azure Container Apps** the image can be built in the cloud by Azure Container Registry and deployed in two
commands (after `az login` and creating a resource group and registry):
```bash
az acr build --registry <acr-name> --image gzh-iris-api:0.1.0 \
  --file p01_containerisation_and_cloud_deployment/c00_docker/Dockerfile .
az containerapp up --name iris-api --resource-group <rg> \
  --image <acr-name>.azurecr.io/gzh-iris-api:0.1.0 --registry-server <acr-name>.azurecr.io \
  --ingress external --target-port 5000
```
Container Apps is built on Kubernetes and KEDA and supports scale to zero, revisions and traffic splitting.

On **AWS**, the image is pushed to Amazon ECR
(`aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com`)
and run on Amazon ECS with the serverless Fargate launch type. *ECS Express Mode* is the closest equivalent of
Cloud Run: from an image, a port and a few settings it creates the service together with an HTTPS endpoint (an
Application Load Balancer), auto-scaling and logging; alternatively, these resources are configured one by one
(more configuration, more control). AWS App Runner, formerly AWS's closest match to Cloud Run, no longer accepts
new customers since 30 April 2026, and AWS points to ECS Express Mode as its replacement. Check the current
offerings of your provider, as managed services evolve quickly.

### Configuration and secrets in the cloud
The twelve-factor configuration of the service pays off here: settings are passed as environment variables of
the service. Secrets (API keys, database passwords) must not be put in plain environment variables of the
deployment command or in the image, but stored in a secret manager (Google Secret Manager, AWS Secrets Manager,
Azure Key Vault) and referenced by the service; the platform injects them at runtime. The service itself should
run under a dedicated identity (service account / managed identity) with only the permissions it needs.

### CI/CD with GitHub Actions
Manual deployments do not scale and are error-prone. A CI/CD pipeline turns every change into a tested, versioned,
deployed image. `github_actions_ci_cd_example.yml` is an example workflow; it is stored with this chapter (not in
`.github/workflows/`, so it does not run in this repository). It defines three jobs:
1. **test** - installs the dependencies and runs the pytest suite of p00. Nothing is built if a test fails.
2. **build** - builds the image with Docker Buildx (with layer caching in the GitHub Actions cache), runs it and
   smoke-tests `/ready` and `/v1/predict` with `curl`, and - except for pull requests - tags exactly this tested
   image with the commit SHA and pushes it to Artifact Registry (no second build).
3. **deploy** - only for pushes to `main`: deploys the image to Cloud Run and smoke-tests the live URL. It runs in
   the GitHub *environment* `production`, which can require a manual approval.

The workflow uses no stored credentials: with **Workload Identity Federation** (WIF), GitHub issues a short-lived
OIDC token for each workflow run (`permissions: id-token: write`), and Google Cloud exchanges it for short-lived
credentials of a deployer service account, so no service account key has to be created, stored or rotated. This
requires a one-off set-up in the Google Cloud project (the `gcloud` commands are in this chapter's `README.md`):
1. a *workload identity pool* with an *OIDC provider* for the issuer `https://token.actions.githubusercontent.com`,
   which maps claims of the token to attributes (e.g. `attribute.repository=assertion.repository`) and has an
   *attribute condition* such as `assertion.repository=='OWNER/REPO'`;
2. a binding of the role `roles/iam.workloadIdentityUser` on the deployer service account for the principals of
   this repository (`principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL/attribute.repository/OWNER/REPO`);
3. the roles the deployer needs: `roles/artifactregistry.writer` (push images), `roles/run.admin` (deploy
   services and, for `--allow-unauthenticated`, set their IAM policy) and `roles/iam.serviceAccountUser` on the
   service account that the Cloud Run service runs as.

Only with the attribute condition (and the repository-specific binding) are the credentials restricted to a
specific repository; a pool without a condition would admit tokens from *any* GitHub repository, which is why
Google requires a condition for GitHub providers. The values that identify the project, the provider and the
service accounts are configured as repository *variables* (they are not secret). The workflow triggers only on
changes to the relevant folders (`paths`), which is useful in a monorepo like this one.

Typical extensions are a vulnerability scan of the image (e.g. Trivy) before pushing, deployment to a staging
environment with integration tests before production, gradual traffic shifting (canary) with automatic rollback
based on error rates, and a scheduled workflow that retrains the model and opens a pull request with the new
model version - turning CI/CD into continuous training (CT), the core of MLOps.

### Operating the service
Once deployed, a service must be observed. Managed platforms collect the container's stdout/stderr logs and basic
metrics (request count, latency percentiles, error rates, instance count, CPU and memory) automatically. For
model services, add:
- **alerts** on error rate and latency (service level objectives);
- **structured logs** of requests and predictions (without personal data), which make it possible to
- **monitor drift**: compare the distribution of inputs and predictions in production with the training data
  and retrain when they diverge;
- **cost monitoring**, especially for GPU instances and services with a minimum number of instances.

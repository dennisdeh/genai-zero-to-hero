## Environment
- Docker Engine with BuildKit (Docker 23+ / Docker Desktop) and the Docker Compose plugin.
- Python conda environment (only for the client and for running the service without Docker):
  `p00_fundamentals/c02_fastapi/conda_env_gzh_api0.yml`
- Docker compose setup: `docker-compose.yml` (the API service and an optional client service)
- Environment variables (all optional):
  - `API_PORT_HOST`: host port of the API in docker compose (default: `5000`)
  - container settings: `MODEL_DIR`, `LOG_LEVEL`, `MAX_BATCH_SIZE`, `PORT` (see `p00_fundamentals/c02_fastapi/README.md`)

## How to run
All commands are run from the **repository root**, which is the build context (the image needs the code in
`p00_fundamentals/c02_fastapi`).

Build and run with plain Docker:
```bash
docker build -f p01_containerisation_and_cloud_deployment/c00_docker/Dockerfile -t gzh-iris-api:0.1.0 .
docker run --rm -d --name gzh-iris-api -p 5000:5000 gzh-iris-api:0.1.0
docker ps                                      # wait for "(healthy)"
python -m p00_fundamentals.c02_fastapi.main_client
docker stop gzh-iris-api
```
Build a different model version: add `--build-arg MODEL_VERSION=1.1.0` (and use a new image tag).

With Docker Compose:
```bash
docker compose -f p01_containerisation_and_cloud_deployment/c00_docker/docker-compose.yml up --build -d
docker compose -f p01_containerisation_and_cloud_deployment/c00_docker/docker-compose.yml ps
docker compose -f p01_containerisation_and_cloud_deployment/c00_docker/docker-compose.yml run --rm client
docker compose -f p01_containerisation_and_cloud_deployment/c00_docker/docker-compose.yml down
```

The API documentation is then available at http://localhost:5000/docs.

## Files
- `Dockerfile`: multi-stage build (builder installs dependencies and trains the model, runtime is a slim
  non-root image with a health check)
- `Dockerfile.dockerignore`: Dockerfile-specific ignore file (allowlist of the files sent to the builder)
- `requirements.txt`: Python dependencies of the image
- `docker-compose.yml`: API service plus an on-demand client service on a shared network

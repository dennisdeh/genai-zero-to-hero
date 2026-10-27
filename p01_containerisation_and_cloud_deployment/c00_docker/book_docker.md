## Docker: containerising a model service
In this chapter the FastAPI model service from p00 (`p00_fundamentals/c02_fastapi`) is packaged into a
container image, run with Docker and orchestrated locally with Docker Compose. The files are:
- `Dockerfile`: the build recipe of the image (multi-stage, slim base image, non-root user, health check);
- `Dockerfile.dockerignore`: what is (not) sent to the builder;
- `requirements.txt`: the Python dependencies installed in the image;
- `docker-compose.yml`: the service definition for local use, including a client on a shared network.

### The Dockerfile, step by step
A Dockerfile is a sequence of instructions, each of which creates a layer of the image (or sets metadata).
The most important instructions are:

| Instruction | Purpose |
|---|---|
| `FROM` | base image to start from (and name of a build stage) |
| `ARG` / `ENV` | build-time variables / environment variables of the container |
| `RUN` | execute a command at build time (installing packages, training a model, ...) |
| `COPY` | copy files from the build context (or another stage) into the image |
| `WORKDIR` | working directory for subsequent instructions and the container |
| `USER` | user that runs subsequent instructions and the container |
| `EXPOSE` | documents the port the application listens on (does not publish it) |
| `HEALTHCHECK` | command that Docker runs periodically to determine whether the container is healthy |
| `CMD` / `ENTRYPOINT` | the process started when the container runs |

#### Choosing a base image
The image builds on `python:3.13-slim`, a Debian-based image with Python but without compilers and most
system tools. Compared with the full `python:3.13` image it is several hundred megabytes smaller, which means
faster pulls, faster scaling and a smaller attack surface. `alpine` images are smaller still, but use the musl C
library for which many scientific Python wheels are not available, which can lead to long source builds.
"Distroless" images remove even the shell and package manager and are an option for hardened production images.

#### Layer caching: order matters
Docker caches each layer and reuses it as long as the instruction and its inputs are unchanged. As soon as one
layer changes, all subsequent layers are rebuilt. Hence the Dockerfile:
1. copies only `requirements.txt` and installs the dependencies (slow, rarely changes);
2. then copies the application code (fast, changes often).

Editing `main_api.py` therefore only re-runs the last few steps; the dependency layer comes from the cache.
`PIP_NO_CACHE_DIR=1` keeps pip's download cache out of the layer.

#### Multi-stage builds
The Dockerfile has two stages. The **builder** stage creates a virtual environment in `/opt/venv`, installs the
dependencies, copies the code and trains the model. The **runtime** stage starts again from a clean slim image and
copies only the results with `COPY --from=builder`: the virtual environment, the code and the model in
`/opt/model`. Anything else that accumulated in the builder (build tools, caches, temporary files) does not end up
in the final image. For a pure-Python service the gain is moderate; for images that need compilers (e.g.
packages built from source) or large build-time downloads it is substantial.

Because both stages use the same `python:${PYTHON_VERSION}-slim` base, the virtual environment can be copied
as a whole. Training the model at build time is a design decision: the image is self-contained and the model
version is tied to the image tag (set it with `--build-arg MODEL_VERSION=1.1.0`). For large models or frequent
retraining, a common alternative is to download the model from a registry or object storage at start-up, or to
mount it as a volume (see below), so that the same image can serve different models.

#### Running as a non-root user
By default, processes in a container run as `root`. If an attacker exploited a vulnerability in the
application, they would have root privileges inside the container, which makes escaping to the host easier.
The runtime stage therefore creates a user with the fixed numeric ID 10001 and switches to it with
`USER 10001:10001`. A numeric ID (rather than a name) lets Kubernetes verify the `runAsNonRoot` policy.
The application files are owned by root and only readable by the user, so the code cannot be modified at runtime.

#### Health check
`HEALTHCHECK` lets Docker probe the container periodically. Slim images contain no `curl`, so Python's
`urllib` is used to call the readiness endpoint `/ready`; a non-2xx response or a timeout makes the command
fail. After `--retries` consecutive failures the container is marked *unhealthy*. Docker Compose can wait for a
healthy service before starting dependants (`depends_on: condition: service_healthy`). Kubernetes ignores
`HEALTHCHECK` and uses its own probes instead (next chapter).

#### The start command and signals
```dockerfile
CMD ["sh", "-c", "exec uvicorn p00_fundamentals.c02_fastapi.main_api:app --host 0.0.0.0 --port ${PORT}"]
```
- `--host 0.0.0.0`: inside a container, `127.0.0.1` is the container's own loopback interface; to be reachable
  through a published port, the server must listen on all interfaces.
- `${PORT}`: the shell form expands the environment variable, so platforms that inject a port (e.g. Google
  Cloud Run) are supported. The default `PORT=5000` is set with `ENV`.
- `exec`: replaces the shell by Uvicorn, so Uvicorn becomes process 1 and directly receives the `SIGTERM` sent by
  `docker stop` or Kubernetes. It can then finish in-flight requests and run the lifespan shut-down code. Without
  `exec`, the shell would not forward the signal and the container would be killed after a grace period.

`PYTHONUNBUFFERED=1` makes Python write logs immediately to stdout/stderr, where `docker logs` (and every
container platform) collects them. `PYTHONDONTWRITEBYTECODE=1` prevents `.pyc` files from being written at
runtime, which also allows a read-only root file system.

### The build context and the dockerignore file
`docker build` sends a *build context* - a directory - to the builder, and `COPY` can only access files within
it. This Dockerfile lives in `p01_containerisation_and_cloud_deployment/c00_docker`, but needs the code in
`p00_fundamentals/c02_fastapi`. The context is therefore the **repository root**, and the Dockerfile is passed
with `-f`:
```bash
docker build -f p01_containerisation_and_cloud_deployment/c00_docker/Dockerfile -t gzh-iris-api:0.1.0 .
```
The final `.` is the context; all `COPY` source paths are relative to it.

Sending the whole repository (datasets, model checkpoints, the `.git` folder) would be slow and could leak
secrets such as `.env` files into the image. A `.dockerignore` file excludes files from the context. Usually it
lies in the context root; with BuildKit (the default builder of current Docker versions) a Dockerfile-specific
ignore file named `<Dockerfile name>.dockerignore` next to the Dockerfile takes precedence. That is why this
chapter uses `Dockerfile.dockerignore`. It follows an *allowlist* strategy: ignore everything (`*`), then
re-include only the needed paths with `!`, and finally exclude generated files (`outputs`, `__pycache__`, `.env`)
even inside the re-included folders.

### Running the container
```bash
docker run --rm -d --name gzh-iris-api -p 5000:5000 gzh-iris-api:0.1.0
docker ps                 # STATUS shows (healthy) once the health check passes
docker logs -f gzh-iris-api
API_URL=http://localhost:5000 python -m p00_fundamentals.c02_fastapi.main_client
docker stop gzh-iris-api
```
`-p 5000:5000` publishes container port 5000 on host port 5000 (`host:container`). Settings are changed with
`-e`, e.g. `-e LOG_LEVEL=DEBUG -e MAX_BATCH_SIZE=100`. Useful commands for inspection are
`docker exec -it gzh-iris-api sh` (a shell in the running container), `docker image ls` (image sizes),
`docker history gzh-iris-api:0.1.0` (the layers and their sizes) and `docker inspect`.

### Volumes: data that outlives the container
A container's writable layer disappears with the container. Persistent or shared data is stored in volumes:
- **named volumes** (`docker volume create`), managed by Docker - used e.g. for the Ollama model cache in
  `p07_llms/c01_running_llms/s01_ollama/docker-compose.yml`;
- **bind mounts**, which map a host folder into the container, e.g. to serve a model trained on the host
  without rebuilding the image:
  ```bash
  docker run --rm -p 5000:5000 \
      -v "$(pwd)/p00_fundamentals/c02_fastapi/outputs/model:/models/iris:ro" \
      -e MODEL_DIR=/models/iris gzh-iris-api:0.1.0
  ```
  The `:ro` suffix mounts the folder read-only.

### Networks: how containers talk to each other
Containers attached to the same user-defined network can reach each other by *service name* via Docker's
internal DNS, without publishing ports on the host. Docker Compose creates such a network automatically; the
compose file in this chapter declares it explicitly (`backend`) to make it visible.

### Docker Compose
Compose describes one or more services, their build, ports, environment, volumes and networks in a YAML file,
replacing long `docker run` commands. In `docker-compose.yml`:
- the `api` service is built from the repository root (`context: ../..`, because relative paths are resolved
  relative to the compose file) and tagged `gzh-iris-api:0.1.0`;
- it runs with a read-only root file system (`read_only: true`) and a `tmpfs` for `/tmp`, a cheap hardening
  measure;
- the `client` service reuses the same image to run `main_client.py` against `http://api:5000` - the host name
  `api` is resolved by the compose network. It belongs to the profile `client`, so `docker compose up` does not
  start it, and `depends_on ... condition: service_healthy` makes it wait for the health check.

```bash
docker compose -f p01_containerisation_and_cloud_deployment/c00_docker/docker-compose.yml up --build -d
docker compose -f p01_containerisation_and_cloud_deployment/c00_docker/docker-compose.yml run --rm client
docker compose -f p01_containerisation_and_cloud_deployment/c00_docker/docker-compose.yml down
```

### Image hygiene and security
- **Tags**: never deploy `latest`; use immutable, meaningful tags such as a semantic version or the git commit
  SHA, so that every deployment can be traced and rolled back.
- **Pin versions** of the base image and Python packages for reproducible builds (the example uses lower
  bounds only, to stay current).
- **Scan images** for known vulnerabilities, e.g. with `docker scout cves` or Trivy, and rebuild regularly to
  pick up security patches of the base image.
- **No secrets in images**: anything in a layer can be extracted with `docker history`/`docker save`. Pass secrets
  at runtime (environment variables from a secret store, mounted files) instead.
- **GPUs**: containers can use NVIDIA GPUs via the NVIDIA Container Toolkit (`--gpus all`, or the
  `deploy.resources.reservations.devices` section in compose, as used for Ollama in p07).

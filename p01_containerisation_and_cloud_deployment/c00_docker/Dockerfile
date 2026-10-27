# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------------------------------
# Container image of the iris model-serving API from p00_fundamentals/c02_fastapi.
#
# The build context must be the REPOSITORY ROOT, because the image needs code from p00 while this
# Dockerfile lives in p01. Build from the repository root with:
#   docker build -f p01_containerisation_and_cloud_deployment/c00_docker/Dockerfile -t gzh-iris-api:0.1.0 .
# The file Dockerfile.dockerignore next to this Dockerfile restricts what is sent to the builder
# (BuildKit picks up <Dockerfile-name>.dockerignore automatically).
#
# Multi-stage build:
#   1. "builder": installs the dependencies into a virtual environment and trains the model
#   2. "runtime": a clean slim image that only receives the virtual environment, code and model
# -----------------------------------------------------------------------------------------------------
ARG PYTHON_VERSION=3.13

# ===== Stage 1: builder ==============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 1. Create a virtual environment that can be copied as a whole into the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 2. Install the dependencies first: this layer is cached and only rebuilt when requirements change
COPY p01_containerisation_and_cloud_deployment/c00_docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# 3. Copy the application code (changes more often, hence after the dependencies)
WORKDIR /app
COPY p00_fundamentals/c02_fastapi/ /app/p00_fundamentals/c02_fastapi/

# 4. Train the model at build time, so the image is self-contained and versioned together with the model.
#    (Alternatives: download the model from a registry at start-up, or mount it as a volume.)
ARG MODEL_VERSION=1.0.0
RUN python -m p00_fundamentals.c02_fastapi.main_train_model \
        --output-dir /opt/model --version "${MODEL_VERSION}"

# ===== Stage 2: runtime ==============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="gzh-iris-api" \
      org.opencontainers.image.description="FastAPI service serving a scikit-learn iris classifier" \
      org.opencontainers.image.source="https://github.com/dennisdeh/genai-zero-to-hero"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=/opt/model \
    LOG_LEVEL=INFO \
    PORT=5000

# 5. Create an unprivileged user with a fixed numeric UID (Kubernetes can then verify runAsNonRoot)
RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

# 6. Copy only what is needed from the builder stage (no compilers, caches or build files)
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/model /opt/model
WORKDIR /app
COPY --from=builder /app /app

USER 10001:10001
EXPOSE 5000

# 7. Health check used by docker / docker compose (slim images have no curl, so python is used).
#    It calls the readiness endpoint, i.e. the container is "healthy" once the model is loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '5000') + '/ready', timeout=3)" || exit 1

# 8. Start the ASGI server. Bind to 0.0.0.0 to be reachable from outside the container. The port is
#    taken from $PORT (some platforms, e.g. Google Cloud Run, inject it). `exec` makes uvicorn PID 1,
#    so it receives SIGTERM directly and shuts down gracefully.
CMD ["sh", "-c", "exec uvicorn p00_fundamentals.c02_fastapi.main_api:app --host 0.0.0.0 --port ${PORT}"]

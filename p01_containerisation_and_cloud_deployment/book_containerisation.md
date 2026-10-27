# Containerisation and Cloud Deployment
This part covers the basics of containerisation and deployment to e.g. cloud providers in a production environment.

The running example is the FastAPI model service from p00 (`p00_fundamentals/c02_fastapi`), which classifies
iris flowers with a scikit-learn pipeline. In three chapters, it is
1. packaged into a container image and run with Docker and Docker Compose (c00);
2. deployed to a local Kubernetes cluster with probes, resource limits and autoscaling (c01);
3. pushed to a container registry and deployed to a managed cloud service, automated with a CI/CD pipeline (c02).

## Why containers for machine learning?
"It works on my machine" is a classic problem in software engineering, and it is particularly severe in machine
learning: a model depends not only on its code, but on exact versions of numerical libraries (a pickled
scikit-learn model may not load with another version), system libraries (BLAS, CUDA, cuDNN), the Python version
and the model artefacts themselves. A **container** packages all of this together with the application into one
unit that runs identically on a laptop, a CI server, a Kubernetes cluster or a cloud service.

Compared to virtual machines, containers do not include an operating system kernel: they are isolated processes
on the host's kernel, using Linux namespaces (what a process can see: files, processes, network) and control
groups (what it can use: CPU, memory). They therefore start in seconds, have little overhead and many of them fit
on one machine. Conda environments (p00) isolate Python packages; containers isolate the whole runtime.

The benefits for ML systems are:
- **reproducibility**: the same image is tested, deployed and can be re-run years later;
- **portability**: one artefact for all environments and cloud providers;
- **scalability**: orchestrators start more copies of an image when the load increases;
- **isolation**: services with conflicting dependencies (e.g. different CUDA versions) run side by side;
- **standardised operations**: logging, health checks, resource limits and deployments work the same for every
  service, whatever language or framework it uses.

## Docker concepts
Docker is the most widely used tool to build and run containers. Its main concepts are:
- **Image**: a read-only template containing a file system (OS libraries, Python, packages, code, model) and
  metadata (environment variables, start command, exposed ports). Images are built from a **Dockerfile** and are
  identified by a name and tag (e.g. `python:3.13-slim`) or by a content digest (`sha256:...`).
- **Layers**: an image is a stack of layers, one per file-system-changing Dockerfile instruction. Layers are
  content-addressed, cached and shared between images: if ten images build on `python:3.13-slim`, its layers are
  stored and downloaded once. Ordering the Dockerfile from rarely to frequently changing steps makes rebuilds fast.
- **Container**: a running (or stopped) instance of an image. Docker adds a thin writable layer on top of the
  image layers; anything written there is lost when the container is removed. Containers should therefore be
  *ephemeral* and *stateless*: they can be stopped, replaced and multiplied at any time.
- **Volumes**: the mechanism for persistent data. Named volumes are managed by Docker; bind mounts map host
  directories into the container (e.g. a folder with model weights or a dataset).
- **Networks**: containers communicate over virtual networks. On a user-defined network, containers reach each
  other by name through Docker's DNS; ports are published to the host only where needed (`-p host:container`).
- **Registry**: a server that stores and distributes images (Docker Hub, GitHub Container Registry, the
  registries of the cloud providers). `docker push` uploads an image, `docker pull` downloads it.
- **Docker Compose**: a YAML description of a multi-container application (services, networks, volumes), started
  with one command. It is used throughout this book, e.g. for Ollama and vector databases in p07 and p08.

Images follow the open OCI (Open Container Initiative) standard, so images built with Docker also run with other
runtimes such as containerd (used by Kubernetes) or Podman.

## From a container to production
A single container on a single machine is not yet a production system. Production requires that crashed
containers are restarted, that traffic is spread over several replicas, that new versions are rolled out without
downtime, and that capacity follows demand. **Kubernetes** automates this for clusters of machines, based on
declarative manifests (c01). **Managed cloud services** such as Google Cloud Run, Azure Container Apps or AWS App
Runner take this further and run a container image without any cluster to operate (c02). Finally, **CI/CD**
pipelines connect the pieces: every code change is tested, built into an image, pushed to a registry and deployed
automatically (c02).

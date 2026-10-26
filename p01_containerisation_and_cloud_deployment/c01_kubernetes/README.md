## Environment
- A local Kubernetes cluster: [kind](https://kind.sigs.k8s.io/) (requires Docker) or
  [minikube](https://minikube.sigs.k8s.io/), plus `kubectl` (includes Kustomize via `kubectl apply -k`).
- The container image `gzh-iris-api:0.1.0`, built as described in `../c00_docker/README.md`.
- Python conda environment for the client and the load test: `p00_fundamentals/c02_fastapi/conda_env_gzh_api0.yml`
- Docker compose setup: none (see `../c00_docker`).
- Environment variables: `API_URL` for the client/load test (default: `http://localhost:5000`).
  The container configuration is set in `configmap.yaml`.

## How to run
All commands are run from the repository root.

1. Build the image and create a cluster:
   ```bash
   docker build -f p01_containerisation_and_cloud_deployment/c00_docker/Dockerfile -t gzh-iris-api:0.1.0 .
   kind create cluster --name gzh
   kind load docker-image gzh-iris-api:0.1.0 --name gzh
   ```
2. Install the metrics-server (needed by the HorizontalPodAutoscaler):
   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
   kubectl patch deployment metrics-server -n kube-system --type=json \
     -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
   ```
   (minikube: `minikube addons enable metrics-server`)
3. Deploy and check:
   ```bash
   kubectl apply -k p01_containerisation_and_cloud_deployment/c01_kubernetes
   kubectl rollout status deployment/iris-api -n gzh
   kubectl get pods,svc,hpa -n gzh
   ```
4. Call the service:
   ```bash
   kubectl port-forward -n gzh service/iris-api 5000:80
   python -m p00_fundamentals.c02_fastapi.main_client      # in a second terminal
   ```
5. Generate load and watch the autoscaler:
   ```bash
   python -m p01_containerisation_and_cloud_deployment.c01_kubernetes.main_load_test --duration 300 --concurrency 16
   kubectl get hpa -n gzh --watch
   ```
6. Clean up:
   ```bash
   kubectl delete -k p01_containerisation_and_cloud_deployment/c01_kubernetes
   kind delete cluster --name gzh
   ```

## Files
- `namespace.yaml`, `configmap.yaml`, `deployment.yaml`, `service.yaml`, `hpa.yaml`, `pdb.yaml`: the manifests
- `kustomization.yaml`: bundles the manifests (`kubectl apply -k`), sets namespace, labels and image tag
- `main_load_test.py`: multi-threaded load generator reporting throughput and latency percentiles

## Kubernetes: orchestrating the model service
Docker runs containers on one machine. In production a service typically runs as several copies (replicas)
spread over several machines, must be restarted when it crashes, updated without downtime and scaled with the
load. This is the job of a container *orchestrator*, and Kubernetes (K8s) is the de-facto standard.
All major clouds offer managed Kubernetes (Google GKE, Amazon EKS, Azure AKS), and for learning it runs
locally with [kind](https://kind.sigs.k8s.io/) (Kubernetes in Docker) or [minikube](https://minikube.sigs.k8s.io/).

This chapter deploys the container image from the previous chapter (`gzh-iris-api:0.1.0`) to a local cluster.

### Core concepts
Kubernetes is *declarative*: one describes the desired state in YAML manifests and applies them with
`kubectl apply`. Controllers in the control plane then continuously compare the actual with the desired state
and act to reconcile them (start pods, replace failed ones, ...). The objects used here are:
- **Pod**: the smallest deployable unit, one or more containers sharing a network namespace and volumes.
  Pods are ephemeral and are not created directly.
- **Deployment**: manages a set of identical pods (via a ReplicaSet), keeps the desired number of replicas
  running and performs rolling updates and rollbacks.
- **Service**: a stable virtual IP and DNS name that load-balances over all *ready* pods selected by labels.
- **ConfigMap** (and **Secret**): configuration decoupled from the image, injected as environment variables or files.
- **HorizontalPodAutoscaler (HPA)**: changes the number of replicas based on metrics such as CPU utilisation.
- **PodDisruptionBudget (PDB)**: limits how many pods may be taken down at once during voluntary disruptions.
- **Namespace**: a logical grouping of resources (here `gzh`).

Objects find each other through **labels and selectors**: the Deployment's pod template carries the label
`app.kubernetes.io/name: iris-api`, and the Deployment, Service, PDB select pods with exactly this label.

### The manifests
#### Deployment (`deployment.yaml`)
The Deployment runs two replicas of the image. The most important settings for a model service are:

**Probes.** Kubernetes uses three kinds of HTTP checks, mapped onto the endpoints of the FastAPI service:
- the *startup probe* (`/health`) gives the application up to 60 s to start; the other probes only begin once it
  succeeds. For large models with long loading times, this avoids being killed during start-up;
- the *liveness probe* (`/health`) restarts a container whose process hangs;
- the *readiness probe* (`/ready`) removes a pod from the Service's endpoints while it cannot serve
  predictions, e.g. because the model is not loaded. No traffic is sent to it, but it is not restarted.

**Resources.** `requests` (100m CPU = 0.1 core, 256 MiB memory) are what the scheduler reserves on a node when
placing the pod; `limits` (500m, 512 MiB) are enforced at runtime: CPU above the limit is throttled, and a
container exceeding its memory limit is killed (`OOMKilled`). Setting both makes the pod's resource usage
predictable. Memory requests should cover the model size plus a margin; for ML services it is also common
to set memory request = limit to avoid OOM kills under node pressure.

**Rolling updates.** With `maxSurge: 1` and `maxUnavailable: 0`, an update (e.g. a new image tag) starts one new
pod, waits until it is *ready*, then terminates an old one, and so on - zero downtime, provided the readiness
probe is meaningful. A failed update is rolled back with `kubectl rollout undo`.

**Security context.** The pod must run as non-root user 10001 (matching the Dockerfile), cannot escalate
privileges, has all Linux capabilities dropped and a read-only root file system; an `emptyDir` volume provides a
writable `/tmp`. These settings satisfy the Kubernetes "restricted" Pod Security Standard.

**Image pull policy.** `IfNotPresent` makes Kubernetes use the image that was loaded into the local cluster
instead of trying to pull it from a registry. (For the tag `latest` the default would be `Always`, another
reason to avoid it.)

#### Service (`service.yaml`)
A `ClusterIP` Service exposes port 80 inside the cluster and forwards to the named container port `http` (5000).
Other workloads reach the API at `http://iris-api.gzh.svc.cluster.local` (or `http://iris-api` from the same
namespace). From the host, `kubectl port-forward` is the simplest access path; in the cloud one would use a
Service of type `LoadBalancer` or an Ingress/Gateway with TLS.

#### ConfigMap (`configmap.yaml`)
The ConfigMap holds `LOG_LEVEL`, `MAX_BATCH_SIZE` and `MODEL_DIR`, injected via `envFrom`. Because the API reads
its configuration from the environment (p00), no code or image change is needed to reconfigure it. Note that pods
only see changed values after a restart (`kubectl rollout restart deployment/iris-api -n gzh`).

#### HorizontalPodAutoscaler (`hpa.yaml`)
The HPA keeps between 2 and 6 replicas and targets an average CPU utilisation of 60 % *of the CPU request*.
Roughly, the desired number of replicas is
$$\text{desired} = \left\lceil \text{current} \cdot \frac{\text{current utilisation}}{\text{target utilisation}} \right\rceil.$$
With 2 pods at 120 % utilisation, the HPA scales to $\lceil 2 \cdot 120/60 \rceil = 4$ pods. The
`behavior.scaleDown` section waits two minutes and removes at most one pod per minute, to avoid flapping.
The HPA needs the *metrics-server* to obtain CPU usage. For ML services, CPU is a reasonable proxy for
CPU-bound inference; GPU services or request-queue based scaling use custom metrics (e.g. via Prometheus or KEDA).

#### PodDisruptionBudget (`pdb.yaml`) and Kustomize (`kustomization.yaml`)
The PDB guarantees at least one available pod while nodes are drained for maintenance. The `kustomization.yaml`
lists all manifests, sets the namespace and a common label, and can replace the image reference - so the same
manifests can point to a registry image in the cloud (`images: - name: gzh-iris-api, newName: <registry>/...`).
Larger setups use Kustomize overlays (dev/staging/prod) or Helm charts.

### Walkthrough with kind
```bash
# 1. create a local cluster and load the image built in c00_docker (no registry needed)
kind create cluster --name gzh
kind load docker-image gzh-iris-api:0.1.0 --name gzh

# 2. install the metrics-server (required by the HPA; kind needs the --kubelet-insecure-tls flag)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

# 3. deploy everything and wait for the rollout
kubectl apply -k p01_containerisation_and_cloud_deployment/c01_kubernetes
kubectl rollout status deployment/iris-api -n gzh
kubectl get pods,svc,hpa -n gzh

# 4. access the service and call it
kubectl port-forward -n gzh service/iris-api 5000:80
python -m p00_fundamentals.c02_fastapi.main_client                 # second terminal
```
With minikube, the equivalent steps are `minikube start`, `minikube image load gzh-iris-api:0.1.0` and
`minikube addons enable metrics-server`.

### Experiments
- **Self-healing**: delete a pod (`kubectl delete pod -n gzh <name>`) and watch the Deployment replace it
  (`kubectl get pods -n gzh --watch`).
- **Autoscaling**: generate load with the load-test script, and watch the HPA in another terminal:
  ```bash
  python -m p01_containerisation_and_cloud_deployment.c01_kubernetes.main_load_test --duration 300 --concurrency 16
  kubectl get hpa -n gzh --watch
  ```
  The script sends batch requests from several threads and reports throughput and latency percentiles.
  Note that `kubectl port-forward` tunnels through a single connection to one pod and can itself become the
  bottleneck; for a realistic test, run the load generator inside the cluster or expose the Service via an Ingress.
- **Rolling update**: build a new model version (`--build-arg MODEL_VERSION=1.1.0 -t gzh-iris-api:0.2.0`),
  load it into kind, then `kubectl set image deployment/iris-api api=gzh-iris-api:0.2.0 -n gzh` and follow
  `kubectl rollout status`; `/v1/model/metadata` reports the new version. Undo with `kubectl rollout undo`.
- **Readiness**: change `MODEL_DIR` in the ConfigMap to a non-existing folder and restart the deployment: the new
  pods never become ready, the rollout stalls, and - thanks to `maxUnavailable: 0` - the old pods keep serving.

Clean up with `kubectl delete namespace gzh` and `kind delete cluster --name gzh`.

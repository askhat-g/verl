# GKE KubeRay Cluster & Docker Image for TPU v6e

This directory contains the Docker image definition and KubeRay `RayCluster` manifests for running `verl` (SFT and GRPO RL) on Google Kubernetes Engine (GKE) with **Cloud TPU v6e** (`tpu-v6e-slice`) node pools:

- [`Dockerfile.tpu`](Dockerfile.tpu): Builds the unified TPU runtime image containing `torch==2.13.0`, `torch_tpu`, `torchtitan`, `vllm==0.29.0`, `vllm-torchtpu`, `jax==0.10.2`, `libtpu==0.0.47`, and `verl` (`us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20260930-grpo32b`).
- [`ray-tpu-v6e8-2slice.yaml`](ray-tpu-v6e8-2slice.yaml): KubeRay `RayCluster` manifest for **Qwen3-0.6B / 4B / 8B / 14B** provisioning 1 CPU Ray head pod (`ray-head`) and 2 multi-host TPU v6e-8 slices (`replicas: 2`, `numOfHosts: 2`, `google.com/tpu: 4` per host = 16 TPU v6e chips total) with GCS Fuse mounted at `/data`.
- [`ray-tpu-v6e16-v6e8-2slice.yaml`](ray-tpu-v6e16-v6e8-2slice.yaml): KubeRay `RayCluster` manifest for **Qwen3-32B** provisioning 1 CPU Ray head pod, 1 multi-host TPU v6e-16 trainer slice (`4x4`, `numOfHosts: 4` = 16 TPU v6e chips), and 1 multi-host TPU v6e-8 rollout slice (`2x4`, `numOfHosts: 2` = 8 TPU v6e chips; 24 chips total) with GCS Fuse mounted at `/data`.
- [`model_0_6b_config.yaml`](model_0_6b_config.yaml), [`model_4b_config.yaml`](model_4b_config.yaml), [`model_8b_config.yaml`](model_8b_config.yaml), [`model_14b_config.yaml`](model_14b_config.yaml), [`model_32b_config.yaml`](model_32b_config.yaml): Self-contained per-model `RayCluster` + `ConfigMap` manifests that use node-local storage (`hostPath: /var/data` mounted at `/data`) and automatically bootstrap the HuggingFace model checkpoint (`Qwen/Qwen3-{0.6B,4B,8B,14B,32B}`) and GSM8K parquet dataset on pod startup (no shared GCS Fuse bucket required).

| Model Size | Trainer Slice Topology | Rollout Slice Topology | Total TPU Chips | GCS Fuse Manifest | Self-Contained (HostPath + Auto-Download) Manifest |
|------------|------------------------|------------------------|-----------------|-------------------|----------------------------------------------------|
| Qwen3-0.6B | 1 x `v6e-8` (`2x4`, 8 chips) | 1 x `v6e-8` (`2x4`, 8 chips) | 16 | [`ray-tpu-v6e8-2slice.yaml`](ray-tpu-v6e8-2slice.yaml) | [`model_0_6b_config.yaml`](model_0_6b_config.yaml) |
| Qwen3-4B | 1 x `v6e-8` (`2x4`, 8 chips) | 1 x `v6e-8` (`2x4`, 8 chips) | 16 | [`ray-tpu-v6e8-2slice.yaml`](ray-tpu-v6e8-2slice.yaml) | [`model_4b_config.yaml`](model_4b_config.yaml) |
| Qwen3-8B | 1 x `v6e-8` (`2x4`, 8 chips) | 1 x `v6e-8` (`2x4`, 8 chips) | 16 | [`ray-tpu-v6e8-2slice.yaml`](ray-tpu-v6e8-2slice.yaml) | [`model_8b_config.yaml`](model_8b_config.yaml) |
| Qwen3-14B | 1 x `v6e-8` (`2x4`, 8 chips) | 1 x `v6e-8` (`2x4`, 8 chips) | 16 | [`ray-tpu-v6e8-2slice.yaml`](ray-tpu-v6e8-2slice.yaml) | [`model_14b_config.yaml`](model_14b_config.yaml) |
| Qwen3-32B | 1 x `v6e-16` (`4x4`, 16 chips) | 1 x `v6e-8` (`2x4`, 8 chips) | 24 | [`ray-tpu-v6e16-v6e8-2slice.yaml`](ray-tpu-v6e16-v6e8-2slice.yaml) | [`model_32b_config.yaml`](model_32b_config.yaml) |

---

## 🛠️ Provisioning & Managing Your GKE TPU Cluster

### 1. Provision GKE TPU v6e Node Pools

Create a GKE cluster (or connect to an existing one) and provision the TPU v6e multi-host slice node pools matching your target model size:

```bash
# Authenticate kubectl with your GKE cluster
gcloud container clusters get-credentials <CLUSTER_NAME> --region <REGION> --project <PROJECT_ID>

# For 0.6B / 4B / 8B / 14B (2 x v6e-8 slices = two 2x4 node pools, 2 hosts per pool):
gcloud container node-pools create tpu-v6e8-slice-1 \
    --cluster <CLUSTER_NAME> --region <REGION> --project <PROJECT_ID> \
    --node-locations <ZONE> --machine-type ct6e-standard-4t \
    --tpu-topology 2x4 --num-nodes 2

gcloud container node-pools create tpu-v6e8-slice-2 \
    --cluster <CLUSTER_NAME> --region <REGION> --project <PROJECT_ID> \
    --node-locations <ZONE> --machine-type ct6e-standard-4t \
    --tpu-topology 2x4 --num-nodes 2

# For 32B (additionally requires 1 x v6e-16 trainer slice = one 4x4 node pool, 4 hosts):
gcloud container node-pools create tpu-v6e16-trainer \
    --cluster <CLUSTER_NAME> --region <REGION> --project <PROJECT_ID> \
    --node-locations <ZONE> --machine-type ct6e-standard-4t \
    --tpu-topology 4x4 --num-nodes 4
```

Ensure the **KubeRay Operator** and **KubeRay TPU Initialization Webhook** are installed on the cluster so multi-host TPU slice environment variables (`TPU_WORKER_ID`, `TPU_WORKER_HOSTNAMES`, and `TPU-{num_chips}-head` resources) are injected into worker pods automatically:

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm upgrade --install kuberay-operator kuberay/kuberay-operator --version 1.2.2
```

### 2. Deploy or Switch the RayCluster

Apply either the GCS Fuse manifest or the self-contained per-model manifest:

```bash
# Option A: GCS Fuse cluster (2x v6e-8 for 0.6B/4B/8B/14B, or v6e-16 + v6e-8 for 32B)
kubectl apply -f examples/tpu/gke/ray-tpu-v6e8-2slice.yaml
# or for 32B:
kubectl apply -f examples/tpu/gke/ray-tpu-v6e16-v6e8-2slice.yaml

# Option B: Self-contained per-model cluster (auto-downloads HF checkpoint & GSM8K to /var/data)
kubectl apply -f examples/tpu/gke/model_32b_config.yaml   # or model_{0_6b,4b,8b,14b}_config.yaml
```

To **switch between models or cluster topologies** (or cleanly reset TPU device state between runs), delete the existing `RayCluster` and apply the new manifest:

```bash
kubectl delete raycluster ray-tpu-v6e-cluster --wait=true
kubectl apply -f examples/tpu/gke/model_32b_config.yaml
```

### 3. Verify Cluster Readiness & Connect to Ray Dashboard

```bash
# Wait for 1 head pod + all TPU worker pods (4 workers for 2x v6e-8; 6 workers for v6e-16 + v6e-8) to reach Running
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster -w

# Verify that all TPU chips and TPU-<N>-head slice resources are registered in Ray
HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster,ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec "${HEAD_POD}" -- ray status

# Port-forward the Ray Dashboard / Job Submission API to localhost:23333
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 > /dev/null 2>&1 &
```

Once the cluster is running, see:
- **GRPO RL Training**: [`examples/tpu/grpo/README.md`](../grpo/README.md)
- **SFT Training**: [`examples/tpu/sft/README.md`](../sft/README.md)

---

## 🧪 TPU CI (`.github/workflows/e2e_tpu.yml`)

The CI does **not** use the long-lived cluster above. It is split into tiers by TPU chip count. Each tier is its own GitHub Actions job and provisions its own short-lived RayCluster on the CI GKE cluster (`jialeic-ci-v6e8-2s-spot`) with [`tests/special_tpu/run_tpu_e2e_ci.sh`](../../../tests/special_tpu/run_tpu_e2e_ci.sh), then runs that tier's tests on it:

| CI job | Tier (`run_tpu_e2e_ci.sh` arg) | TPU workers | Scope | Tests today |
|--------|--------------------------------|-------------|-------|-------------|
| `v6e 1-chip (TPU platform tests)` | `v6e-1chip` | 1 x v6e-1 (whole `ct6e-standard-1t` node) | TPU platform | `test_tpu_platform`: [`tests/special_tpu/tpu_smoke_test.py`](../../../tests/special_tpu/tpu_smoke_test.py): `import torch_tpu`, `.to("tpu")`, bf16 matmul, autograd, TPU-vs-CPU MLP forward/backward |
| `v6e 4-chip (trainer tests)` | `v6e-4chip` | 1 x v6e-4 subslice | Trainer | `test_trainer_sft`: Qwen3-0.6B GSM8K SFT (TorchTitan FSDP2) |
| `v6e 8-chip (RL tests)` | `v6e-8chip` | 2 x v6e-4 subslices | RL | `test_rl_grpo`: Qwen3-0.6B GSM8K GRPO, 4-chip trainer on `tpu-group-0` + 4-chip vLLM rollout on `tpu-group-1` |

To add a test, write a `test_*` function in `run_tpu_e2e_ci.sh` (one Ray job, optionally with a verifier in [`tests/special_tpu/verify_tpu_e2e_log.py`](../../../tests/special_tpu/verify_tpu_e2e_log.py)) and call it from the matching tier in `run_tier`. Tests in a tier run one after another on the same RayCluster.

- **Queueing**: each tier has its own Kueue queue ([`tests/special_tpu/gke/kueue-tpu-ci.yaml`](../../../tests/special_tpu/gke/kueue-tpu-ci.yaml)), and the tier's RayCluster ([`tests/special_tpu/gke/raycluster-ci.yaml`](../../../tests/special_tpu/gke/raycluster-ci.yaml)) is labelled `kueue.x-k8s.io/queue-name: verl-tpu-ci-<1chip|4chip|8chip>`. Kueue keeps the cluster suspended until its queue has enough free TPU quota, then admits the whole cluster at once (gang admission). Concurrent PRs therefore wait in first-come-first-served order per tier (`StrictFIFO`) instead of competing for hosts, and two runs can never deadlock holding half a cluster each.

  | Queue | TPU quota | Shares with |
  |-------|-----------|-------------|
  | `verl-tpu-ci-1chip` | 1 chip on the 1x1 node pool | nothing |
  | `verl-tpu-ci-4chip` | 8 chips guaranteed (two trainer jobs) | borrows unused chips from `8chip` (cohort `verl-tpu-ci-v6e`, 16 chips total) |
  | `verl-tpu-ci-8chip` | 8 chips guaranteed (one RL job) | borrows unused chips from `4chip` |

  There is no preemption: borrowed chips are returned when the CI job finishes. Watch the queues with `kubectl get localqueue` / `kubectl get workloads`. CI only creates the queues when they are missing and never modifies them; roll out changes to `kueue-tpu-ci.yaml` with `kubectl apply -f tests/special_tpu/gke/kueue-tpu-ci.yaml`.
- **Subslicing**: each TPU worker replica requests one v6e host (`google.com/tpu: 4`) with the `cloud.google.com/gke-tpu-slice-topology: 2x2` annotation, i.e. a single-host KubeRay subslice (v6e-4) of the 2x4 (v6e-8) node pools, and shows up in Ray as its own slice (`tpu-group-<i>`). A whole host is the smallest multi-chip unit: GKE rejects smaller TPU requests on these node pools and libtpu cannot start an ICI session on a 2-chip subset of a host (`START_SESSION failed`). To avoid idling 3 chips of a host, the `v6e-1chip` tier runs on a dedicated 1x1 node pool instead (no subslice annotation, `google.com/tpu: 1`, own Kueue flavor `verl-tpu-ci-v6e-1t`). The pool scales from zero, so a 1-chip run pays a ~3 min node boot + image pull and costs nothing while idle:

  ```bash
  gcloud container node-pools create verl-ci-v6e-1t --cluster jialeic-ci-v6e8-2s-spot \
      --region us-central2 --project tpu-pytorch --node-locations us-central2-b \
      --machine-type ct6e-standard-1t --spot --disk-size 100 \
      --enable-autoscaling --num-nodes 0 --min-nodes 0 --max-nodes 1
  ```
- **Teardown**: the script deletes its RayCluster on exit (success, failure, timeout or signal), the workflow repeats the deletion in an `if: always()` step, and every run reaps CI RayClusters older than 3 hours.

> [!WARNING]
> Do not apply `ray-tpu-v6e8-2slice.yaml` on the CI cluster: it pins all 16 chips outside of Kueue and CI RayClusters would stay unschedulable.

Run a tier by hand (from a machine with `kubectl` access and the `ray` CLI):

```bash
bash tests/special_tpu/run_tpu_e2e_ci.sh v6e-1chip   # or v6e-4chip / v6e-8chip
```

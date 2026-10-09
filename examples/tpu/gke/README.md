# GKE Cluster Manifests & Docker Image for TPU (v5p, v6e & 7x)

This directory contains the Docker image definition and Kubernetes manifests for running `verl` (SFT and GRPO RL) on Google Kubernetes Engine (GKE) with **Cloud TPU v5p**, **v6e**, and **7x** node pools:

- [`Dockerfile.tpu`](Dockerfile.tpu): Builds the unified TPU runtime image containing `torch==2.13.0+cpu`, `torch_tpu`, `tpu-sync-torch`, `torchtitan`, `vllm==0.29.0`, `vllm-torchtpu`, `jax==0.10.2`, `libtpu==0.0.47`, and `verl` dependencies (`us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20261006-tsync1006`).
- [`ray-jobset-v5p-nap.yaml`](ray-jobset-v5p-nap.yaml): Ephemeral Ray cluster manifest (`Service` + Kueue-managed `JobSet`) for **GKE Node Auto-Provisioning (NAP) TPU v5p** clusters (`bodaborg-v5p-nap` in `europe-west4`, project `cloud-tpu-shared-capacity`) where the KubeRay operator CRD is not installed. Rendered and managed end-to-end by [`examples/tpu/grpo/run_grpo_v5p_nap.sh`](../grpo/run_grpo_v5p_nap.sh).
- [`ray-tpu-v6e8-2slice.yaml`](ray-tpu-v6e8-2slice.yaml): KubeRay `RayCluster` manifest provisioning 1 CPU Ray head pod (`ray-head`) and 2 multi-host TPU v6e-8 slices (`numOfHosts: 2`, `google.com/tpu: 4` per host = 16 TPU v6e chips total) with GCS Fuse mounted at `/data` and `400Gi` host memory per TPU pod.
- [`ray-tpu-v6e32-2slice.yaml`](ray-tpu-v6e32-2slice.yaml): 2 x TPU v6e-32 slices (`4x8`, `numOfHosts: 8` = 64 chips) as one worker group (`replicas: 2`), RayCluster `ray-tpu-v6e32-cluster`, image `verl-tpu:v20261006-tsync1006`; used by [`run_qwen3_32b_torchtitan.sh`](../grpo/run_qwen3_32b_torchtitan.sh) (Qwen3-32B, FSDP32 trainer + vLLM TP32 rollout).
- [`ray-tpu-v7x-2slice.yaml`](ray-tpu-v7x-2slice.yaml) / [`ray-tpu-v7x-2x2x4-2slice.yaml`](ray-tpu-v7x-2x2x4-2slice.yaml): KubeRay `RayCluster` manifests for TPU 7x (Ironwood) single-host (`2x2x1`) and multi-host (`2x2x4`) slices.

---

## 🛠️ Deploying on GKE

### Option 1: Ephemeral Kueue `JobSet` on GKE NAP TPU v5p (`bodaborg-v5p-nap`)

On shared GKE NAP clusters managed by **Kueue + JobSet** (`bodaborg-v5p-nap`), use [`examples/tpu/grpo/run_grpo_v5p_nap.sh`](../grpo/run_grpo_v5p_nap.sh) to provision an isolated Ray `JobSet` + `Service`, run the GRPO job, pull logs to local disk, and automatically delete only your `JobSet` and `Service` on exit:

```bash
# Fast 5-step Qwen3-0.6B GRPO smoke test on bodaborg-v5p-nap (2 x 2x2x1 v5p slices = 8 chips):
SMOKE_TEST=1 bash examples/tpu/grpo/run_grpo_v5p_nap.sh

# Full 100-step Qwen3-0.6B GSM8K GRPO run:
bash examples/tpu/grpo/run_grpo_v5p_nap.sh

# Heterogeneous slices: Qwen3-4B with 1 x 8-chip Trainer slice (2x2x2, 2 hosts)
# + 2 separate 4-chip Sampler slices (2x2x1, 1 host each, 2 x TP=4 vLLM replicas):
SMOKE_TEST=1 \
JOB_SCRIPT=examples/tpu/grpo/run_qwen3_4b_torchtitan.sh \
TRAINER_HOSTS_PER_SLICE=2 \
TRAINER_NODE_TOPOLOGY=2x2x2 \
ROLLOUT_SLICE_REPLICAS=2 \
ROLLOUT_HOSTS_PER_SLICE=1 \
ROLLOUT_NODE_TOPOLOGY=2x2x1 \
ROLLOUT_TP=4 \
bash examples/tpu/grpo/run_grpo_v5p_nap.sh
```

See [`examples/tpu/grpo/README.md`](../grpo/README.md) for full details and step-by-step manual commands.

### Option 2: Long-Lived KubeRay `RayCluster` (TPU v6e / 7x)

```bash
# Apply the KubeRay cluster manifest
kubectl apply -f examples/tpu/gke/ray-tpu-v6e8-2slice.yaml

# Wait for 1 head pod + 4 TPU worker pods (2 slices x 2 hosts) to reach 2/2 Running
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster -w

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

- **Docker image**: pinned in [`tests/special_tpu/gke/tpu-ci-image.env`](../../../tests/special_tpu/gke/tpu-ci-image.env) (`TPU_CI_IMAGE`, built from [`Dockerfile.tpu`](Dockerfile.tpu)) and rendered into [`tests/special_tpu/gke/raycluster-ci.yaml`](../../../tests/special_tpu/gke/raycluster-ci.yaml) by `run_tpu_e2e_ci.sh`.
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

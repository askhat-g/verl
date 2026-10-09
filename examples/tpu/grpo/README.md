# GRPO RL Training on Google Cloud TPU (TPU v5p, v6e & 7x)

This directory contains examples and scripts for running **GRPO (Group Relative Policy Optimization) RL Training** on Google Cloud TPU **v5p**, **v6e**, and **7x (Ironwood, `2x2x1` single-host topology)** instances using `verl`.

The training setup uses:
- **Actor Engine**: TorchTitan (`model_engine=torchtitan`)
- **Rollout Engine**: vLLM (`actor_rollout_ref.rollout.name=vllm`)
- **Several rollout replicas**: when the rollout pool holds more than one vLLM replica (`rollout.tensor_model_parallel_size` smaller than the number of rollout devices; with `run_qwen3_0_6b_torchtitan.sh` set `ROLLOUT_TP=<chips per host>` to get one replica per rollout host), Raiden syncs each replica in its own transfer and the parity check compares each replica with the trainer separately. Look for `[RAIDEN PARITY] step N: checking rollout replica K` followed by `PARITY VERIFIED` once per replica in the driver log.
- **Placement Strategy**: Non-colocated multi-slice execution (Slice 0 for Trainer/Actor, Slice 1 for Rollout)
- **TPU v5p (`2x2x1` Single-Host & Multi-Host Topologies)**: Each `2x2x1` host provides 4 physical chips = 4 logical Megacore devices (95 GB HBM per chip, 3D mesh topology `2,2,1` via `TPU_V5P_TOPOLOGY_MAP` when `TPU_ACCELERATOR_TYPE=v5p`). Multi-host v5p slices (`2x2x2` = 8 chips, `2x2x4` = 16 chips, `2x4x4` = 32 chips, `4x4x4` = 64 chips) are also mapped in `TPU_V5P_TOPOLOGY_MAP`.
- **TPU 7x (`2x2x1` Single-Host Topology)**: Each `2x2x1` host (`numOfHosts: 1`) provides 4 physical dual-core chips = 8 addressable TensorCore devices (`NNODES=1`, `N_CHIPS=8`, 4D mesh topology `2,2,1,2`).
- **Multi-host trainer slices on TPU 7x**: the trainer slice can also be `2x2x2` (2 hosts, 16 devices), `2x2x4` (4 hosts, 32 devices), `2x4x4` (8 hosts, 64 devices) or `4x4x4` (16 hosts, 128 devices). The 4D topology is looked up by device count in `TPU_V7X_TOPOLOGY_MAP` (`verl/plugin/platform/platform_tpu.py`); a shape missing from that table logs a warning and falls back to the single-device topology, which does not work for a multi-host trainer.

---

## ⚡ Option A: Ephemeral Ray Cluster on GKE NAP TPU v5p (`bodaborg-v5p-nap`)

On shared GKE Node Auto-Provisioning (NAP) clusters managed by **Kueue + JobSet** (such as `bodaborg-v5p-nap` in `europe-west4`, project `cloud-tpu-shared-capacity`), where the KubeRay operator is not installed, [`run_grpo_v5p_nap.sh`](run_grpo_v5p_nap.sh) automates the entire ephemeral Ray cluster lifecycle (modeled after [`tests/special_tpu/run_tpu_e2e_ci.sh`](../../../tests/special_tpu/run_tpu_e2e_ci.sh)):

1. **Spawns a uniquely named Ray cluster** (`JobSet` + Head `Service` from [`examples/tpu/gke/ray-jobset-v5p-nap.yaml`](../gke/ray-jobset-v5p-nap.yaml)) queued via Kueue (`multislice-queue`) with 1 CPU head pod (`cpu-np`) + 2 single-host `2x2x1` TPU v5p slices (`tpu-group-0` for Trainer, `tpu-group-1` for vLLM Rollout; 8 TPU v5p chips total).
2. **Waits for Kueue admission, GKE NAP node provisioning, and GCSFuse sidecar readiness**, then verifies all 8 TPU chips register with Ray.
3. **Syncs the local `verl` repository to the Ray head pod and submits the Qwen3-0.6B GRPO job** ([`run_qwen3_0_6b_torchtitan.sh`](run_qwen3_0_6b_torchtitan.sh)).
4. **Polls job status, pulls the complete Ray job log to local disk** (`/tmp/verl_nap_logs/<cluster_name>-grpo.log`), and verifies training rewards, finite gradients, and validation metrics via [`tests/special_tpu/verify_tpu_e2e_log.py`](../../../tests/special_tpu/verify_tpu_e2e_log.py).
5. **Tears down ONLY the `JobSet` and `Service` created by that run** (`trap cleanup EXIT`), releasing Kueue quota and GKE NAP TPU nodes without disturbing any other jobs on the cluster.

### 1. One-Command Automated Run

```bash
# Fast 5-step Qwen3-0.6B smoke test (1 x 4-chip Trainer slice + 1 x 4-chip Sampler slice = 8 chips):
SMOKE_TEST=1 bash examples/tpu/grpo/run_grpo_v5p_nap.sh

# Full 100-step Qwen3-0.6B GSM8K GRPO training run:
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

Useful environment variable overrides for `run_grpo_v5p_nap.sh`:
- `JOB_SCRIPT=examples/tpu/grpo/run_qwen3_4b_torchtitan.sh`: Switch the training recipe (e.g., `Qwen3-0.6B`, `Qwen3-4B`, `Qwen3-8B`).
- `TRAINER_SLICE_REPLICAS=1 TRAINER_HOSTS_PER_SLICE=2 TRAINER_NODE_TOPOLOGY=2x2x2`: Configure the Trainer TPU slice topology (`1` host = `2x2x1` = 4 chips; `2` hosts = `2x2x2` = 8 chips; `4` hosts = `2x2x4` = 16 chips).
- `ROLLOUT_SLICE_REPLICAS=2 ROLLOUT_HOSTS_PER_SLICE=1 ROLLOUT_NODE_TOPOLOGY=2x2x1 ROLLOUT_TP=4`: Provision multiple independent Sampler slices (`tpu-group-1`, `tpu-group-2`, ...) and run one `TP=4` vLLM replica per slice, synced concurrently via Raiden P2P.
- `LOG_FILE=/path/to/local.log`: Custom local path for the pulled Ray job log (default: `/tmp/verl_nap_logs/<cluster_name>-grpo.log`).
- `KEEP_CLUSTER=1`: Skip automatic teardown on exit so you can inspect or submit additional jobs to the Ray cluster.

### 2. Step-by-Step Manual Workflow on `bodaborg-v5p-nap`

If you prefer to run each stage manually (spawn Ray cluster -> submit job -> pull logs -> delete Ray cluster):

```bash
# 1. Connect to bodaborg-v5p-nap using an isolated kubeconfig
export KUBECONFIG="/tmp/kubeconfig-bodaborg-v5p-nap"
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --region europe-west4 \
  --project cloud-tpu-shared-capacity

# 2. Render and apply a uniquely named Ray JobSet + Head Service (<= 32 chars)
export RAY_CLUSTER_NAME="verl-nap-${USER:0:8}-$(date +%H%M)"
export RAY_NAMESPACE="default"
export KUEUE_QUEUE="multislice-queue"
export PRIORITY_CLASS="medium"
export SERVICE_ACCOUNT="default"
export HEAD_NODEPOOL="cpu-np"
export TPU_NODE_SELECTOR_ACCEL="tpu-v5p-slice"
export TRAINER_SLICE_REPLICAS="1" TRAINER_HOSTS_PER_SLICE="1" TRAINER_NODE_TOPOLOGY="2x2x1"
export ROLLOUT_SLICE_REPLICAS="1" ROLLOUT_HOSTS_PER_SLICE="1" ROLLOUT_NODE_TOPOLOGY="2x2x1"
export PREWARM_MODEL_PATH="/data/jialei/assets/hf/Qwen3-0.6B"
export RESERVATION_NAME="cloudtpu-20260902214500-1810493672"
export GCS_BUCKET="torchprime"
export IMAGE="us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20261006-tsync1006"
export HEAD_CPU="12" HEAD_MEMORY="48Gi"
export WORKER_CPU="48" WORKER_MEMORY="400Gi" WORKER_TPU_CHIPS="4"
export TPU_ACCELERATOR_TYPE="v5p"
export RUN_ID="manual" RUN_OWNER="${USER}"

envsubst < examples/tpu/gke/ray-jobset-v5p-nap.yaml | kubectl apply -f -

# 3. Wait for all 3 pods (1 head + 2 TPU workers) to reach Running & Ready
kubectl get pods -l "verl-nap/cluster=${RAY_CLUSTER_NAME}" -w
HEAD_POD=$(kubectl get pods -l "verl-nap/cluster=${RAY_CLUSTER_NAME},verl-nap/role=head" -o jsonpath='{.items[0].metadata.name}')

# 4. Port-forward the Ray Dashboard / Jobs API (bodaborg-v5p-nap grants pods/portforward, not pods/exec)
kubectl port-forward "pod/${HEAD_POD}" 28265:8265 >/dev/null 2>&1 &
PF_PID=$!

# Verify Ray has registered 8.0 TPU across tpu-group-0 and tpu-group-1
curl -s "http://127.0.0.1:28265/api/cluster_status?format=text"

# 5. Submit Qwen3-0.6B GRPO from your local repo
ray job submit \
  --address "http://127.0.0.1:28265" \
  --working-dir . \
  --runtime-env-json '{
    "excludes": [".git", "tests", "docs", ".github"],
    "env_vars": {
      "TPU_ACCELERATOR_TYPE": "v5p",
      "OMP_NUM_THREADS": "4",
      "SMOKE_TEST": "1",
      "NNODES_TRAINER": "1",
      "N_CHIPS_TRAINER": "4",
      "NNODES_ROLLOUT": "1",
      "N_CHIPS_ROLLOUT": "4",
      "MAX_RESPONSE_LEN": "1024",
      "ROLLOUT_N": "4",
      "TEST_FREQ": "5"
    }
  }' \
  --no-wait \
  -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh

# 6. Check status & pull logs to local disk (replace <JOB_ID> with raysubmit_... ID)
ray job status --address "http://127.0.0.1:28265" <JOB_ID>
ray job logs --address "http://127.0.0.1:28265" <JOB_ID> > /tmp/grpo_v5p_nap.log

# 7. Stop port-forward and tear down ONLY your JobSet and Service to release Kueue quota and NAP TPU nodes
kill "${PF_PID}" 2>/dev/null || true
kubectl delete jobset "${RAY_CLUSTER_NAME}" service "${RAY_CLUSTER_NAME}-head-svc" -n "${RAY_NAMESPACE}"
```

### 3. Key Architecture Notes for `bodaborg-v5p-nap`
- **Kueue + JobSet (No KubeRay CRD)**: `bodaborg-v5p-nap` uses Kueue (`multislice-queue`) and `jobset.x-k8s.io/v1alpha2` rather than the KubeRay operator. [`examples/tpu/gke/ray-jobset-v5p-nap.yaml`](../gke/ray-jobset-v5p-nap.yaml) provisions a `JobSet` with 1 CPU `head` job (`cpu-np`) and 2 `tpu-group` slices (`2x2x1`, 4 TPU v5p chips each), plus a ClusterIP `Service` for the Ray head.
- **Per-Slice libtpu & MegaScale Isolation**: GKE's TPU `JobSet` webhook injects `MEGASCALE_NUM_SLICES=3` across all `replicatedJobs` (`head` + 2 `tpu-group` slices) and uses ALTS credentials by default. The worker startup script strips `MEGASCALE_*` so Trainer (`tpu-group-0`) and vLLM Rollout (`tpu-group-1`) form independent libtpu meshes, and `PlatformTPU` sets `--slicebuilder_use_insecure_grpc=true --undefok=slicebuilder_use_insecure_grpc` in `LIBTPU_INIT_ARGS`.
- **GCSFuse on NAP TPU Nodes Only**: Because the static `cpu-np` node pool on `bodaborg-v5p-nap` does not have GKE Workload Identity enabled, `/data` (`gs://torchprime`) is mounted only on the NAP-provisioned TPU worker pods (`--num-cpus=0` on `ray-head` schedules `TaskRunner` and all CPU/TPU actors onto the TPU worker pods). To avoid transatlantic GCSFuse random-seek thrashing (`europe-west4` to US multi-region `gs://torchprime`) when 4 ranks `mmap` `model.safetensors`, each worker pre-warms `model.safetensors` sequentially into the Linux kernel page cache before `ray start` and sets `OMP_NUM_THREADS=4`.
- **64 GiB `/dev/shm`**: `ray-jobset-v5p-nap.yaml` mounts a `64Gi` memory-backed `emptyDir` at `/dev/shm` so `torch_tpu`'s compilation cache (`/dev/shm/torch_tpu_cache`) and PyTorch `DataLoader` POSIX semaphores do not exhaust Kubernetes' default 64 MiB `/dev/shm`.

---

## 🚀 Option B: Long-Lived KubeRay Cluster (TPU 7x & v6e)

### 1. Prerequisites
Ensure you have a running Ray cluster on TPU 7x (`examples/tpu/gke/ray-tpu-v7x-2slice.yaml`) or TPU v6e (`examples/tpu/gke/ray-tpu-v6e8-2slice.yaml`) nodes with `verl` installed across all head and worker nodes.

Environment variables required:
- `MODEL_PATH`: Path to HuggingFace model checkpoint (e.g. `/data/jialei/assets/hf/Qwen3-0.6B`)
- `TRAIN_FILE` & `TEST_FILE`: Parquet dataset files (e.g. GSM8K dataset)
- `WANDB_API_KEY` (Optional): For experiment tracking on Weights & Biases

---

### 2. Ensure Clean Cluster and Set Up Port Forwarding (Optional)

Before submitting a new job, you can reset TPU cluster state by deleting all cluster pods (or worker pods), which will be automatically recreated by the KubeRay operator:

```bash
# Delete all pods to reset head and worker nodes (TPU 7x cluster)
kubectl delete pod -l ray.io/cluster=ray-tpu-v7x-cluster

# Port forward Ray head dashboard service to local port 23333 in the background
kubectl port-forward svc/ray-tpu-v7x-cluster-head-svc 23333:8265 > /dev/null 2>&1 &
```

---

### 3. Submit a GRPO Training Job

You can submit the training job to your Ray cluster using the Ray CLI (`ray job submit`):

```bash
# Set active Ray cluster address (e.g. localhost:23333 if port-forwarded)
export RAY_ADDRESS="http://localhost:23333"

# Submit GRPO RL training job
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json '{
    "excludes": [".git", "logs", "*.log", "*.pt", "*.bin"],
    "env_vars": {
      "PYTHONPATH": ".",
      "PYTHONUNBUFFERED": "1",
      "VERL_PLATFORM": "tpu",
      "TPU_ACCELERATOR_TYPE": "tpu7x",
      "VLLM_USE_V1": "0",
      "RAY_memory_monitor_refresh_ms": "0",
      "RAY_memory_usage_threshold": "0.99",
      "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
      "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"
    }
  }' \
  -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
```

The script defaults to a 100-step GSM8K run. For a quick bring-up check that only
validates that the stack comes up, prepend `SMOKE_TEST=1`:

```bash
  -- bash -c 'SMOKE_TEST=1 bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh'
```

---

## 📊 Monitoring Progress

### Check Job Status
```bash
ray job status --address http://localhost:23333 <JOB_ID>
```

### Stream Live Logs
```bash
ray job logs --follow --address http://localhost:23333 <JOB_ID>
```

### Verify Raiden Weight Sync
Both checks are selected by a single engine kwarg (code default `off`):
`+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.verify_parity=<off|norm|exact|all>`
(`True` means `norm`, `False` means `off`). This is the only place the setting is read: the trainer's
`RaidenCheckpointEngine` and the orchestrator both see `engine_kwargs.raiden`, so a top-level
`engine_kwargs.verify_parity` is ignored. Both checks are implemented by `RaidenParityCheck` in
[`raiden_checkpoint_engine.py`](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py).
`run_qwen3_0_6b_torchtitan.sh` sets it to `True`.

| Check | Value | When it runs | What it catches |
|-------|-------|--------------|-----------------|
| Exact parity | `exact` or `all` | Step-0 sync of a fresh run (trainer weights equal the checkpoint vLLM loaded) | Any wrong byte: misplaced shard slices, wrong tiling, bad fusion/transpose |
| Norm parity | `norm` (`True`) or `all` | Every sync from step 1 | Lost or scaled weights (L1/L2 within `1e-3`); cannot see reordered bytes |

Run the exact check after changing the trainer/rollout sharding or upgrading `tpu-sync` / `torch-tpu`. The
orchestrator (TaskRunner) logs one summary for all sampler ranks, e.g.
`[RAIDEN PARITY EXACT | Step 0] 32 sampler ranks, [515] tensors checked per rank, 0 mismatched, 0 unresolved`,
followed by one line per mismatched tensor and rank. It costs about 0.4 s once per rank.

The norm check compares the trainer's global L1/L2/numel against the sum over sampler ranks. Each
rank reports a `replicas` count per tensor (how many TP ranks hold the same slice, e.g. GQA k/v heads
when TP size > number of KV heads), and the orchestrator divides by it so replicated slices are counted once.
A passing step logs `[RAIDEN PARITY VERIFIED | Step N] 100% DISTRIBUTED NORM PARITY CONFIRMED!` with the
global L1/L2 values and deltas. A failing step logs `[RAIDEN PARITY MISMATCH | Step N]` with the first 10
mismatched tensors.

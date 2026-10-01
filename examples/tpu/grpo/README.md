# GRPO RL Training on Google Cloud TPU (TPU 7x & v6e)

This directory contains examples and scripts for running **GRPO (Group Relative Policy Optimization) RL Training** on Google Cloud TPU **7x (Ironwood, `2x2x1` single-host topology)** and **v6e** instances using `verl`.

The training setup uses:
- **Actor Engine**: TorchTitan (`model_engine=torchtitan`)
- **Rollout Engine**: vLLM (`actor_rollout_ref.rollout.name=vllm`)
- **Placement Strategy**: Non-colocated multi-slice execution (Slice 0 for Trainer/Actor, Slice 1 for Rollout)
- **TPU 7x (`2x2x1` Single-Host Topology)**: Each `2x2x1` host (`numOfHosts: 1`) provides 4 physical dual-core chips = 8 addressable TensorCore devices (`NNODES=1`, `N_CHIPS=8`, 4D mesh topology `2,2,1,2`).

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have a running Ray cluster on TPU 7x (`examples/tpu/gke/ray-tpu-v7x-2slice.yaml`) or TPU v6e (`examples/tpu/gke/ray-tpu-v6e8-2slice.yaml`) nodes with `verl` installed across all head and worker nodes.

First, deploy the KubeRay `RayCluster` matching your target model size from [`examples/tpu/gke/`](../gke/README.md):

| Model Size | GRPO Launch Script | GCS Fuse Cluster Manifest | Self-Contained (HostPath) Cluster Manifest | Topology (Trainer + Rollout) |
|------------|--------------------|---------------------------|--------------------------------------------|------------------------------|
| Qwen3-0.6B | [`run_qwen3_0_6b_torchtitan.sh`](run_qwen3_0_6b_torchtitan.sh) | [`ray-tpu-v6e8-2slice.yaml`](../gke/ray-tpu-v6e8-2slice.yaml) | [`model_0_6b_config.yaml`](../gke/model_0_6b_config.yaml) | `v6e-8` (8 chips) + `v6e-8` (8 chips) |
| Qwen3-4B | [`run_qwen3_4b_torchtitan.sh`](run_qwen3_4b_torchtitan.sh) | [`ray-tpu-v6e8-2slice.yaml`](../gke/ray-tpu-v6e8-2slice.yaml) | [`model_4b_config.yaml`](../gke/model_4b_config.yaml) | `v6e-8` (8 chips) + `v6e-8` (8 chips) |
| Qwen3-8B | [`run_qwen3_8b_torchtitan.sh`](run_qwen3_8b_torchtitan.sh) | [`ray-tpu-v6e8-2slice.yaml`](../gke/ray-tpu-v6e8-2slice.yaml) | [`model_8b_config.yaml`](../gke/model_8b_config.yaml) | `v6e-8` (8 chips) + `v6e-8` (8 chips) |
| Qwen3-14B | [`run_qwen3_14b_torchtitan.sh`](run_qwen3_14b_torchtitan.sh) | [`ray-tpu-v6e8-2slice.yaml`](../gke/ray-tpu-v6e8-2slice.yaml) | [`model_14b_config.yaml`](../gke/model_14b_config.yaml) | `v6e-8` (8 chips) + `v6e-8` (8 chips) |
| Qwen3-32B | [`run_qwen3_32b_torchtitan.sh`](run_qwen3_32b_torchtitan.sh) | [`ray-tpu-v6e16-v6e8-2slice.yaml`](../gke/ray-tpu-v6e16-v6e8-2slice.yaml) | [`model_32b_config.yaml`](../gke/model_32b_config.yaml) | `v6e-16` (16 chips) + `v6e-8` (8 chips) |

```bash
# Deploy (or recreate) the RayCluster for your target model size:
kubectl delete raycluster ray-tpu-v6e-cluster --ignore-not-found --wait=true
kubectl apply -f examples/tpu/gke/model_32b_config.yaml   # or model_{0_6b,4b,8b,14b}_config.yaml

# Wait for head + TPU worker pods to reach Running
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster -w
```

> [!NOTE]
> The `run_qwen3_{8b,14b,32b}_torchtitan.sh` scripts automatically invoke [`prepare_tpu_env.py`](prepare_tpu_env.py) across all cluster nodes before training starts to ensure the GSM8K parquet dataset and target HuggingFace checkpoint exist under `/data/jialei` (and evict previous model checkpoints when using node-local storage).

---

### 2. Ensure Clean Cluster and Set Up Port Forwarding (Optional)

Before submitting a new job on an already-running cluster, you can reset TPU state by deleting the cluster pods (which KubeRay automatically recreates) and port-forwarding the Ray dashboard:

```bash
# Delete all pods to reset head and worker nodes (TPU 7x cluster)
kubectl delete pod -l ray.io/cluster=ray-tpu-v7x-cluster

# Port forward Ray head dashboard service to local port 23333 in the background
kubectl port-forward svc/ray-tpu-v7x-cluster-head-svc 23333:8265 > /dev/null 2>&1 &
```

---

### 3. Submit a GRPO Training Job

You can submit any of the Qwen3 (`0.6B`, `4B`, `8B`, `14B`, or `32B`) training jobs to your Ray cluster using `ray job submit`:

```bash
# Set active Ray cluster address (e.g. localhost:23333 if port-forwarded)
export RAY_ADDRESS="http://localhost:23333"

# Submit GRPO RL training job (replace script path with 0_6b, 4b, 8b, 14b, or 32b)
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
  -- bash examples/tpu/grpo/run_qwen3_32b_torchtitan.sh
```

The script defaults to a 100-step GSM8K run. For a quick 5-step bring-up check that validates that the stack comes up, prepend `SMOKE_TEST=1`:

```bash
  -- bash -c 'SMOKE_TEST=1 bash examples/tpu/grpo/run_qwen3_32b_torchtitan.sh'
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

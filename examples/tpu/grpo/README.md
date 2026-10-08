# GRPO RL Training on Google Cloud TPU (TPU 7x & v6e)

This directory contains examples and scripts for running **GRPO (Group Relative Policy Optimization) RL Training** on Google Cloud TPU **7x (Ironwood, `2x2x1` single-host topology)** and **v6e** instances using `verl`.

The training setup uses:
- **Actor Engine**: TorchTitan (`model_engine=torchtitan`)
- **Rollout Engine**: vLLM (`actor_rollout_ref.rollout.name=vllm`)
- **Several rollout replicas**: when the rollout pool holds more than one vLLM replica (`rollout.tensor_model_parallel_size` smaller than the number of rollout devices; with `run_qwen3_0_6b_torchtitan.sh` set `ROLLOUT_TP=<chips per host>` to get one replica per rollout host), Raiden syncs each replica in its own transfer and the parity check compares each replica with the trainer separately. Look for `[RAIDEN PARITY] step N: checking rollout replica K` followed by `PARITY VERIFIED` once per replica in the driver log.
- **Placement Strategy**: Non-colocated multi-slice execution (Slice 0 for Trainer/Actor, Slice 1 for Rollout)
- **TPU 7x (`2x2x1` Single-Host Topology)**: Each `2x2x1` host (`numOfHosts: 1`) provides 4 physical dual-core chips = 8 addressable TensorCore devices (`NNODES=1`, `N_CHIPS=8`, 4D mesh topology `2,2,1,2`).
- **Multi-host trainer slices on TPU 7x**: the trainer slice can also be `2x2x2` (2 hosts, 16 devices), `2x2x4` (4 hosts, 32 devices), `2x4x4` (8 hosts, 64 devices) or `4x4x4` (16 hosts, 128 devices). The 4D topology is looked up by device count in `TPU_V7X_TOPOLOGY_MAP` (`verl/plugin/platform/platform_tpu.py`); a shape missing from that table logs a warning and falls back to the single-device topology, which does not work for a multi-host trainer.

---

## 🚀 Quick Start

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

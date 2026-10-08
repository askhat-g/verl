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

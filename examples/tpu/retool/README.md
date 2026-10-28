# ReTool on TPU (`Qwen3-30B-A3B` MoE)

Self-contained TPU recipe for **ReTool** ([arXiv:2504.11536](https://arxiv.org/abs/2504.11536)) with `Qwen/Qwen3-30B-A3B` (`30.5B` total parameters, `3.3B` active per token, `128` routed experts, `top_k=8`) using the **TorchTitan** trainer (`SimpleFSDP=8`, Ulysses `CP=4`, `EP=1`, `TPU_EAGER_MODE=DEFER_NEVER`, `SplashAttention`, MegaBlox Pallas `grouped_mm`) and **vLLM** multi-turn async rollout (`TP=32, EP=32`) with **Raiden** direct P2P weight synchronization.

## Hardware Topology (`2 x TPU v6e-32` Slices)

Uses `examples/tpu/gke/ray-tpu-v6e32-2slice.yaml` (`2` slices of `8 hosts x 4 chips = 32 chips` each):

| Role | Default GKE Slice (`ray-tpu-v6e32-2slice.yaml`) | Parallelism | Memory & Throughput Notes |
| :--- | :--- | :--- | :--- |
| **Trainer** (`32` chips) | `tpu-group-0` (`8 hosts x 4 chips`, `4x8`) | `SimpleFSDP=8`, `CP=4`, `TP=1`, `EP=1` | `TPU_EAGER_MODE=DEFER_NEVER`, `VERL_TPU_SEQ_BUCKET_SIZE=2048` (`512`-token `SplashAttention` blocks, `256`-row-aligned MegaBlox tiles). |
| **Rollout** (`32` chips) | `tpu-group-1` (`8 hosts x 4 chips`, `4x8`) | `1` server of `TP=32`, `EP=32` | `ROLLOUT_EP == ROLLOUT_TP` (`4` whole experts/chip), `gpu_memory_utilization=0.62`. |
| **Code Sandbox** | Local CPU on worker pods (`127.0.0.1:18080`) | `4` `uvicorn` workers / pod | Loopback-only FastAPI Python sandbox (`8s` default timeout, `1 GiB` `RLIMIT_AS`, process-group `SIGKILL` cleanup). |

On `2 x TPU 7x 2x2x4` slices (`examples/tpu/gke/ray-tpu-v7x16-2slice.yaml`), pass `TPU_ACCELERATOR_TYPE=tpu7x NNODES_TRAINER=4 N_CHIPS_TRAINER=8 NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8`.

## End-to-End Workflow

### 1. Datasets and Base Model Layout

Expected dataset and model layout under `RAY_DATA_HOME` / `DATA_ROOT` (default `/data/jialei` GCS Fuse mount):
- Base model: `${RAY_DATA_HOME}/assets/hf/Qwen3-30B-A3B` (`Qwen/Qwen3-30B-A3B`)
- Stage 1 SFT dataset: `${RAY_DATA_HOME}/dataset/ReTool-SFT/data/train-00000-of-00001.parquet` (`Russwest0/ReTool-SFT`)
- Stage 2 RL datasets:
  - `${DATA_ROOT}/dataset/BytedTsinghua-SIA/DAPO-Math-17k`
  - `${DATA_ROOT}/dataset/Maxwell-Jia/AIME_2024`
  - `${DATA_ROOT}/dataset/yentinglin/aime_2025`

Both scripts automatically stage HuggingFace checkpoints from `RAY_DATA_HOME` to host `/tmp/models/...` across all TPU hosts via `examples/tpu/grpo/stage_model_local.py` (`STAGE_MODEL=1` by default).

### 2. Stage 1: Multi-Turn Cold-Start SFT (`372` Steps on `32` TPU Chips)

Trains `Qwen3-30B-A3B` for `6` epochs (`372` steps at `train_batch_size=32`, `max_length=16384`) and exports streaming HuggingFace `.safetensors` shards (`checkpoint.save_contents=['hf_model']`) to `${RAY_DATA_HOME}/checkpoint/multiturn-sft-qwen3-30b-a3b-tpu/global_step_372/huggingface`:

```bash
bash examples/tpu/retool/run_qwen3_30b_a3b_sft_torchtitan.sh
```

### 3. Stage 2: Multi-Turn DAPO RL with Code Interpreter (`64` TPU Chips)

Launch the default **`512`-rollouts/step (`8192` response length, `8` max turns, `400` steps)** schedule (which automatically stages the Stage 1 SFT checkpoint to `/tmp/models/multiturn-sft-qwen3-30b-a3b-tpu` across all TPU nodes and auto-starts the 4-worker loopback `sandbox_fusion_server.py` daemon on `127.0.0.1:18080` inside the `ExecutionWorker` pod):

```bash
bash examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh
```

To run the **`16384` response / `16`-turn `1,024`-rollouts/step** variant:

```bash
MAX_RESPONSE_LENGTH=16384 MAX_TURNS=16 N_RESP_PER_PROMPT=16 ACTOR_LR=2e-6 CLIP_GRAD=0.0625 TOTAL_TRAINING_STEPS=150 \
    bash examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh
```

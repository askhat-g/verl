# SFT (Supervised Fine-Tuning) on Google Cloud TPU (v6e)

This directory contains examples and scripts for running **Multi-Turn / Instruction SFT (Supervised Fine-Tuning)** on Google Cloud TPU v6e using `verl`, `verl-hardware-plugin`, and the **TorchTitan** engine (`engine=torchtitan`) via `verl.trainer.sft_trainer` (`torchrun`).

The training setup uses:
- **Entrypoint**: `torchrun ... -m verl.trainer.sft_trainer`
- **Training Engine**: TorchTitan (`engine=torchtitan`, PyTorch FSDP2 on `torch_tpu`)
- **Sequence Packing & Bucketing**: `model.use_remove_padding=True` with `data.pad_mode=no_padding`, `engine.pad_to_length=True`, and `engine.pad_to_length_bucket=256` to pad packed sequences to static 256-token buckets and prevent XLA HLO recompilation
- **Parallelism**: Pure FSDP2 (`engine.tensor_parallel_size=1`, `engine.data_parallel_shard_size=<total_chips>`)

---

## 🚀 Quick Start on TPU

### 1. Prepare Model Checkpoint & Dataset

Ensure the model checkpoint and preprocessed GSM8K-SFT parquet files are available under `/data` (or override `DATA_HOME` / `MODEL_PATH` / `TRAIN_FILE` / `TEST_FILE`):
- `MODEL_PATH`: `/data/assets/hf/Qwen3-0.6B`
- `TRAIN_FILE`: `/data/data/gsm8k_sft/train.parquet`
- `TEST_FILE`: `/data/data/gsm8k_sft/test.parquet`

```bash
# Preprocess GSM8K multi-turn SFT parquet files
python3 examples/data_preprocess/gsm8k_multiturn_sft.py --local_save_dir /data/data/gsm8k_sft
```

---

### 2. Launch SFT Training (`verl.trainer.sft_trainer`)

#### Single-Host TPU v6e-4 (1 host $\times$ 4 TPU chips)

```bash
NNODES=1 NPROC_PER_NODE=4 bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
```

For a quick 8-step smoke test with validation at Step 4 and Step 8, pass `SMOKE_TEST=1`:

```bash
SMOKE_TEST=1 NNODES=1 NPROC_PER_NODE=4 bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
```

#### Multi-Host TPU v6e-8 Slice (2 hosts $\times$ 4 TPU chips = 8 TPU chips)

On a 2-host TPU v6e-8 slice, set `TPU_WORKER_HOSTNAMES="<host0_ip>,<host1_ip>"` and launch on each host with `NODE_RANK=0` and `NODE_RANK=1`:

```bash
# Host 0
TPU_WORKER_HOSTNAMES="10.0.0.1,10.0.0.2" MASTER_ADDR="10.0.0.1" \
  NNODES=2 NPROC_PER_NODE=4 NODE_RANK=0 \
  bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh

# Host 1
TPU_WORKER_HOSTNAMES="10.0.0.1,10.0.0.2" MASTER_ADDR="10.0.0.1" \
  NNODES=2 NPROC_PER_NODE=4 NODE_RANK=1 \
  bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
```

---

## ⚙️ Key Configuration & Tuning Notes

1. **TPU Slice Topology (`NNODES` & `NPROC_PER_NODE`)**:
   - On a single-host **TPU v6e-4** slice (`2x2` topology = 1 VM host $\times$ 4 TPU chips), `PlatformTPU` automatically configures `PJRT_DEVICE`, `TORCH_TPU_SLICEBUILDER_ADDRESSES`, `TPU_PROCESS_PORT`, `TPU_VISIBLE_CHIPS`, and topology bounds from `torchrun`'s `LOCAL_RANK` / `RANK` / `WORLD_SIZE`.
   - On a multi-host **TPU v6e-8** slice (`2x4` topology = 2 VM hosts $\times$ 4 TPU chips/host), `NNODES=2` and `TPU_WORKER_HOSTNAMES` allow `libtpu`'s multi-host slice builder to initialize all 8 chips in the slice together.
2. **Pure FSDP2 (`engine.tensor_parallel_size=1`)**:
   - Keep `engine.tensor_parallel_size=1` and shard across all TPU chips using `engine.data_parallel_shard_size=<total_chips>`. Tensor parallelism (`tensor_parallel_size > 1`) on `torch_tpu` routes logits through `DTensor.full_tensor()`, whose backward pass produces non-finite (`NaN`/`Inf`) gradients.
3. **Sequence Bucketing (`engine.pad_to_length=True`, `engine.pad_to_length_bucket=256`)**:
   - Packed 1D sequences are padded to multiples of `engine.pad_to_length_bucket` (default `256`, with `VERL_TPU_SEQ_BUCKET_SIZE` fallback) and aligned across data-parallel ranks so XLA compiles a bounded set of static bucket shapes (`256, 512, ..., 2048`) and reuses them with 100% cache hit rate.
4. **Qwen3 Chat Template (`data.ignore_input_ids_mismatch=True`)**:
   - Qwen3's chat template injects `<think>\n\n</think>\n\n` only on the final assistant turn and strips `<think>` blocks from earlier assistant turns, so full-conversation tokenization legitimately differs from per-turn concatenation in `MultiTurnSFTDataset`.

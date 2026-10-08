#!/usr/bin/env bash
# SFT | Qwen3-0.6B | GSM8K-SFT | TorchTitan Engine | TPU v6e (torchrun SPMD)
#
# Hardware Setup:
#   Default: 1 Slice of TPU v6e-8 (2 physical host VMs, 4 TPU chips per host = 8 TPU chips total),
#   or single-host TPU v6e-4 via `NNODES=1 NPROC_PER_NODE=4`.
#
# Parallelism Config:
#   Pure FSDP2 (TP=1, DP_SHARD=TOTAL_CHIPS, PP=1) is used by default on TPU.
#   Do NOT enable tensor_parallel_size > 1 on TPU without re-testing: TorchTitan's
#   DTensor full_tensor() backward path produces non-finite gradients at TP > 1.

set -xeuo pipefail

export VERL_PLATFORM=tpu
export PYTHONUNBUFFERED=1

# Project and Experiment details
project_name="${PROJECT_NAME:-verl_tpu_sft}"
exp_name="${EXPERIMENT_NAME:-qwen3_0.6b_gsm8k_sft_torchtitan}"

# Paths (overridable via env vars; defaults to /data)
DATA_HOME="${DATA_HOME:-/data}"
MODEL_PATH="${MODEL_PATH:-${DATA_HOME}/assets/hf/Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_HOME}/data/gsm8k_sft/train.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_HOME}/data/gsm8k_sft/test.parquet}"

# Distributed / TPU topology configs (defaults to single-host 4 chips; override NNODES=2 for v6e-8)
NNODES="${NNODES:-${NNODES_TRAINER:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${N_CHIPS_TRAINER:-4}}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TOTAL_CHIPS=$((NNODES * NPROC_PER_NODE))

# Parallelism: Pure FSDP across all trainer chips
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${TOTAL_CHIPS}}"

# Sequence length bucket padding (`verl#8150`) to prevent XLA HLO recompilation on variable-length batches
PAD_TO_LENGTH="${PAD_TO_LENGTH:-True}"
PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-256}"

SMOKE_TEST="${SMOKE_TEST:-0}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-8}"
    TEST_FREQ="${TEST_FREQ:-4}"
else
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
    TEST_FREQ="${TEST_FREQ:-5}"
fi

# Note on `data.ignore_input_ids_mismatch=True`:
# Qwen3's chat template injects `<think>\n\n</think>\n\n` only on the last assistant turn and strips
# `<think>` blocks from earlier assistant turns, causing full-conversation tokenization to differ from
# per-turn incremental concatenation in `MultiTurnSFTDataset`.
torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.val_max_samples=32 \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=False \
    data.max_length=2048 \
    data.max_token_len_per_gpu=2048 \
    data.ignore_input_ids_mismatch=True \
    model.use_remove_padding=True \
    engine=torchtitan \
    model=hf_model \
    model.path="${MODEL_PATH}" \
    optim=torchtitan \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_factor=0.1 \
    optim.decay_type=cosine \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    engine.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    engine.pipeline_parallel_size=1 \
    engine.context_parallel_size=1 \
    engine.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    engine.use_torch_compile=False \
    engine.attn_type=varlen \
    engine.max_seq_len=2048 \
    engine.pad_to_length="${PAD_TO_LENGTH}" \
    engine.pad_to_length_bucket="${PAD_TO_LENGTH_BUCKET}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq=-1 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=2 \
    trainer.resume_mode=disable \
    trainer.device=tpu \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    "$@"

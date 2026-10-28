#!/usr/bin/env bash
# ReTool Stage 1 (Cold-Start Multi-Turn SFT) | Qwen3-30B-A3B (MoE) | ReTool-SFT | TorchTitan Engine (SimpleFSDP=8, CP=4, EP=1, DEFER_NEVER) | TPU v6e-32 (4x8)
#
# Targets 1 x TPU v6e-32 slice (examples/tpu/gke/ray-tpu-v6e32-2slice.yaml, tpu-group-0: 8 hosts x 4 chips = 32 chips).
# On TPU 7x 2x2x4 (examples/tpu/gke/ray-tpu-v7x16-2slice.yaml), pass:
#   TPU_ACCELERATOR_TYPE=tpu7x NNODES_TRAINER=4 N_CHIPS_TRAINER=8
#
# Qwen3-30B-A3B architecture constraints on TPU:
#   - 48 MoE layers, hidden_size=2048, 32 query heads, 4 KV heads (head_dim=128), 128 routed experts (top_k=8,
#     moe_intermediate_size=768), ~30.5B total params (~3.3B active per token).
#   - CONTEXT_PARALLEL_SIZE must divide num_key_value_heads (4), so max Ulysses CP is 4 (FSDP=8, CP=4 on 32 chips).
#   - EXPERT_PARALLEL_SIZE must be 1 on TPU (TpuGroupedExperts MegaBlox Pallas kernel requires 256-row-aligned
#     routed rows).
#   - TPU_EAGER_MODE=DEFER_NEVER runs each eager op outside compiled blocks in Python order (see
#     examples/tpu/grpo/run_qwen3_32b_torchtitan.sh).

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_use_enhanced_launch_barrier=false}"
export VERL_TPU_SEQ_BUCKET_SIZE="${VERL_TPU_SEQ_BUCKET_SIZE:-2048}"
if [[ -n "${TORCH_LOGS-recompiles}" ]]; then
    export TORCH_LOGS="${TORCH_LOGS-recompiles}"
else
    unset TORCH_LOGS
fi
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
for extra_py_dir in /torchtitan/torchtitan /torchtitan /app /torchtpu-vllm /torchtpu-vllm/src /vllm-torchtpu /vllm-torchtpu/src; do
    if [[ -d "${extra_py_dir}" && ":${PYTHONPATH:-}:" != *":${extra_py_dir}:"* ]]; then
        PYTHONPATH="${extra_py_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
done
export PYTHONPATH=".:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

project_name="${PROJECT_NAME:-verl_tpu_retool}"
experiment_name="${EXPERIMENT_NAME:-multiturn-sft-qwen3-30b-a3b-tpu}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
TRAIN_DATA="${TRAIN_DATA:-${RAY_DATA_HOME}/dataset/ReTool-SFT/data/train-00000-of-00001.parquet}"
EVAL_DATA="${EVAL_DATA:-${RAY_DATA_HOME}/dataset/ReTool-SFT/data/train-00000-of-00001.parquet}"
MODEL_SRC_PATH="${MODEL_SRC_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-30B-A3B}"
MODEL_PATH="${MODEL_PATH:-/tmp/models/Qwen3-30B-A3B}"
STAGE_MODEL="${STAGE_MODEL:-1}"
SAVE_PATH="${SAVE_PATH:-${RAY_DATA_HOME}/checkpoint/${experiment_name}}"

if [[ "${STAGE_MODEL}" == "1" ]]; then
    python3 "${VERL_ROOT}/examples/tpu/grpo/stage_model_local.py" "${MODEL_SRC_PATH}" "${MODEL_PATH}" 8
fi

# TPU v6e-32 slice topology (examples/tpu/gke/ray-tpu-v6e32-2slice.yaml): 8 hosts x 4 chips = 32 trainer chips.
export NNODES_TRAINER="${NNODES_TRAINER:-8}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-4}"
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-$((TOTAL_TRAINER_CHIPS / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))}"
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_NEVER}"
USE_SIMPLE_FSDP="${USE_SIMPLE_FSDP:-True}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"

# Train at 16384 max_length with Ulysses Context Parallelism (CP=4, FSDP=8) on 32 TPU chips
# (6 epochs x 62 steps/epoch at train_batch_size=32 = 372 total steps)
MAX_LENGTH="${MAX_LENGTH:-16384}"
TRUNCATION="${TRUNCATION:-right}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$((MICRO_BATCH_SIZE_PER_GPU * MAX_LENGTH / CONTEXT_PARALLEL_SIZE))}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-6}"
SAVE_FREQ="${SAVE_FREQ:-186}"
TEST_FREQ="${TEST_FREQ:-62}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"
LOGGER_LIST="${LOGGER_LIST:-['console','tensorboard']}"

EXTRA_ARGS=()
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
    EXTRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

python3 -m verl.trainer.sft_trainer_ray \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${EVAL_DATA}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.max_length="${MAX_LENGTH}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    data.pad_mode=no_padding \
    data.truncation="${TRUNCATION}" \
    data.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    model=hf_model \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    engine=torchtitan \
    engine.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    engine.pipeline_parallel_size=1 \
    engine.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    engine.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    engine.expert_parallel_size="${EXPERT_PARALLEL_SIZE}" \
    engine.use_torch_compile=True \
    engine.use_splash_attention=True \
    engine.use_simple_fsdp="${USE_SIMPLE_FSDP}" \
    engine.tpu_eager_mode="${TPU_EAGER_MODE}" \
    engine.activation_checkpoint=full \
    engine.attn_type=varlen \
    engine.max_seq_len="${MAX_LENGTH}" \
    optim=torchtitan \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_factor=0.1 \
    optim.decay_type=cosine \
    "checkpoint.save_contents=['hf_model']" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.logger="${LOGGER_LIST}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.resume_mode=disable \
    trainer.device=tpu \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

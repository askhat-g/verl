#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Recipe 2 (Phase 1B / 2A): DAPO Math (DAPO-Math-17k -> AIME 2024) on TPU
# Target branch: jialei777/verl-upstream (tpu-main) with Raiden weight sync
#
# Frozen 1:1 with upstream recipe/dapo/test_dapo_7b.sh:
#   - train_batch_size=512, ppo_mini_batch_size=32, rollout.n=16
#   - max_prompt_length=2048, max_response_length=2048, overlong_buffer.len=512 (penalty_factor=1.0)
#   - clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=10.0
#   - use_kl_in_reward=False, use_kl_loss=False, loss_agg_mode=token-mean
#   - algorithm.filter_groups.enable=True, algorithm.filter_groups.metric=acc
#   - lr=1e-6, lr_warmup_steps=10, weight_decay=0.1, clip_grad=1.0
#
# Supported MODEL_FAMILY (Dense models with Raiden weight sync):
#   - qwen3_4b_base:       Qwen/Qwen3-4B-Base       (2 x v6e-8:  2 trainer hosts x 4 chips + 2 rollout hosts x 4 chips)
#   - qwen3_4b:            Qwen/Qwen3-4B            (2 x v6e-8:  2 trainer hosts x 4 chips + 2 rollout hosts x 4 chips)
#   - qwen3_32b:           Qwen/Qwen3-32B           (2 x v6e-32: 8 trainer hosts x 4 chips + 8 rollout hosts x 4 chips)
#
# Usage:
#   # Few-step validation / smoke test on Qwen3-4B-Base (DAPO Math with Raiden weight sync):
#   MODEL_FAMILY=qwen3_4b_base SMOKE_TEST=1 bash examples/tpu/grpo/run_recipe2_dapo_math_torchtitan.sh
#
#   # Full 200-step E2E convergence run on Qwen3-4B-Base, Qwen3-4B, or Qwen3-32B:
#   MODEL_FAMILY=qwen3_4b_base SMOKE_TEST=0 bash examples/tpu/grpo/run_recipe2_dapo_math_torchtitan.sh
# ==============================================================================

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VLLM_USE_V1=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"
export TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL="${TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL:-300s}"
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

MODEL_FAMILY="${MODEL_FAMILY:-qwen3_4b_base}"
RECIPE_VARIANT="${RECIPE_VARIANT:-dapo_math}"
SMOKE_TEST="${SMOKE_TEST:-0}"
RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
DATA_ROOT="${DATA_ROOT:-${RAY_DATA_HOME}/data}"
MODEL_ROOT="${MODEL_ROOT:-${RAY_DATA_HOME}/assets/hf}"
LOG_ROOT="${VERL_FILE_LOGGER_ROOT:-${RAY_DATA_HOME}/logs}"
CKPT_ENGINE_BACKEND="${CKPT_ENGINE_BACKEND:-raiden}"
RELEASE_SYNC_BUFFERS="${RELEASE_SYNC_BUFFERS:-True}"

# ==============================================================================
# FROZEN RECIPE HYPERPARAMETERS (1:1 with recipe/dapo/test_dapo_7b.sh — DO NOT MODIFY)
# ==============================================================================
readonly TRAIN_BSZ=512
readonly MINI_BSZ=32
readonly ROLLOUT_N=16
readonly MAX_PROMPT_LEN=2048
readonly MAX_RESP_LEN=2048
readonly OVERLONG_LEN=512
readonly OVERLONG_PENALTY_FACTOR=1.0
readonly CLIP_RATIO_LOW=0.2
readonly CLIP_RATIO_HIGH=0.28
readonly CLIP_RATIO_C=10.0
readonly LOSS_AGG_MODE="token-mean"
readonly LR=1e-6
readonly LR_WARMUP_STEPS=10
readonly WEIGHT_DECAY=0.1
readonly CLIP_GRAD=1.0
readonly TEMPERATURE=1.0
readonly TOP_P=1.0
readonly TOP_K=-1

case "${MODEL_FAMILY}" in
  qwen3_4b_base)
    MODEL_NAME="Qwen3-4B-Base"
    # 2 x v6e-8 slices (2 hosts x 4 chips per slice = 8 chips per slice)
    export NNODES_TRAINER="${NNODES_TRAINER:-2}"
    export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
    export NNODES_ROLLOUT="${NNODES_ROLLOUT:-2}"
    export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"
    TRAINER_TP="${TENSOR_PARALLEL_SIZE:-1}"
    ROLLOUT_TP="${ROLLOUT_TP:-8}"
    AC_MODE="${ACTIVATION_CHECKPOINT:-selective}"
    MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-4096}"
    ;;
  qwen3_4b)
    MODEL_NAME="Qwen3-4B"
    # 2 x v6e-8 slices (2 hosts x 4 chips per slice on jialeic-tpu-v6e8-2s-spot)
    export NNODES_TRAINER="${NNODES_TRAINER:-2}"
    export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
    export NNODES_ROLLOUT="${NNODES_ROLLOUT:-2}"
    export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"
    TRAINER_TP="${TENSOR_PARALLEL_SIZE:-1}"
    ROLLOUT_TP="${ROLLOUT_TP:-8}"
    AC_MODE="${ACTIVATION_CHECKPOINT:-selective}"
    MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-4096}"
    ;;
  qwen3_32b)
    MODEL_NAME="Qwen3-32B"
    # 2 x v6e-32 slices: 8 physical VM hosts x 4 TPU chips = 32 chips per slice
    export NNODES_TRAINER="${NNODES_TRAINER:-8}"
    export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
    export NNODES_ROLLOUT="${NNODES_ROLLOUT:-8}"
    export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"
    TRAINER_TP="${TENSOR_PARALLEL_SIZE:-1}"
    ROLLOUT_TP="${ROLLOUT_TP:-8}"
    AC_MODE="${ACTIVATION_CHECKPOINT:-full}"
    MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-4096}"
    ;;
  *)
    echo "Unsupported MODEL_FAMILY=${MODEL_FAMILY}. Choose from: qwen3_4b_base, qwen3_4b, qwen3_32b." >&2
    exit 1
    ;;
esac

TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / TRAINER_TP))
DP_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESP_LEN))
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
USE_SIMPLE_FSDP="${USE_SIMPLE_FSDP:-True}"
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_AND_FUSE}"
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

case "${RECIPE_VARIANT}" in
  dapo_math)
    TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/dapo_math/dapo-math-17k.parquet}"
    VAL_FILE="${VAL_FILE:-${DATA_ROOT}/aime_2024/aime-2024.parquet}"
    FILTER_GROUPS_ENABLE=True
    FILTER_GROUPS_METRIC=acc
    VAL_DO_SAMPLE=True
    VAL_TEMP="${TEMPERATURE}"
    VAL_TOP_P="${TOP_P}"
    # Note: BytedTsinghua-SIA/AIME-2024 (aime-2024.parquet) already repeats the
    # 30 AIME problems 32x (960 rows), so val_kwargs.n=1 computes avg@32 (cf. test_dapo_7b.sh)
    VAL_N=1
    REWARD_ARGS=(
      "reward.reward_manager.name=dapo"
      "reward.num_workers=8"
      "+reward.reward_kwargs.max_resp_len=${MAX_RESP_LEN}"
      "+reward.reward_kwargs.overlong_buffer_cfg.enable=True"
      "+reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_LEN}"
      "+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}"
      "+reward.reward_kwargs.overlong_buffer_cfg.log=True"
    )
    ;;
  prime_code)
    TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/eurus_2_rl/train.parquet}"
    VAL_FILE="${VAL_FILE:-${DATA_ROOT}/eurus_2_rl/test.parquet}"
    FILTER_GROUPS_ENABLE=False
    FILTER_GROUPS_METRIC=acc
    VAL_DO_SAMPLE=False
    VAL_TEMP=0.0
    VAL_TOP_P=1.0
    VAL_N=1
    REWARD_ARGS=(
      "reward.reward_manager.name=naive"
      "reward.num_workers=8"
    )
    ;;
  *)
    echo "Unsupported RECIPE_VARIANT=${RECIPE_VARIANT}. Choose from: dapo_math, prime_code." >&2
    exit 1
    ;;
esac

MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"

# Load local WandB credentials (.env is gitignored; disable xtrace to avoid echoing WANDB_API_KEY)
set +x
if [[ -f ".env" ]]; then
  source ".env"
fi
set -x
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
PROJECT_NAME="${PROJECT_NAME:-${WANDB_PROJECT:-verl-torchtpu-grpo-test}}"
EXP_TAG="${MODEL_FAMILY}_${RECIPE_VARIANT}_tpu"

if [[ "${SMOKE_TEST}" == "1" ]]; then
  TOTAL_STEPS="${TOTAL_STEPS:-3}"
  TEST_FREQ="${TEST_FREQ:-3}"
  EXP_NAME="${EXPERIMENT_NAME:-${EXP_TAG}_frozen_smoke}"
else
  TOTAL_STEPS="${TOTAL_STEPS:-200}"
  TEST_FREQ="${TEST_FREQ:-10}"
  EXP_NAME="${EXPERIMENT_NAME:-${EXP_TAG}}"
fi

# Patch generation_config.json with multi-EOS [151645, 151643] if model dir is local
if [[ -d "${MODEL_PATH}" ]]; then
  python3 examples/tpu/grpo/prepare_recipe_assets.py --patch-model-dirs "${MODEL_PATH}"
fi

export VERL_FILE_LOGGER_ROOT="${LOG_ROOT}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RAY_DATA_HOME}/tensorboard/${EXP_NAME}}"
mkdir -p "${VERL_FILE_LOGGER_ROOT}/${PROJECT_NAME}" "${TENSORBOARD_DIR}"

echo "[Recipe 2] Launching ${MODEL_NAME} (${EXP_NAME}) for ${TOTAL_STEPS} steps with backend=${CKPT_ENGINE_BACKEND}..."
echo "[Recipe 2] Frozen DAPO params: train_batch_size=${TRAIN_BSZ}, ppo_mini_batch_size=${MINI_BSZ}, n=${ROLLOUT_N}, filter_groups=${FILTER_GROUPS_ENABLE}"

python3 -m verl.trainer.main_ppo \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.num_warmup_batches=1 \
  trainer.v1.separate_async.parameter_sync_step=1 \
  transfer_queue.enable=True \
  model_engine=torchtitan \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}" \
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}" \
  algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
  algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESP_LEN}" \
  +data.max_length="${MAX_MODEL_LEN}" \
  +data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  data.filter_overlong_prompts=False \
  data.truncation=left \
  +data.pad_mode=no_padding \
  actor_rollout_ref.actor.strategy=torchtitan \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_torch_compile=True \
  actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
  actor_rollout_ref.actor.torchtitan.use_splash_attention=True \
  actor_rollout_ref.actor.torchtitan.use_simple_fsdp="${USE_SIMPLE_FSDP}" \
  actor_rollout_ref.actor.torchtitan.tpu_eager_mode="${TPU_EAGER_MODE}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
  actor_rollout_ref.actor.optim.clip_grad="${CLIP_GRAD}" \
  actor_rollout_ref.actor.optim.min_lr_factor=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BSZ}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TRAINER_TP}" \
  actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DP_SHARD_SIZE}" \
  actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
  actor_rollout_ref.actor.torchtitan.attn_type=varlen \
  actor_rollout_ref.actor.torchtitan.activation_checkpoint="${AC_MODE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.checkpoint_engine.backend="${CKPT_ENGINE_BACKEND}" \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.release_buffers_after_sync="${RELEASE_SYNC_BUFFERS}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMP}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  "${REWARD_ARGS[@]}" \
  trainer.val_before_train=True \
  trainer.logger="['console','wandb','tensorboard','file']" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.log_val_generations=4 \
  trainer.rollout_data_dir=/tmp/verl_dump/rollout \
  trainer.validation_data_dir=/tmp/verl_dump/validation \
  trainer.save_freq=-1 \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs=10 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.nnodes="${NNODES_TRAINER}" \
  trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
  actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
  actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
  +rollout.nnodes="${NNODES_ROLLOUT}" \
  +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
  "$@"

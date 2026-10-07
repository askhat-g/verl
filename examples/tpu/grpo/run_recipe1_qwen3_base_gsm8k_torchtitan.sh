#!/usr/bin/env bash
# Copyright 2025 Google LLC
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
# Recipe 1 (Phase 1A): Small Base Model (Qwen3-4B-Base) Math GRPO on TPU v6e-8 x2 Slices
# Target branch: jialei777/verl-upstream (tpu-main) with Raiden weight sync
#
# Model:
#   - Qwen/Qwen3-4B-Base on GSM8K (validated 200-step val/acc: 49.2% -> 92.2%)
#
# Usage:
#   # 5-step smoke test on Qwen3-4B-Base:
#   SMOKE_TEST=1 bash examples/tpu/grpo/run_recipe1_qwen3_base_gsm8k_torchtitan.sh
#
#   # Full 200-step E2E convergence run on Qwen3-4B-Base:
#   SMOKE_TEST=0 bash examples/tpu/grpo/run_recipe1_qwen3_base_gsm8k_torchtitan.sh
# ==============================================================================

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VLLM_USE_V1=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

SMOKE_TEST="${SMOKE_TEST:-0}"
RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
DATA_ROOT="${DATA_ROOT:-${RAY_DATA_HOME}/data}"
MODEL_ROOT="${MODEL_ROOT:-${RAY_DATA_HOME}/assets/hf}"
LOG_ROOT="${VERL_FILE_LOGGER_ROOT:-${RAY_DATA_HOME}/logs}"
CKPT_ENGINE_BACKEND="${CKPT_ENGINE_BACKEND:-raiden}"
RELEASE_SYNC_BUFFERS="${RELEASE_SYNC_BUFFERS:-True}"

# TPU 2-slice v6e-8 topology (2 VM hosts x 4 chips per slice = 8 chips per slice)
export NNODES_TRAINER="${NNODES_TRAINER:-2}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-2}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"

TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))

MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Base}"
EXP_TAG="${EXP_TAG:-qwen3_4b_base_gsm8k_tpu}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"

MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_ROOT}/gsm8k/test.parquet}"

# Load local WandB credentials (.env is gitignored; disable xtrace to avoid echoing WANDB_API_KEY)
set +x
if [[ -f ".env" ]]; then
  source ".env"
fi
set -x
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
PROJECT_NAME="${PROJECT_NAME:-${WANDB_PROJECT:-verl-torchtpu-grpo-test}}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

USE_SIMPLE_FSDP="${USE_SIMPLE_FSDP:-True}"
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_AND_FUSE}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / TENSOR_PARALLEL_SIZE))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"
ROLLOUT_TP="${ROLLOUT_TP:-${TOTAL_ROLLOUT_CHIPS}}"

# Truncated importance sampling for separate_async off-policy stability
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

if [[ "${SMOKE_TEST}" == "1" ]]; then
  TOTAL_STEPS="${TOTAL_STEPS:-5}"
  TEST_FREQ="${TEST_FREQ:-5}"
  EXP_NAME="${EXPERIMENT_NAME:-${EXP_TAG}_smoke}"
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

echo "[Recipe 1] Launching ${MODEL_NAME} (${EXP_NAME}) for ${TOTAL_STEPS} steps with backend=${CKPT_ENGINE_BACKEND}..."
echo "[Recipe 1] Structured JSONL reward curve will be written to: ${VERL_FILE_LOGGER_ROOT}/${PROJECT_NAME}/${EXP_NAME}.jsonl"

python3 -m verl.trainer.main_ppo \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.num_warmup_batches=1 \
  trainer.v1.separate_async.parameter_sync_step=1 \
  transfer_queue.enable=True \
  model_engine=torchtitan \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
  algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  +data.max_length=4096 \
  +data.max_token_len_per_gpu=4096 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.pad_mode=no_padding \
  actor_rollout_ref.actor.strategy=torchtitan \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_torch_compile=True \
  actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
  actor_rollout_ref.actor.torchtitan.use_splash_attention=True \
  actor_rollout_ref.actor.torchtitan.use_simple_fsdp="${USE_SIMPLE_FSDP}" \
  actor_rollout_ref.actor.torchtitan.tpu_eager_mode="${TPU_EAGER_MODE}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
  actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
  actor_rollout_ref.actor.torchtitan.attn_type=varlen \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.checkpoint_engine.backend="${CKPT_ENGINE_BACKEND}" \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.release_buffers_after_sync="${RELEASE_SYNC_BUFFERS}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
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

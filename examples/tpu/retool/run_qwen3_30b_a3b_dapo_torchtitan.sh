#!/usr/bin/env bash
# ReTool Stage 2 (Multi-Turn DAPO RL with Code Interpreter) | Qwen3-30B-A3B (MoE) | TorchTitan (SimpleFSDP=8, CP=4, EP=1, DEFER_NEVER) & Async Rollout (TP=32, EP=32, Raiden) | TPU v6e-32 x2 Slices
#
# Targets 2 x TPU v6e-32 slices (examples/tpu/gke/ray-tpu-v6e32-2slice.yaml):
#   - Slice tpu-group-0 (8 hosts x 4 chips = 32 chips): TorchTitan trainer (SimpleFSDP=8, CP=4, TP=1, EP=1,
#     TPU_EAGER_MODE=DEFER_NEVER, use_dynamic_bsz=True, VERL_TPU_SEQ_BUCKET_SIZE=2048).
#   - Slice tpu-group-1 (8 hosts x 4 chips = 32 chips): 1 multi-host vLLM server with ROLLOUT_TP=32 and ROLLOUT_EP=32
#     (4 whole experts per chip, matching examples/tpu/grpo/run_qwen3_30b_a3b_torchtitan.sh).
# On 2 x TPU 7x 2x2x4 slices (examples/tpu/gke/ray-tpu-v7x16-2slice.yaml), pass:
#   TPU_ACCELERATOR_TYPE=tpu7x NNODES_TRAINER=4 N_CHIPS_TRAINER=8 NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8
#
# Default schedule (512 rollouts/step, 8K response / 8 turns, 400 steps):
#   TRAIN_BATCH_SIZE=64, PPO_MINI_BATCH_SIZE=64, N_RESP_PER_PROMPT=8, MAX_RESPONSE_LENGTH=8192, MAX_TURNS=8,
#   bypass_mode=True, 4-worker loopback SandboxFusion (8s timeout).
# Long-horizon 16K / 16-turn override (1,024 rollouts/step):
#   MAX_RESPONSE_LENGTH=16384 MAX_TURNS=16 N_RESP_PER_PROMPT=16 ACTOR_LR=2e-6 CLIP_GRAD=0.0625 TOTAL_TRAINING_STEPS=150 \
#     bash examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_use_enhanced_launch_barrier=false}"
export VERL_TPU_SEQ_BUCKET_SIZE="${VERL_TPU_SEQ_BUCKET_SIZE:-2048}"
export VERL_AGENT_LOOP_MAX_INFLIGHT_PROMPTS="${VERL_AGENT_LOOP_MAX_INFLIGHT_PROMPTS:-64}"
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

# Loopback SandboxFusion port (avoiding KubeRay's 8080). The server is started by the sandbox execution actor on the
# node it is placed on (ray_actor_options in sandbox_fusion_tool_config_tpu.yaml), not by this script.
SANDBOX_FUSION_PORT="${SANDBOX_FUSION_PORT:-18080}"

# ================= data/model/tool =================
RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
DATA_ROOT="${DATA_ROOT:-${RAY_DATA_HOME}}"

dapo_math_17k="${DATA_ROOT}/dataset/BytedTsinghua-SIA/DAPO-Math-17k"
aime_2024="${DATA_ROOT}/dataset/Maxwell-Jia/AIME_2024"
aime_2025="${DATA_ROOT}/dataset/yentinglin/aime_2025"

sft_experiment_name="${SFT_EXPERIMENT_NAME:-multiturn-sft-qwen3-30b-a3b-tpu}"
DEFAULT_SFT_SRC="${RAY_DATA_HOME}/checkpoint/${sft_experiment_name}/global_step_372/huggingface"
if [[ -z "${MODEL_SRC_PATH:-}" ]]; then
    if [[ -f "${DEFAULT_SFT_SRC}/model.safetensors.index.json" ]]; then
        MODEL_SRC_PATH="${DEFAULT_SFT_SRC}"
    else
        MODEL_SRC_PATH="${RAY_DATA_HOME}/assets/hf/Qwen3-30B-A3B"
    fi
fi
model_path="${MODEL_PATH:-/tmp/models/${sft_experiment_name}}"
STAGE_MODEL="${STAGE_MODEL:-1}"

if [[ "${STAGE_MODEL}" == "1" ]]; then
    python3 "${VERL_ROOT}/examples/tpu/grpo/stage_model_local.py" "${MODEL_SRC_PATH}" "${model_path}" 8
fi

train_files="['${dapo_math_17k}']"
test_files="['${aime_2025}', '${aime_2024}']"

tool_config_path="examples/tpu/retool/sandbox_fusion_tool_config_tpu.yaml"
retool_py_path="pkg://examples.tpu.retool.retool_tpu"

project_name="${PROJECT_NAME:-verl_tpu_retool}"
experiment_name="${EXPERIMENT_NAME:-qwen3-30b-a3b_retool_dapo_tpu}"
default_local_dir="${CKPTS_DIR:-/tmp/checkpoint/${experiment_name}}"

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss="${USE_KL_LOSS:-False}"
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns="${MAX_TURNS:-8}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-8192}"
max_model_len=$((max_prompt_length + max_response_length))
actor_lr="${ACTOR_LR:-1.4e-6}"
min_lr_factor="${MIN_LR_FACTOR:-1.0}"
clip_grad="${CLIP_GRAD:-0.088}"

train_batch_size="${TRAIN_BATCH_SIZE:-64}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-64}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
n_resp_per_prompt_val="${N_RESP_PER_PROMPT_VAL:-30}"

# ================= TPU 2 x v6e-32 slice topology & MoE parallelism =================
export NNODES_TRAINER="${NNODES_TRAINER:-8}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-8}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"

TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))

# On a multi-host slice vLLM's TP size equals the slice device count (see run_qwen3_30b_a3b_torchtitan.sh).
ROLLOUT_TP="${ROLLOUT_TP:-${TOTAL_ROLLOUT_CHIPS}}"
ROLLOUT_EP="${ROLLOUT_EP:-${ROLLOUT_TP}}"
ROLLOUT_BLOCK_SIZE="${ROLLOUT_BLOCK_SIZE:-256}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-$((train_batch_size / ppo_mini_batch_size))}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-4}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-16}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TRAINER_EP="${TRAINER_EP:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-4}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"

USE_SIMPLE_FSDP="${USE_SIMPLE_FSDP:-True}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-4608}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-4608}"
actor_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}"
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
torchtitan_max_seq_len=$(( (actor_max_token_len_per_gpu > log_prob_max_token_len_per_gpu ? actor_max_token_len_per_gpu : log_prob_max_token_len_per_gpu) * CONTEXT_PARALLEL_SIZE ))
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_NEVER}"
CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-raiden}"
if [[ -z "${RELEASE_SYNC_BUFFERS:-}" ]]; then
    if python3 "${SCRIPT_DIR}/retool_tpu.py" --check-raiden-unbind >/dev/null 2>&1; then
        RELEASE_SYNC_BUFFERS="True"
    else
        RELEASE_SYNC_BUFFERS="False"
    fi
fi
VERIFY_PARITY="${VERIFY_PARITY:-off}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.62}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"

# Off-policy correction & bypass_mode for separate_async overlap
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
BYPASS_MODE="${BYPASS_MODE:-True}"
LOSS_ARGS=("algorithm.rollout_correction.bypass_mode=${BYPASS_MODE}")
if [[ "${BYPASS_MODE}" == "True" || "${BYPASS_MODE}" == "true" ]]; then
    LOSS_ARGS+=("actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode")
fi

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_FREQ="${SAVE_FREQ:-20}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"
TEST_FREQ="${TEST_FREQ:-25}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
TRUNCATION="${TRUNCATION:-left}"
LOGGER_LIST="${LOGGER_LIST:-['console','tensorboard']}"

python3 -m verl.trainer.main_ppo \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PYTHONPATH}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_FUSION_PORT=${SANDBOX_FUSION_PORT}" \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}" \
    trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD}" \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    "${LOSS_ARGS[@]}" \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.return_raw_chat=True \
    data.train_batch_size="${train_batch_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    +data.max_length="${max_model_len}" \
    +data.max_token_len_per_gpu="${actor_max_token_len_per_gpu}" \
    data.filter_overlong_prompts=False \
    data.truncation="${TRUNCATION}" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    +data.pad_mode=no_padding \
    data.custom_cls.path="${retool_py_path}" \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path="${retool_py_path}" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.strategy=torchtitan \
    "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.actor.optim.min_lr_factor="${min_lr_factor}" \
    actor_rollout_ref.actor.optim.clip_grad="${clip_grad}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_max_token_len_per_gpu}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${log_prob_max_token_len_per_gpu}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${log_prob_max_token_len_per_gpu}" \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_splash_attention=True \
    actor_rollout_ref.actor.torchtitan.tpu_eager_mode="${TPU_EAGER_MODE}" \
    actor_rollout_ref.actor.torchtitan.use_simple_fsdp="${USE_SIMPLE_FSDP}" \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.expert_parallel_size="${TRAINER_EP}" \
    actor_rollout_ref.actor.torchtitan.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${torchtitan_max_seq_len}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
    actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size="${ROLLOUT_BLOCK_SIZE}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.checkpoint_engine.backend="${CHECKPOINT_BACKEND}" \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.release_buffers_after_sync="${RELEASE_SYNC_BUFFERS}" \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.verify_parity="${VERIFY_PARITY}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${max_turns}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${max_turns}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${tool_config_path}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n="${n_resp_per_prompt_val}" \
    trainer.logger="${LOGGER_LIST}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.log_val_generations=20 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"

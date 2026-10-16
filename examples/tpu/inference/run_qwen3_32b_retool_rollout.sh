#!/usr/bin/env bash
# ReTool Stage 2 rollout-only benchmark | Qwen3-32B SFT | vLLM rollout (8 x TP=4) + SandboxFusion code interpreter | TPU v6e
#
# Runs only the rollout half of the ReTool Stage 2 recipe (recipe/retool/run_qwen3_32b_dapo_tpu.sh) through
# rollout_benchmark.py: the same env, data, model, tool and Hydra overrides, with no trainer and no weight sync, so the
# generation time of one RL step can be measured on its own. Trainer-only overrides are kept verbatim (unused).
#
# Setup (once per checkout):
#   1. The ReTool TPU files (SandboxFusion tool + local server, dataset and reward glue) live in the ReTool recipe of
#      askhat-g/verl-recipe, not in upstream verl-recipe. Point this checkout's `recipe` submodule at it:
#        git submodule update --init recipe
#        git -C recipe remote add retool-recipe https://github.com/askhat-g/verl-recipe.git
#        git -C recipe fetch retool-recipe
#        git -C recipe checkout cfe5527   # ReTool TPU recipe (askhat-g/verl-recipe PR #1)
#      This must replace the `recipe` folder, not sit next to it: the tool is loaded as
#      recipe.retool.retool.CustomSandboxFusionTool and `recipe` has no __init__.py, so Python merges every `recipe`
#      folder on its path into one package. `git status` then shows `recipe` as modified; that is expected.
#   2. Model: the ReTool Stage 1 SFT checkpoint (multiturn-sft-qwen3-32b-tpu/global_step_372/huggingface) under
#      /dev/shm/checkpoint/ or ${HDFS_ROOT}/checkpoint/ on every rollout host, or MODEL_PATH=<dir>. Loading it straight
#      from a GCS FUSE mount is slow (vLLM memory-maps it); see README.md for how to pre-read it.
#   3. Data: DAPO-Math-17k, AIME 2024 and AIME 2025 under ${DATA_ROOT}/dataset on the node that runs this script.
#
# Usage (extra arguments are Hydra overrides, see rollout_benchmark.py):
#   bash examples/tpu/inference/run_qwen3_32b_retool_rollout.sh +benchmark.mode=train data.seed=1   # 256 prompts x 16
#   bash examples/tpu/inference/run_qwen3_32b_retool_rollout.sh +benchmark.mode=val                 # AIME 24+25 x 30
#   bash examples/tpu/inference/run_qwen3_32b_retool_rollout.sh +benchmark.max_prompts=8 data.seed=1 # smoke test
#   # As in training, the first TPU slice is left for the trainer and replicas (whole hosts each) go on the others:
#   TPU_ACCELERATOR_TYPE=v6e-4 NNODES_ROLLOUT=1 bash examples/tpu/inference/run_qwen3_32b_retool_rollout.sh   # 1 x v6e-4
#   TPU_ACCELERATOR_TYPE=v6e-8 NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8 ROLLOUT_TP=8 bash ...                     # 4 x v6e-8
#   # Running it directly needs this checkout on every node. Otherwise submit it as a Ray job, which copies the code:
#   ray job submit --working-dir . --runtime-env-json '{"excludes": [".git"], "env_vars": {"PYTHONPATH": "."}}' \
#       -- bash examples/tpu/inference/run_qwen3_32b_retool_rollout.sh +benchmark.max_prompts=8 data.seed=1

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-32}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"
export VERL_TPU_SEQ_BUCKET_SIZE="${VERL_TPU_SEQ_BUCKET_SIZE:-18432}"
export VERL_AGENT_LOOP_MAX_INFLIGHT_PROMPTS="${VERL_AGENT_LOOP_MAX_INFLIGHT_PROMPTS:-64}"
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RETOOL_DIR="${VERL_ROOT}/recipe/retool"
cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${RETOOL_DIR}/sandbox_fusion_tools.py" ]]; then
    set +x
    echo "ERROR: ${RETOOL_DIR} is not the ReTool TPU recipe (sandbox_fusion_tools.py is missing). Run:" >&2
    echo "  git -C ${VERL_ROOT}/recipe remote add retool-recipe https://github.com/askhat-g/verl-recipe.git" >&2
    echo "  git -C ${VERL_ROOT}/recipe fetch retool-recipe && git -C ${VERL_ROOT}/recipe checkout cfe5527" >&2
    exit 1
fi

# SandboxFusion code execution server on port 18080. Every node that runs the tool also starts its own on demand
# (recipe/retool/sandbox_fusion_tools.py); retool_sandbox_fusion_tool_config.yaml points the tool at that port.
SANDBOX_FUSION_PORT=18080
if ! python3 "${RETOOL_DIR}/sandbox_fusion_server.py" --port "${SANDBOX_FUSION_PORT}" --check >/dev/null 2>&1; then
    nohup python3 "${RETOOL_DIR}/sandbox_fusion_server.py" --host 0.0.0.0 --port "${SANDBOX_FUSION_PORT}" \
        > /tmp/sandbox_fusion_server.log 2>&1 &
    sleep 2
fi

# ================= data/model/tool =================
HDFS_ROOT="${HDFS_ROOT:-/data/jialei}"
DATA_ROOT="${DATA_ROOT:-/data/jialei}"

dapo_math_17k="${DATA_ROOT}/dataset/BytedTsinghua-SIA/DAPO-Math-17k"
aime_2024="${DATA_ROOT}/dataset/Maxwell-Jia/AIME_2024"
aime_2025="${DATA_ROOT}/dataset/yentinglin/aime_2025"
sft_experiment_name="${SFT_EXPERIMENT_NAME:-multiturn-sft-qwen3-32b-tpu}"
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -f "/dev/shm/checkpoint/${sft_experiment_name}/global_step_372/huggingface/model.safetensors.index.json" ]]; then
        model_path="/dev/shm/checkpoint/${sft_experiment_name}/global_step_372/huggingface"
    elif [[ -f "${HDFS_ROOT}/checkpoint/${sft_experiment_name}/global_step_372/huggingface/model.safetensors.index.json" ]]; then
        model_path="${HDFS_ROOT}/checkpoint/${sft_experiment_name}/global_step_372/huggingface"
    elif [[ -f "/tmp/multiturn-sft-qwen3-32b-tpu-hf/model.safetensors.index.json" ]]; then
        model_path="/tmp/multiturn-sft-qwen3-32b-tpu-hf"
    elif [[ -f "/dev/shm/Qwen3-32B/model.safetensors.index.json" ]]; then
        model_path="/dev/shm/Qwen3-32B"
    elif [[ -f "/tmp/Qwen3-32B/model.safetensors.index.json" ]]; then
        model_path="/tmp/Qwen3-32B"
    else
        model_path="${HDFS_ROOT}/assets/hf/Qwen3-32B"
    fi
    if [[ "${model_path}" != *"${sft_experiment_name}"* ]]; then
        echo "WARNING: ReTool SFT checkpoint not found, benchmarking the base model ${model_path}" >&2
    fi
else
    model_path="${MODEL_PATH}"
fi

train_files="['${dapo_math_17k}']"
test_files="['${aime_2025}', '${aime_2024}']"

# tool & custom dataset/reward
tool_config_path="${SCRIPT_DIR}/retool_sandbox_fusion_tool_config.yaml"
retool_py_path="${RETOOL_DIR}/retool.py"

# logging / checkpoint
project_name="${PROJECT_NAME:-${WANDB_PROJECT:-verl-torchtpu-grpo-test}}"
experiment_name="${EXPERIMENT_NAME:-qwen3-32b_retool_dapo_tpu}"
if [[ -z "${CKPTS_DIR:-}" ]]; then
    if [[ -d "${HDFS_ROOT}" ]]; then
        default_local_dir="${HDFS_ROOT}/checkpoint/${experiment_name}"
    else
        default_local_dir="/dev/shm/checkpoint/${experiment_name}"
    fi
else
    default_local_dir="${CKPTS_DIR}"
fi

# ================= algorithm (matching recipe/retool/run_qwen2-32b_dapo.sh) =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns="${MAX_TURNS:-16}"
# Full 16384 response + 2048 prompt = 18432 context length via Ulysses CP=8 (FSDP=4, CP=8 on 32 chips)
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-16384}"
max_model_len=$((max_prompt_length + max_response_length))
actor_lr="${ACTOR_LR:-1e-6}"

train_batch_size="${TRAIN_BATCH_SIZE:-256}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-16}"
n_resp_per_prompt_val="${N_RESP_PER_PROMPT_VAL:-30}"

# ================= TPU 2 x v6e-32 topology & performance (PR #62) =================
export NNODES_TRAINER="${NNODES_TRAINER:-8}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-8}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"

TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))
ROLLOUT_TP="${ROLLOUT_TP:-4}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-8}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"

actor_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$((max_model_len / CONTEXT_PARALLEL_SIZE))}"
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$((max_model_len / CONTEXT_PARALLEL_SIZE))}"
torchtitan_max_seq_len=$(( (actor_max_token_len_per_gpu > log_prob_max_token_len_per_gpu ? actor_max_token_len_per_gpu : log_prob_max_token_len_per_gpu) * CONTEXT_PARALLEL_SIZE ))
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_NEVER}"
RELEASE_SYNC_BUFFERS="${RELEASE_SYNC_BUFFERS:-True}"
VERIFY_PARITY="${VERIFY_PARITY:-off}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"

# Off-policy correction for separate_async overlap
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
TRUNCATION="${TRUNCATION:-left}"

python3 "${SCRIPT_DIR}/rollout_benchmark.py" \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
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
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
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
    actor_rollout_ref.actor.torchtitan.use_simple_fsdp=True \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${torchtitan_max_seq_len}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.checkpoint_engine.backend=raiden \
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
    trainer.logger="['console','wandb','tensorboard','file']" \
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
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"

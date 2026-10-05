#!/usr/bin/env bash
# GRPO | Qwen3-8B | GSM8K | TorchTitan Training (SimpleFSDP, torch.compile + splash attention, DEFER_AND_FUSE) & vLLM Rollout | TPU v6e-8 x2 Slices
# V1 PPOTrainer (Separate Async Overlap), Raiden weight sync.
#
# Mirrors run_qwen3_4b_torchtitan.sh so weight-sync timings are directly comparable.
# With rollout TP=8, every large Qwen3-8B tensor is TPU tile-aligned
# (e.g. down_proj per rank = [4096, 12288 / 8 = 1536], 1536 % 128 == 0), so Raiden
# should never fall back to CPU TileBuffer() during Sampler H2D.

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VLLM_USE_V1=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# JAX/XLA Launch Barrier Configuration
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"

# PyTorch Dynamo / Compile Logging and Recompile Limits
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

exp_name="${EXPERIMENT_NAME:-qwen3_8b_gsm8k_fsdp_compile}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
# Free the bf16 tensors Raiden sends from and the WeightSynchronizer pinning them after each transfer, so they
# do not stay resident through training until the next sync. For 8B on 8 chips that is ~3.4 GiB/chip: ~1.7 GiB
# of local shards plus ~1.7 GiB of full q/k/v, which torchtitan's fused-wqkv save hook still all-gathers.
# This is the engine default; set False to keep and rebind them instead (faster init, more HBM).
RELEASE_SYNC_BUFFERS="${RELEASE_SYNC_BUFFERS:-True}"
# Full AC, as in torchtitan's qwen3-8b v6e-8 recipe (selective AC OOMs on v6e).
ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-full}"
# SimpleFSDP: the FSDP all-gather/reduce-scatter are traced into each compiled TransformerBlock instead of running
# from FSDP2 hooks. Compared with FSDP2 (which fills HBM in update_actor, with the TPU runtime constantly unloading
# and reloading programs), it lowers the update_actor tensor peak by ~3.7 GiB per chip and runs update_actor ~6x
# faster. FSDP2 (False) is not supported for 8B on 8 chips: update_actor runs out of HBM.
USE_SIMPLE_FSDP="${USE_SIMPLE_FSDP:-True}"
# torch_tpu eager mode for the ops outside the compiled blocks (LM head, log-prob/loss, grad clipping, optimizer):
# DEFER_AND_FUSE fuses them into larger XLA programs; null keeps torch_tpu's default of one program per op. Even
# without the reference model, null raises the update_actor tensor peak to ~31.2 GiB per chip at 8B, at the limit.
# DEFER_AND_FUSE's programs keep ~3.5 GiB of scratch reserved between steps, which the TPU runtime gives back when
# memory is tight.
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_AND_FUSE}"
# KL-to-reference loss, off by default. The reference model keeps a second copy of the weights on every trainer
# chip (fp32, ~3.9 GiB per chip at 8B on 8 chips; parameter offload is not available for it on TPU), leaving only
# ~0.5 GiB at the update_actor peak. Set True to enable it.
USE_KL_LOSS="${USE_KL_LOSS:-False}"
# Per-micro-batch token budget for training and log-prob passes. Logits ([tokens, 151936]) and
# attention scale with it, so 2048 keeps one micro-batch well inside HBM (torchtitan's v6e-8 recipe
# trains at seq_len 2048). Must stay >= MAX_MODEL_LEN.
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-2048}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

# Project details
project_name='verl_tpu_grpo'

# Paths
RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-8B}"

TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"

# TPU 2-slice v6e-8 configurations
export NNODES_TRAINER=2       # 2 physical VM hosts for training slice
export N_CHIPS_TRAINER=4      # 4 TPU chips per training host

export NNODES_ROLLOUT=2       # 2 physical VM hosts for rollout slice
export N_CHIPS_ROLLOUT=4      # 4 TPU chips per rollout host

TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))

# Sequence budget
MAX_PROMPT_LEN=512
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))

# Actor parallelism (default: Full FSDP across all 8 trainer chips: tp=1, dp_shard=8)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / TENSOR_PARALLEL_SIZE))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"

# Off-policy correction (truncated importance sampling). Required in the decoupled
# separate_async regime; see run_qwen3_0_6b_torchtitan.sh for the full derivation.
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

# TorchTitan trainer: per-TransformerBlock torch.compile(backend="tpu") with the splash attention Pallas
# kernel. Compile on TPU requires splash (compiled SDPA yields NaN gradients on torch_tpu), and with splash
# the trainer builds no dense [seq, seq] attention mask. By default the FSDP collectives run inside the compiled
# blocks (USE_SIMPLE_FSDP) and the eager ops between them are fused (TPU_EAGER_MODE=DEFER_AND_FUSE), as in
# run_qwen3_4b_torchtitan.sh. The reference model inherits these settings from the actor.

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
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    +data.max_length=4096 \
    +data.max_token_len_per_gpu=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint="${ACTIVATION_CHECKPOINT}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TOTAL_ROLLOUT_CHIPS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=raiden \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.release_buffers_after_sync="${RELEASE_SYNC_BUFFERS}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.log_val_generations=4 \
    trainer.rollout_data_dir=/tmp/verl_dump/rollout \
    trainer.validation_data_dir=/tmp/verl_dump/validation \
    trainer.save_freq=-1 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" "$@"

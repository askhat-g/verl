#!/usr/bin/env bash
# GRPO | Qwen3-32B | GSM8K | TorchTitan Training (SimpleFSDP, torch.compile + splash attention, DEFER_NEVER) & vLLM Rollout | TPU v6e-32 x2 Slices
# V1 PPOTrainer (Separate Async Overlap), Raiden weight sync.
#
# Scales run_qwen3_8b_torchtitan.sh (2 x v6e-8) to 2 x v6e-32 (examples/tpu/gke/ray-tpu-v6e32-2slice.yaml):
# slicke tpu-group-0 (8 hosts x 4 chips) runs the FSDP32 trainer and slice tpu-group-1 runs one TP32 vLLM server.
#
# Memory budget per trainer chip (32 GiB HBM, FSDP over 32 chips): fp32 params + grads + Adam m/v of 32.8B
# params is 32.8B * 16 B / 32 = ~16.4 GiB. Raiden sends bf16 local shards (~2 GiB) plus the full q/k/v that
# torchtitan's fused-wqkv save hook all-gathers (~6.7 GiB at 32B), freed after each sync by
# RELEASE_SYNC_BUFFERS=True.
#
# Qwen3-32B has 8 KV heads, so with rollout TP=32 vLLM replicates each KV head on 4 ranks. The MLP intermediate
# size 25600 / 32 = 800 is not a multiple of 128, so Raiden may fall back to CPU TileBuffer() for down/up/gate
# projections during Sampler H2D (correct, but slower). ROLLOUT_TP=8 keeps every tensor tile aligned at the cost
# of 4 vLLM replicas.

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

exp_name="${EXPERIMENT_NAME:-qwen3_32b_gsm8k_fsdp_compile}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
# Free the bf16 tensors Raiden sends from and the WeightSynchronizer pinning them after each transfer, so they
# do not stay resident through training until the next sync (see the memory budget above).
RELEASE_SYNC_BUFFERS="${RELEASE_SYNC_BUFFERS:-True}"
# Raiden weight-sync verification (RaidenParityCheck): off | norm (per-step L1/L2) | exact (step-0 bit-for-bit
# compare with the checkpoint vLLM loaded) | all. Use exact after changing sharding or upgrading tpu-sync / torch-tpu.
VERIFY_PARITY="${VERIFY_PARITY:-off}"
# Full AC, as for 8B (selective AC OOMs on v6e).
ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-full}"
# SimpleFSDP: FSDP all-gather/reduce-scatter traced into each compiled TransformerBlock. Set False for FSDP2.
USE_SIMPLE_FSDP="${USE_SIMPLE_FSDP:-True}"
# torch_tpu eager mode for the ops outside the compiled blocks. DEFER_NEVER runs each eager op as its own program,
# in Python order, so every trainer rank runs the same program sequence. DEFER_AND_FUSE (used by the 8B recipe) hangs
# here in update_actor within 1-3 steps: the vLLM image's sitecustomize sets
# TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false in every process, so the eager embedding all-gather /
# reduce-scatter are fused with neighbouring ops, and torch_tpu's timing-dependent repeated-op heuristic can cut that
# fused program differently on one of the 32 ranks. That rank then runs a different binary for the program holding
# the collective and all 32 chips wait forever. DEFER_NEVER is also faster here (update_actor 28.4 s vs 31.9 s).
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_NEVER}"
# KL-to-reference loss, off by default: the reference model keeps a second fp32 copy of the weights
# (~4.1 GiB per chip at 32B on 32 chips).
USE_KL_LOSS="${USE_KL_LOSS:-False}"
# Per-micro-batch token budget for training and log-prob passes. Must stay >= MAX_MODEL_LEN.
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-2048}"
# FSDP2 reshard policy (ignored with USE_SIMPLE_FSDP=True).
RESHARD_AFTER_FORWARD="${RESHARD_AFTER_FORWARD:-always}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"

# Project details
project_name='verl_tpu_grpo'

# Paths
RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
# Loading straight from GCS Fuse is slow for 62 GB (mmap page faults at ~20 MB/s per host); point MODEL_PATH at a
# local copy on every host to speed up model load.
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-32B}"

TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"

# TPU 2-slice v6e-32 configurations
export NNODES_TRAINER="${NNODES_TRAINER:-8}"     # 8 physical VM hosts for training slice
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"   # 4 TPU chips per training host

export NNODES_ROLLOUT="${NNODES_ROLLOUT:-8}"     # 8 physical VM hosts for rollout slice
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"   # 4 TPU chips per rollout host

TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
ROLLOUT_TP="${ROLLOUT_TP:-${TOTAL_ROLLOUT_CHIPS}}"

# Sequence budget
MAX_PROMPT_LEN=512
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))

# Actor parallelism (default: Full FSDP across all 32 trainer chips: tp=1, dp_shard=32)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / TENSOR_PARALLEL_SIZE))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"

# Off-policy correction (truncated importance sampling). Required in the decoupled
# separate_async regime; see run_qwen3_0_6b_torchtitan.sh for the full derivation.
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

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
    actor_rollout_ref.actor.torchtitan.reshard_after_forward="${RESHARD_AFTER_FORWARD}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
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
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.verify_parity="${VERIFY_PARITY}" \
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

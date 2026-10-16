# Rollout-Only Benchmark on Google Cloud TPU

[`rollout_benchmark.py`](rollout_benchmark.py) runs only the rollout half of a `trainer.v1` RL recipe: the vLLM
replicas, the agent loop (with its tools) and the streaming reward, with no trainer, weight sync or checkpointing.
Use it to measure how long a recipe's generation step takes and to try rollout settings without paying for training.

It does not re-implement any rollout logic. It builds the same verl objects that `PPOTrainerSeparateAsync` builds, in
the same order and from the same Hydra config (`LLMServerManager` standalone replicas, `GlobalRequestLoadBalancer`,
`FullyAsyncLLMServerClient`, `AgentLoopManagerTQ`, `RewardLoopManager`, TransferQueue), so a sampled response is
produced exactly as it would be at step 0 of training.

## ReTool (Qwen3-32B SFT + SandboxFusion)

[`run_qwen3_32b_retool_rollout.sh`](run_qwen3_32b_retool_rollout.sh) uses the settings of the ReTool Stage 2 recipe
(`recipe/retool/run_qwen3_32b_dapo_tpu.sh`): the Stage 1 SFT checkpoint writes Python, runs it in a SandboxFusion code
interpreter, reads the output and keeps going for up to 16 turns per DAPO-Math (or AIME) problem.

### Setup (once per checkout)

1. The ReTool TPU files (SandboxFusion tool and local server, dataset and reward glue) live in the ReTool recipe of
   [`askhat-g/verl-recipe`](https://github.com/askhat-g/verl-recipe), not in upstream `verl-recipe`. Point the
   `recipe` submodule at it:

   ```bash
   git submodule update --init recipe
   git -C recipe remote add retool-recipe https://github.com/askhat-g/verl-recipe.git
   git -C recipe fetch retool-recipe
   git -C recipe checkout cfe5527   # ReTool TPU recipe (askhat-g/verl-recipe PR #1)
   ```

   This has to replace the `recipe` folder, not sit next to it: the tool is loaded as
   `recipe.retool.retool.CustomSandboxFusionTool`, and because `recipe` has no `__init__.py`, Python merges every
   `recipe` folder on its path into one package. `git status` then shows `recipe` as modified, which is expected.
2. Model: the ReTool Stage 1 SFT checkpoint (`multiturn-sft-qwen3-32b-tpu/global_step_372/huggingface`) under
   `/dev/shm/checkpoint/` or `${HDFS_ROOT}/checkpoint/` on every rollout host, or `MODEL_PATH=<dir>`. Without it the
   script falls back to base Qwen3-32B and prints a warning; `summary.json` records the model path that was used.
   vLLM memory-maps the weights, which is slow on a GCS FUSE mount such as `/data` (about 6 minutes per 4 GB shard in
   testing). Copy the checkpoint to `/dev/shm`, or read it once on each rollout host so it loads from the page cache:
   `ls <dir>/model-*.safetensors | xargs -P 8 -I{} sh -c 'cat {} > /dev/null'` (about 4 minutes for Qwen3-32B).
3. Data: DAPO-Math-17k, AIME 2024 and AIME 2025 under `${DATA_ROOT}/dataset` (default `/data/jialei`) on the node that
   runs the script.

### Run

```bash
# One RL step's worth of rollouts: 256 DAPO-Math prompts x 16 = 4,096 multi-turn trajectories
bash examples/tpu/inference/run_qwen3_32b_retool_rollout.sh +benchmark.mode=train data.seed=1
# Validation: AIME 2024 + 2025 x 30, reports val-core/<data_source>/acc/mean@30 like the trainer
bash examples/tpu/inference/run_qwen3_32b_retool_rollout.sh +benchmark.mode=val
# Smoke test on a few prompts
bash examples/tpu/inference/run_qwen3_32b_retool_rollout.sh +benchmark.max_prompts=8 data.seed=1
```

Running the script directly needs the checkout on every node (e.g. on a shared filesystem). Otherwise submit it as a
Ray job from the checkout, so Ray copies the code (including `recipe/`) to every node, as the TPU CI does in
`tests/special_tpu/run_tpu_e2e_ci.sh`:

```bash
ray job submit --working-dir . --runtime-env-json '{"excludes": [".git"], "env_vars": {"PYTHONPATH": "."}}' \
    -- bash examples/tpu/inference/run_qwen3_32b_retool_rollout.sh +benchmark.max_prompts=8 data.seed=1
```

The defaults match the recipe's rollout: 8 vLLM replicas x TP=4 on a v6e-32 slice (4 chips per host). As in training,
the first TPU slice is left for the trainer and the replicas go on the other slices (or on the ones listed in
`actor_rollout_ref.rollout.tpu_slices=[...]`), and a replica always owns whole hosts. Set the layout to the rollout
slices you have, e.g. `TPU_ACCELERATOR_TYPE=v6e-4 NNODES_ROLLOUT=1` for one v6e-4 rollout slice (1 replica x TP=4) or
`TPU_ACCELERATOR_TYPE=v6e-8 NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=8 ROLLOUT_TP=8` for four v6e-8 rollout slices.

| Override | Default | Meaning |
| :--- | :--- | :--- |
| `+benchmark.mode` | `train` | `train`: `data.train_batch_size` prompts x `rollout.n`; `val`: the val files x `rollout.val_kwargs.n` |
| `+benchmark.num_batches` | `1` | Train mode: consecutive batches, each timed separately |
| `+benchmark.max_prompts` | all | Cap on prompts per batch |
| `+benchmark.output_dir` | `/tmp/rollout_benchmark/<mode>_<time>` | Output directory on the node running the task runner |
| `+benchmark.dump_text` | `true` | Decode prompt and response text into `trajectories.jsonl` |
| `data.seed` | random | Draw the same prompts in every run so results are comparable |

Any recipe override can be appended too, e.g. `actor_rollout_ref.rollout.max_num_seqs=256`, to see how a rollout
setting changes the generation time.

### Output

- `summary.json`: the resolved rollout settings (model path, sampling parameters, TP, replica count, vLLM engine
  arguments, turn limits), the vLLM start-up time and, per batch, the wall time, generated and total tokens/s,
  ms/token, percentiles of response length, tool turns and per-trajectory completion time, and accuracy. Val mode adds
  the trainer's `val-core/<data_source>/acc/{mean,maj,best}@N` metrics.
- `trajectories.jsonl`: one line per trajectory with prompt and response text, token counts, turns, score,
  correctness and timings.

The wall time of a batch is usually set by a few runaway responses that run to the 16,384-token limit and finish
alone at single-sequence speed, so compare `completion_s/p99` as well as `wall_time_s`.

### Notes

- Batches run one at a time. During training `PPOTrainerSeparateAsync` keeps two batches in flight
  (`num_warmup_batches=1`), so each request is generated the same way but the replicas see less load.
- The model weights come straight from the SFT checkpoint (`load_format=safetensors`), the same bf16 weights the
  trainer syncs to the rollout at step 0.
- Every vLLM restart leaves one exited `VLLM::EngineCore` process per host until the pod restarts; verl's start-up
  check prints the count per host.
- Like the trainer, the script imports verl's TorchTitan engine at start-up, so the image's torchtitan has to match
  this branch. `TPU_CI_IMAGE` in `tests/special_tpu/gke/tpu-ci-image.env` is the image the branch is tested with.

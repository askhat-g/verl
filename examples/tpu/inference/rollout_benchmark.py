# Copyright 2024-2025 BAAI and Google LLC
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
"""Rollout-only benchmark for verl agent-loop recipes.

Runs only the rollout ("sampler") half of a ``trainer.v1`` recipe: standalone vLLM replicas, the recipe's agent loop
(e.g. the multi-turn ``tool_agent`` with a code interpreter) and its streaming reward. There is no trainer, no weight
sync and no checkpointing, so the generation time can be measured on its own. ``run_qwen3_32b_retool_rollout.sh``
runs it with the ReTool Stage 2 recipe settings.

To keep the inference identical to training, this file does not re-implement any rollout logic. It wires up the same
verl objects that ``PPOTrainerSeparateAsync`` + ``TaskRunnerV1`` create, in the same order and with the same config:

    RLHFDataset (or data.custom_cls) -> AgentLoopManagerTQ (agent loop + tools + RewardLoopManager)
        -> FullyAsyncLLMServerClient -> GlobalRequestLoadBalancer -> LLMServerManager (standalone vLLM replicas)

Benchmark options are passed as extra Hydra overrides (all optional):

    +benchmark.mode=train          # "train": data.train_batch_size prompts x rollout.n with training sampling params
                                   # "val":   the val files x rollout.val_kwargs.n with validation sampling params
    +benchmark.num_batches=1       # train mode: number of consecutive batches, each timed separately
    +benchmark.max_prompts=null    # cap prompts per batch (smoke tests)
    +benchmark.output_dir=null     # default: /tmp/rollout_benchmark/<mode>_<timestamp>
    +benchmark.dump_text=true      # decode prompts / responses into trajectories.jsonl
    +benchmark.poll_interval=1.0   # seconds between TransferQueue completion polls
"""

import json
import logging
import os
import socket
import time
import uuid
from pprint import pprint
from typing import Any, Optional

import hydra
import numpy as np
import ray
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.experimental.reward_loop import RewardLoopManager
from verl.plugin.platform import get_platform
from verl.plugin.platform.platform_tpu_slices import resolve_tpu_rollout_layout
from verl.trainer import main_ppo
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.device import auto_set_device
from verl.utils.logging_utils import configure_verl_logging
from verl.workers.config import HFModelConfig
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Same Hydra config tree as verl.trainer.main_ppo, so every recipe override resolves identically.
VERL_TRAINER_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(main_ppo.__file__)), "config")

TRAJECTORY_FIELDS = [
    "prompts",
    "responses",
    "response_mask",
    "rm_scores",
    "num_turns",
    "data_source",
    "reward_model",
    "extra_fields",
    "metrics",
]
COLLECT_CHUNK_SIZE = 256
PROGRESS_LOG_INTERVAL_S = 60.0


def _percentiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Mean / p50 / p90 / p99 / min / max of ``values`` under ``prefix/``; empty input gives an empty dict."""
    if values.size == 0:
        return {}
    return {
        f"{prefix}/mean": float(values.mean()),
        f"{prefix}/p50": float(np.percentile(values, 50)),
        f"{prefix}/p90": float(np.percentile(values, 90)),
        f"{prefix}/p99": float(np.percentile(values, 99)),
        f"{prefix}/min": float(values.min()),
        f"{prefix}/max": float(values.max()),
    }


def summarize_trajectories(
    records: list[dict[str, Any]], wall_time_s: float, max_response_length: int, num_failed_prompts: int = 0
) -> dict[str, Any]:
    """Throughput, length, turn, timing and accuracy summary of one benchmark batch.

    Args:
        records: One dict per finished trajectory with ``prompt_len``, ``response_len`` (generated + tool tokens),
            ``generated_tokens`` (``sum(response_mask)``, model tokens only), ``num_turns``, ``score`` and the
            optional ``acc``, ``generate_s``, ``tool_calls_s``, ``num_cached_tokens``, ``done_s`` (seconds from dispatch
            until the trajectory was stored).
        wall_time_s: Seconds from dispatching the batch until every prompt group reached a terminal status.
        max_response_length: ``rollout.response_length``; responses that reach it count as clipped.
        num_failed_prompts: Prompt groups whose agent loop raised (no trajectories stored).

    Returns:
        Flat metric dict. Token rates are None when the wall time or token count is zero.
    """
    summary: dict[str, Any] = {
        "num_trajectories": len(records),
        "num_prompts": len({r["uid"] for r in records}),
        "num_failed_prompts": num_failed_prompts,
        "wall_time_s": wall_time_s,
    }
    if not records:
        return summary

    prompt_len = np.array([r["prompt_len"] for r in records], dtype=np.float64)
    response_len = np.array([r["response_len"] for r in records], dtype=np.float64)
    generated = np.array([r["generated_tokens"] for r in records], dtype=np.float64)
    total_generated = float(generated.sum())
    total_tokens = float(prompt_len.sum() + response_len.sum())
    has_time = wall_time_s > 0

    summary.update(
        {
            "tokens/generated_total": int(total_generated),
            "tokens/tool_response_total": int(response_len.sum() - total_generated),
            "tokens/prompt_total": int(prompt_len.sum()),
            "tokens/total": int(total_tokens),
            "throughput/generated_tokens_per_s": total_generated / wall_time_s if has_time else None,
            "throughput/total_tokens_per_s": total_tokens / wall_time_s if has_time else None,
            "throughput/trajectories_per_s": len(records) / wall_time_s if has_time else None,
            "latency/ms_per_generated_token": 1000.0 * wall_time_s / total_generated if total_generated else None,
            # Same definition as the recipe report's rollout "ms / tok": step tokens (prompt + response).
            "latency/ms_per_total_token": 1000.0 * wall_time_s / total_tokens if total_tokens else None,
            "response_length/clip_ratio": float((response_len >= max_response_length).mean()),
        }
    )
    summary.update(_percentiles(response_len, "response_length"))
    summary.update(_percentiles(generated, "generated_tokens"))
    summary.update(_percentiles(np.array([r["num_turns"] for r in records], dtype=np.float64), "num_turns"))
    # Seconds from dispatch until each trajectory was stored: the batch wall time is set by the slowest ones.
    done_s = np.array([r["done_s"] for r in records if r.get("done_s") is not None], dtype=np.float64)
    summary.update(_percentiles(done_s, "completion_s"))
    summary["score/mean"] = float(np.mean([r["score"] for r in records]))

    optional_means = {
        "acc/mean": "acc",
        "timing/generate_s/mean": "generate_s",
        "timing/tool_calls_s/mean": "tool_calls_s",
        "prefix_cache/cached_tokens/mean": "num_cached_tokens",
    }
    for metric_name, field in optional_means.items():
        values = [float(r[field]) for r in records if r.get(field) is not None]
        if values:
            summary[metric_name] = float(np.mean(values))
    for field in ("generate_s", "tool_calls_s"):
        values = [float(r[field]) for r in records if r.get(field) is not None]
        if values:
            summary[f"timing/{field}/max"] = float(np.max(values))
    return summary


def compute_val_metrics(
    records: list[dict[str, Any]], expected_acc_counts: dict[tuple[str, str], int]
) -> dict[str, float]:
    """Validation metrics in the trainer's naming (``val-core/<data_source>/acc/mean@30``, ...).

    Mirrors ``PPOTrainer._validate`` + ``PPOTrainer._val_metrics_update`` so numbers line up with the recipe's WandB
    validation curves.
    """
    if not records:
        return {}
    reward_extra_infos: dict[str, list] = {"reward": [r["score"] for r in records]}
    extra_keys = sorted({key for r in records for key in r["reward_extra_info"]})
    for key in extra_keys:
        reward_extra_infos[key] = [r["reward_extra_info"].get(key) for r in records]

    data_src2var2metric2val = process_validation_metrics(
        [r["data_source"] for r in records],
        [r["uid"] for r in records],
        reward_extra_infos,
        expected_acc_counts=expected_acc_counts,
    )
    metric_dict: dict[str, float] = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val)
            for metric_name, metric_val in metric2val.items():
                is_core = (
                    var_name == core_var
                    and any(metric_name.startswith(pfx) for pfx in ("mean", "maj", "best"))
                    and f"@{n_max}" in metric_name
                )
                section = "val-core" if is_core else "val-aux"
                metric_dict[f"{section}/{data_source}/{var_name}/{metric_name}"] = float(metric_val)

    turns = np.array([r["num_turns"] for r in records], dtype=np.float64)
    metric_dict["val-aux/num_turns/min"] = float(turns.min())
    metric_dict["val-aux/num_turns/max"] = float(turns.max())
    metric_dict["val-aux/num_turns/mean"] = float(turns.mean())
    return metric_dict


def _unwrap(value: Any) -> Any:
    """TransferQueue returns non-tensor values either raw or wrapped in ``NonTensorData``."""
    return getattr(value, "data", value)


def _as_list(column: Any) -> list:
    """Python values of one TensorDict column.

    ``td[key]`` gives a ``LinkedList`` for non-tensor columns while ``td.pop(key)`` gives a ``NonTensorStack``; tensors
    and ``NonTensorStack`` have ``tolist()``, anything else is iterated and unwrapped.
    """
    if hasattr(column, "tolist"):
        return column.tolist()
    return [_unwrap(value) for value in column]


def records_from_trajectories(
    keys: list[str], data: TensorDict, batch_idx: int, tokenizer: Any = None
) -> list[dict[str, Any]]:
    """One flat record per trajectory from a ``tq.kv_batch_get(keys, select_fields=TRAJECTORY_FIELDS)`` result.

    Keys follow the ``{uid}_{session_id}_{index}`` layout written by ``AgentLoopWorkerTQ``. Prompt / response text is
    decoded only when a tokenizer is given.
    """
    prompts, responses = list(data["prompts"]), list(data["responses"])
    response_masks, rm_scores = list(data["response_mask"]), list(data["rm_scores"])
    num_turns = _as_list(data["num_turns"])
    data_sources = _as_list(data["data_source"])
    reward_models = _as_list(data["reward_model"])
    extra_fields = _as_list(data["extra_fields"])
    metrics = _as_list(data["metrics"])
    records = []
    for i, key in enumerate(keys):
        uid, session_id, _ = key.rsplit("_", 2)
        extra = _unwrap(extra_fields[i]) or {}
        reward_extra_info = extra.get("reward_extra_info", {}) or {}
        metric = _unwrap(metrics[i]) or {}
        reward_model = _unwrap(reward_models[i]) or {}
        acc = reward_extra_info.get("acc")
        record = {
            "batch": batch_idx,
            "key": key,
            "uid": uid,
            "session_id": int(session_id),
            "data_source": str(_unwrap(data_sources[i])),
            "ground_truth": reward_model.get("ground_truth"),
            "prompt_len": int(prompts[i].numel()),
            "response_len": int(responses[i].numel()),
            "generated_tokens": int(response_masks[i].sum().item()),
            "num_turns": int(_unwrap(num_turns[i])),
            "score": float(rm_scores[i].sum().item()),
            "acc": None if acc is None else float(acc),
            "pred": reward_extra_info.get("pred"),
            "reward_extra_info": reward_extra_info,
            "generate_s": metric.get("generate_sequences"),
            "tool_calls_s": metric.get("tool_calls"),
            "num_preempted": metric.get("num_preempted"),
            "num_cached_tokens": extra.get("num_cached_tokens"),
        }
        if tokenizer is not None:
            record["prompt"] = tokenizer.decode(prompts[i], skip_special_tokens=True)
            record["response"] = tokenizer.decode(responses[i], skip_special_tokens=True)
        records.append(record)
    return records


class RolloutBenchmark:
    """Owns the rollout objects of the recipe and times batches through them."""

    def __init__(self, config: DictConfig):
        self.config = config
        bench = config.get("benchmark", None) or {}
        self.mode = str(bench.get("mode", "train"))
        if self.mode not in ("train", "val"):
            raise ValueError(f"benchmark.mode must be 'train' or 'val', got {self.mode!r}")
        self.num_batches = int(bench.get("num_batches", 1)) if self.mode == "train" else 1
        max_prompts = bench.get("max_prompts", None)
        self.max_prompts = int(max_prompts) if max_prompts is not None else None
        self.dump_text = bool(bench.get("dump_text", True))
        self.poll_interval = float(bench.get("poll_interval", 1.0))
        self.output_dir = bench.get("output_dir", None) or os.path.join(
            "/tmp/rollout_benchmark", f"{self.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        self.partition_id = "train" if self.mode == "train" else "val"
        self.rollout_config = config.actor_rollout_ref.rollout

    # ------------------------------------------------------------------ setup (mirrors the trainer)

    def init(self) -> None:
        """Build tokenizer, dataloader, reward workers, vLLM replicas and agent loop workers like the trainer."""
        # PPOTrainerSeparateAsync.__init__: place rollout replicas on the rollout TPU slices.
        if get_platform().device_name == "tpu":
            resolve_tpu_rollout_layout(self.config)

        # PPOTrainer._init_tokenizer
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model)
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor
        self._init_dataloader()

        # PPOTrainer._setup step 7: rule-based streaming reward (custom_reward_function=compute_score).
        self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=None)

        # PPOTrainerSeparateAsync._setup: standalone replicas only (hybrid_engine=False -> no hybrid replicas, so
        # the standalone ranks start at 0).
        start = time.perf_counter()
        self.server_manager = LLMServerManager.create(config=self.config, start_rank=0)
        self.server_init_s = time.perf_counter() - start

        # TaskRunnerV1.init_agent_loop_manager with PPOTrainerSeparateAsync.get_llm_client.
        self.agent_loop_manager = AgentLoopManagerTQ.create(
            config=self.config,
            llm_client=self.server_manager.get_client(client_cls=FullyAsyncLLMServerClient),
            teacher_client=None,
            reward_loop_worker_handles=self.reward_loop_manager.reward_loop_worker_handles,
        )

    def _init_dataloader(self) -> None:
        """Same datasets, samplers and batch sizes as ``PPOTrainer._init_dataloader``."""
        data_config = self.config.data
        if self.mode == "train":
            dataset = create_rl_dataset(
                data_config.train_files,
                data_config,
                self.tokenizer,
                self.processor,
                is_train=True,
                max_samples=data_config.get("train_max_samples", -1),
            )
            # Async trainer modes fetch one prompt at a time (exact refill counts), see _init_dataloader.
            if self.config.trainer.v1.trainer_mode != "sync":
                with open_dict(self.config):
                    self.config.data.gen_batch_size = 1
            gen_batch_size = data_config.get("gen_batch_size", None) or data_config.train_batch_size
            self.dataloader = StatefulDataLoader(
                dataset=dataset,
                batch_size=gen_batch_size,
                num_workers=data_config["dataloader_num_workers"],
                drop_last=True,
                collate_fn=collate_fn,
                sampler=create_rl_sampler(data_config, dataset),
            )
        else:
            dataset = create_rl_dataset(
                data_config.val_files,
                data_config,
                self.tokenizer,
                self.processor,
                is_train=False,
                max_samples=data_config.get("val_max_samples", -1),
            )
            self.dataloader = StatefulDataLoader(
                dataset=dataset,
                batch_size=data_config.val_batch_size or len(dataset),
                num_workers=data_config["dataloader_num_workers"],
                shuffle=data_config.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
            # PPOTrainer._validate walks the whole val dataloader once (a single batch unless val_batch_size is set).
            self.num_batches = len(self.dataloader)
        self.dataloader_it = iter(self.dataloader)
        logger.info(f"[RolloutBenchmark] {self.mode} dataset size: {len(dataset)}")

    # ------------------------------------------------------------------ batches (mirror the trainer data flow)

    def _fetch_one(self) -> TensorDict:
        """PPOTrainer._fetch_one_gen_batch."""
        try:
            batch_dict = next(self.dataloader_it)
        except StopIteration:
            self.dataloader_it = iter(self.dataloader)
            batch_dict = next(self.dataloader_it)
        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        return tu.get_tensordict(batch_dict)

    def _next_batch(self, global_steps: int) -> TensorDict:
        """PPOTrainer._next_train_batch (train) or one val dataloader batch (PPOTrainer._validate)."""
        if self.mode == "train":
            num_prompts = self.config.data.train_batch_size
            if self.max_prompts is not None:
                num_prompts = min(num_prompts, self.max_prompts)
            gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
            if num_prompts % gen_batch_size != 0:
                raise ValueError(f"num_prompts={num_prompts} must be a multiple of gen_batch_size={gen_batch_size}")
            chunks = [self._fetch_one() for _ in range(num_prompts // gen_batch_size)]
            batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        else:
            batch = self._fetch_one()
            if self.max_prompts is not None and len(batch) > self.max_prompts:
                batch = batch[: self.max_prompts]
            tu.assign_non_tensor_data(batch, "validate", True)
        tu.assign_non_tensor_data(batch, "global_steps", global_steps)
        return batch

    def _submit(self, batch: TensorDict, global_steps: int) -> list[str]:
        """PPOTrainer._submit_batch_to_rollout (train) / the submit half of PPOTrainer._validate (val)."""
        uids = list(batch["uid"])
        tags = [{"is_prompt": True, "status": "pending", "global_steps": global_steps} for _ in uids]
        if self.partition_id == "train":
            fields = batch.select(*[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)])
            tq.kv_batch_put(keys=uids, partition_id="train", tags=tags, fields=fields)
        else:
            tq.kv_batch_put(keys=uids, partition_id="val", tags=tags)
        self.agent_loop_manager.generate_sequences(batch)
        return uids

    def _wait(self, uids: list[str], start: float) -> tuple[set[str], set[str], dict[str, float]]:
        """Poll TransferQueue until every prompt group is ``finished`` or ``failure``.

        Returns the finished and failed prompt uids, and for every stored trajectory key the seconds since ``start``
        at which it first appeared (``poll_interval`` resolution), which gives the completion-time tail.
        """
        uid_set = set(uids)
        pending, finished, failed = set(uids), set(), set()
        first_seen: dict[str, float] = {}
        last_log = time.perf_counter()
        while pending:
            items = (tq.kv_list(self.partition_id) or {}).get(self.partition_id, {})
            now = time.perf_counter()
            for key, tag in items.items():
                if key not in first_seen and not tag.get("is_prompt", False) and key.split("_")[0] in uid_set:
                    first_seen[key] = now - start
            for uid in list(pending):
                status = items.get(uid, {}).get("status")
                if status in ("finished", "failure"):
                    pending.discard(uid)
                    (finished if status == "finished" else failed).add(uid)
            if not pending:
                break
            if now - last_log > PROGRESS_LOG_INTERVAL_S:
                print(
                    f"[RolloutBenchmark] {now - start:.0f}s: prompts finished={len(finished)} failed={len(failed)} "
                    f"pending={len(pending)}, trajectories stored={len(first_seen)}",
                    flush=True,
                )
                last_log = now
            time.sleep(self.poll_interval)
        return finished, failed, first_seen

    def _collect(self, uids: list[str], batch_idx: int) -> list[dict[str, Any]]:
        """Read every trajectory of ``uids`` back from TransferQueue, then clear them."""
        uid_set = set(uids)
        items = (tq.kv_list(self.partition_id) or {}).get(self.partition_id, {})
        keys = sorted(
            (key for key, tag in items.items() if not tag.get("is_prompt", False) and key.split("_")[0] in uid_set),
            key=lambda k: (k.rsplit("_", 2)[0], int(k.rsplit("_", 2)[1]), int(k.rsplit("_", 2)[2])),
        )
        records: list[dict[str, Any]] = []
        for start in range(0, len(keys), COLLECT_CHUNK_SIZE):
            chunk = keys[start : start + COLLECT_CHUNK_SIZE]
            data = tq.kv_batch_get(keys=chunk, partition_id=self.partition_id, select_fields=TRAJECTORY_FIELDS)
            records.extend(
                records_from_trajectories(chunk, data, batch_idx, self.tokenizer if self.dump_text else None)
            )
        tq.kv_clear(keys=[*uids, *keys], partition_id=self.partition_id)
        return records

    # ------------------------------------------------------------------ driver

    def _key_params(self) -> dict[str, Any]:
        """Resolved rollout parameters, so a run can be checked against the recipe at a glance."""
        rollout = self.rollout_config
        sampling = rollout.val_kwargs if self.mode == "val" else rollout
        n = rollout.val_kwargs.n if self.mode == "val" else rollout.n
        return {
            "model_path": self.config.actor_rollout_ref.model.path,
            "mode": self.mode,
            "n": n,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "calculate_log_probs": rollout.calculate_log_probs,
            "prompt_length": rollout.prompt_length,
            "response_length": rollout.response_length,
            "tensor_model_parallel_size": rollout.tensor_model_parallel_size,
            "nnodes": rollout.nnodes,
            "n_gpus_per_node": rollout.n_gpus_per_node,
            "num_replicas": len(self.server_manager.get_replicas()),
            "max_model_len": rollout.max_model_len,
            "max_num_seqs": rollout.max_num_seqs,
            "max_num_batched_tokens": rollout.max_num_batched_tokens,
            "gpu_memory_utilization": rollout.gpu_memory_utilization,
            "enable_prefix_caching": rollout.enable_prefix_caching,
            "enable_chunked_prefill": rollout.enable_chunked_prefill,
            "enforce_eager": rollout.enforce_eager,
            "dtype": rollout.dtype,
            "load_format": rollout.load_format,
            "multi_turn.max_user_turns": rollout.multi_turn.max_user_turns,
            "multi_turn.max_assistant_turns": rollout.multi_turn.max_assistant_turns,
            "multi_turn.format": rollout.multi_turn.format,
            "multi_turn.tool_config_path": rollout.multi_turn.tool_config_path,
            "agent.num_workers": rollout.agent.num_workers,
            "enable_thinking": self.config.data.get("apply_chat_template_kwargs", {}).get("enable_thinking"),
        }

    def run(self) -> dict[str, Any]:
        """Run the configured batches and write ``summary.json`` / ``trajectories.jsonl``."""
        os.makedirs(self.output_dir, exist_ok=True)
        result: dict[str, Any] = {
            "host": socket.gethostname(),
            "output_dir": self.output_dir,
            "server_init_s": self.server_init_s,
            "params": self._key_params(),
            "batches": [],
        }
        trajectories_path = os.path.join(self.output_dir, "trajectories.jsonl")
        with open(trajectories_path, "w", encoding="utf-8") as traj_file:
            for batch_idx in range(self.num_batches):
                global_steps = 0 if self.mode == "val" else batch_idx + 1
                batch = self._next_batch(global_steps)
                rollout_n = self.rollout_config.val_kwargs.n if self.mode == "val" else self.rollout_config.n
                data_sources = batch["data_source"] if "data_source" in batch.keys() else ["unknown"] * len(batch)
                expected_acc_counts = {
                    (str(_unwrap(ds)), str(_unwrap(uid))): int(rollout_n)
                    for ds, uid in zip(data_sources, batch["uid"], strict=True)
                }
                print(
                    f"[RolloutBenchmark] batch {batch_idx}: dispatching {len(batch)} prompts x n={rollout_n}",
                    flush=True,
                )
                start = time.perf_counter()
                uids = self._submit(batch, global_steps)
                _, failed, first_seen = self._wait(uids, start)
                wall_time_s = time.perf_counter() - start
                records = self._collect(uids, batch_idx)
                for record in records:
                    record["done_s"] = first_seen.get(record["key"])

                summary = summarize_trajectories(
                    records,
                    wall_time_s,
                    int(self.rollout_config.response_length),
                    num_failed_prompts=len(failed),
                )
                summary["batch"] = batch_idx
                if self.mode == "val":
                    summary.update(compute_val_metrics(records, expected_acc_counts))
                result["batches"].append(summary)
                print(f"[RolloutBenchmark] batch {batch_idx} summary: {json.dumps(summary, indent=2)}", flush=True)

                for record in records:
                    record.pop("reward_extra_info", None)
                    traj_file.write(json.dumps(record, default=str) + "\n")

        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[RolloutBenchmark] wrote {summary_path} and {trajectories_path} on host {result['host']}", flush=True)
        return result


@ray.remote
class RolloutBenchmarkTaskRunner:
    """Driver actor, scheduled by ``run_ppo`` exactly like ``TaskRunnerV1``."""

    def run(self, config: DictConfig) -> Optional[dict[str, Any]]:
        configure_verl_logging()
        # Same preamble as TaskRunnerV1.run.
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        try:
            benchmark = RolloutBenchmark(config)
            benchmark.init()
            return benchmark.run()
        finally:
            tq.close()


@hydra.main(config_path=VERL_TRAINER_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    main_ppo.run_ppo(config, task_runner_class=RolloutBenchmarkTaskRunner)


if __name__ == "__main__":
    main()

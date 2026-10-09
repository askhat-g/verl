# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Google LLC
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

import asyncio
import gc
import logging
import os
import time
from typing import Any, Generator, Iterable, Optional

import ray
import torch

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
)

logger = logging.getLogger(__name__)


def compute_tensor_stats(items) -> dict:
    """Compute deterministic L1/L2 norms and parameter counts across tensors.
    Reductions are performed directly on device (TPU) to avoid copying full tensor weights across PCIe.
    """
    total_numel = 0
    total_l1 = 0.0
    total_l2_sq = 0.0
    per_tensor = {}

    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            name, p = item
        else:
            name = None
            p = item
        p_local = p.to_local() if hasattr(p, "to_local") else p

        # Perform reductions directly on device (TPU HBM) in float32 for precision and speed
        p_float = p_local.float()
        t_l1 = float(p_float.abs().sum().item())
        t_l2_sq = float(p_float.pow(2).sum().item())
        numel = p_local.numel()

        if name is not None:
            per_tensor[name] = {
                "l1": t_l1,
                "l2": float(t_l2_sq**0.5),
                "l2_sq": t_l2_sq,
                "numel": numel,
                "shape": list(p_local.shape),
                "dtype": str(p_local.dtype),
            }
        total_numel += numel
        total_l1 += t_l1
        total_l2_sq += t_l2_sq

    return {
        "total_numel": total_numel,
        "num_tensors": len(items),
        "l1_norm": total_l1,
        "l2_norm": float(total_l2_sq**0.5),
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


def create_torch_weight_synchronizer(
    device_tensors: list[list[torch.Tensor]],
    local_port: int = 0,
    parallelism: int = 8,
    listener_port: int = 0,
    bind_ip: str = "127.0.0.1",
):
    """Creates an instance of WeightSynchronizer using tpu_sync."""
    from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer

    return WeightSynchronizer(
        device_tensors,
        local_port=local_port,
        parallelism=parallelism,
        listener_port=listener_port,
        bind_ip=bind_ip,
        unsafe_skip_buffer_lock=True,
        auto_h2d=False,
    )


def _unwrap_tensor(t: Any) -> Any:
    """Extract the raw underlying local tensor from nn.Parameter or DTensor wrappers."""
    t = t.to_local().data if hasattr(t, "to_local") else (t.data if hasattr(t, "data") else t)
    return t


def filter_tied_embeddings(
    named_items: Iterable[tuple[str, Any]], tie_word_embeddings: bool = True
) -> list[tuple[str, Any]]:
    """Drop ``lm_head.weight`` from the weights to send when it is tied to the input embedding.

    For tied models (``tie_word_embeddings=True``, e.g. Qwen3-0.6B / 4B), ``lm_head.weight`` is the same
    tensor as ``embed_tokens.weight``, so it is not sent; the sampler copies ``embed_tokens`` into its
    ``lm_head`` after receiving the weights.

    For untied models (``tie_word_embeddings=False``, e.g. Qwen3-8B and larger), ``lm_head.weight`` is a
    separate trained weight and is kept. Dropping it would leave the sampler with a wrong ``lm_head``
    (stale, or overwritten with ``embed_tokens``).

    Args:
        named_items: ``(name, tensor)`` pairs in HF naming.
        tie_word_embeddings: ``hf_config.tie_word_embeddings`` of the trained model.

    Returns:
        The ``(name, tensor)`` pairs to send.
    """
    items = list(named_items)
    has_embed = any("embed_tokens" in k or "tok_embeddings" in k for k, _ in items)
    if has_embed and tie_word_embeddings:
        items = [(k, v) for k, v in items if not (k == "lm_head.weight" or k.endswith(".lm_head.weight"))]
    return items


def validate_and_sanitize_tensors(
    named_tensors: Iterable[tuple[str, Any]],
    device: Optional[torch.device] = None,
) -> list[tuple[str, torch.Tensor]]:
    """Validate and sanitize tensors for zero-copy DMA registration with Raiden.

    Performs physical memory validation:
    1. Drops None, non-tensor objects, zero-element tensors, and unallocated meta tensors.
    2. Unwraps DTensor/nn.Parameter to local tensor buffers.
    3. Ensures tensors physically reside in TPU HBM.
    4. Guarantees memory contiguity (contiguous buffers) required for direct DMA.
    """
    if device is None:
        device = torch.device("tpu")

    sanitized = []
    for item in named_tensors:
        name, p = item[0], item[1]
        if p is None:
            continue
        t = _unwrap_tensor(p)
        if not isinstance(t, torch.Tensor):
            continue

        # Skip 0-element tensors and unallocated meta tensors (prevents C++ nullptr faults)
        if t.numel() == 0 or getattr(t, "is_meta", False):
            continue

        # Ensure tensor physically resides in TPU device memory
        if not (hasattr(t, "device") and str(t.device).startswith("tpu")):
            try:
                t = t.to(device)
            except Exception as e:
                logger.warning(f"Could not move {name} to TPU: {e}")
                continue

        # Ensure contiguous memory layout (required for direct DMA pointer calculation)
        if not t.is_contiguous():
            t = t.contiguous()

        sanitized.append((name, t))

    return sanitized


def _dim0_shard_info(spec) -> Optional[tuple[tuple, int, int]]:
    """Return ``(local_shape, num_shards, shard_index)`` if the local shard is an even dim-0 cut on a 1-D mesh.

    That is the FSDP2 ``Shard(0)`` layout, which Raiden can describe directly: the variable keeps its full
    shape, is split ``num_shards`` ways on dim 0, and this rank holds block ``shard_index``. Uneven cuts
    are excluded because torch gives the remainder to the leading shards while Raiden gives it to the last.
    """
    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    if spec.mesh is None or spec.mesh.ndim != 1:
        return None
    place, _, _ = derive_dtensor_placement(spec)
    if not isinstance(place, BlockPlacement) or not place.is_flat_contiguous:
        return None
    num_shards = spec.mesh.size(0)
    full_dim0 = int(spec.full_shape[0])
    if num_shards <= 1 or full_dim0 % num_shards:
        return None
    return tuple(place.local_shape), num_shards, int(place.global_offset[0]) // (full_dim0 // num_shards)


def export_local_shards(
    engine, tie_word_embeddings: bool = True
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, tuple[int, int]]]:
    """Export this rank's weights for Raiden without all-gathering FSDP shards.

    Returns ``(named_tensors, shard_info)``. ``shard_info[name] = (num_shards, shard_index)`` for tensors
    exported as a local dim-0 shard; tensors absent from it are full (replicated) on every rank. Tensors
    whose layout Raiden cannot express are all-gathered here, in the same order on every rank.
    ``lm_head.weight`` is dropped only for tied models (see ``filter_tied_embeddings``).
    """
    from torch.distributed.tensor import DTensor

    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    gen, _ = engine.get_per_tensor_param_shard()
    items = filter_tied_embeddings(
        ((name, (local, spec)) for name, local, spec in gen), tie_word_embeddings=tie_word_embeddings
    )

    named, shard_info = [], {}
    for name, (local, spec) in items:
        if spec.place is not None or spec.hf_slots is not None:
            raise NotImplementedError(
                f"Raiden sharded export does not support exporter-defined placements or expert stacks ({name})"
            )
        full_shape = tuple(int(d) for d in spec.full_shape)
        info = _dim0_shard_info(spec)
        if info is not None:
            local_shape, num_shards, shard_index = info
            named.append((name, local.view(local_shape)))
            shard_info[name] = (num_shards, shard_index)
            continue
        place = derive_dtensor_placement(spec)[0] if spec.mesh is not None else 0
        if not isinstance(place, BlockPlacement):
            # unsharded or fully replicated: the local tensor is already the full parameter
            named.append((name, local.view(full_shape)))
            continue
        strides = [1] * len(full_shape)
        for d in range(len(full_shape) - 2, -1, -1):
            strides[d] = strides[d + 1] * full_shape[d + 1]
        dt = DTensor.from_local(
            local.view(place.local_shape),
            spec.mesh,
            spec.placements,
            run_check=False,
            shape=torch.Size(full_shape),
            stride=tuple(strides),
        )
        named.append((name, dt.full_tensor()))
    return named, shard_info


def merge_rank_stats(rank_stats: dict) -> dict:
    """Sum per-tensor partial stats (l1, l2_sq, numel) posted by every trainer rank."""
    per_tensor = {}
    for stats in rank_stats.values():
        for name, t in stats.get("per_tensor", {}).items():
            acc = per_tensor.setdefault(name, {"l1": 0.0, "l2_sq": 0.0, "numel": 0})
            acc["l1"] += t["l1"]
            acc["l2_sq"] += t["l2_sq"]
            acc["numel"] += t["numel"]
    for acc in per_tensor.values():
        acc["l2"] = acc["l2_sq"] ** 0.5
    total_l2_sq = sum(t["l2_sq"] for t in per_tensor.values())
    return {
        "total_numel": sum(t["numel"] for t in per_tensor.values()),
        "num_tensors": len(per_tensor),
        "l1_norm": sum(t["l1"] for t in per_tensor.values()),
        "l2_norm": total_l2_sq**0.5,
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


class RaidenParityCheck:
    """Raiden weight-sync verification, selected by ``engine_kwargs.raiden.verify_parity``.

    Two complementary checks:

    * ``"norm"`` (or ``True``): on every sync from step 1, the trainer ranks post per-tensor L1/L2 stats and the
      orchestrator compares them with the samplers' stats after the install (``verify_norms``). Cheap, but blind to
      misplaced bytes: a reordered or mis-sliced tensor keeps its norms.
    * ``"exact"``: on the step-0 sync of a fresh run, when the trainer weights equal the checkpoint vLLM loaded,
      every sampler worker compares each received tensor bit for bit with the vLLM parameter it is about to
      overwrite (``check_exact``) and returns the result (``exact_result``); the orchestrator logs one summary for
      all ranks (``report_exact``). Catches any wrong byte (shard slice, tiling, fusion/transpose). Use it after
      changing sharding or upgrading tpu-sync / torch-tpu. Cost: ~0.4 s once per rank (Qwen3-32B, TP32).

    ``"all"`` enables both; ``False`` / ``"off"`` (default) neither.
    """

    MODES = ("off", "norm", "exact", "all")
    MAX_DETAIL_LINES = 60

    def __init__(self, mode: Any = "off") -> None:
        self.mode = self.parse_mode(mode)
        self.checked = 0
        self.mismatched: list[str] = []

    @classmethod
    def parse_mode(cls, value: Any) -> str:
        if value is None or value is False:
            return "off"
        if value is True:
            return "norm"
        mode = str(value).strip().lower()
        mode = {"false": "off", "0": "off", "none": "off", "true": "norm", "1": "norm"}.get(mode, mode)
        if mode not in cls.MODES:
            raise ValueError(f"verify_parity must be a bool or one of {cls.MODES}, got {value!r}")
        return mode

    @classmethod
    def from_config(cls, engine_kwargs) -> "RaidenParityCheck":
        """Read ``engine_kwargs.raiden.verify_parity``, the same key the trainer's RaidenCheckpointEngine receives."""
        return cls((engine_kwargs.get("raiden") or {}).get("verify_parity", False))

    @property
    def norm(self) -> bool:
        return self.mode in ("norm", "all")

    @property
    def exact(self) -> bool:
        return self.mode in ("exact", "all")

    # ---- exact check: orchestrator side ----

    def exact_install_kwargs(self, global_steps: Optional[int]) -> dict:
        """kwargs for the samplers' install_raiden_weights RPC: request the exact check on the step-0 sync only.

        Later syncs (and the first sync after resuming from a checkpoint) carry trained weights, which no longer
        match what vLLM loaded, so the exact check cannot run there.
        """
        return {"exact_parity": True} if self.exact and not global_steps else {}

    @classmethod
    def report_exact(cls, global_steps: Optional[int], install_results: list) -> bool:
        """Log one summary of the per-rank ``exact_result`` dicts that every sampler worker returns under
        ``install_raiden_weights(...)["exact_parity"]``."""
        results = [
            w["exact_parity"]
            for res in install_results
            for w in (res if isinstance(res, list | tuple) else [res])
            if isinstance(w, dict) and isinstance(w.get("exact_parity"), dict)
        ]
        step = global_steps or 0
        if not results:
            logger.warning(f"[RAIDEN PARITY EXACT | Step {step}] no sampler returned a result")
            return False
        results.sort(key=lambda r: r["rank"])
        num_mismatched = sum(r["num_mismatched"] for r in results)
        num_unresolved = sum(len(r["unresolved"]) for r in results)
        checked_per_rank = sorted({r["checked"] for r in results})
        summary = (
            f"[RAIDEN PARITY EXACT | Step {step}] {len(results)} sampler ranks, {checked_per_rank} tensors checked "
            f"per rank, {num_mismatched} mismatched, {num_unresolved} unresolved"
        )
        if num_mismatched == 0 and num_unresolved == 0:
            logger.info(f"{summary}: every received tensor matches the checkpoint bit for bit")
            return True
        details = [f"  * rank {r['rank']}: {line}" for r in results for line in r["mismatched"]]
        details += [f"  * rank {r['rank']}: unresolved {r['unresolved'][:5]}" for r in results if r["unresolved"]]
        detail_text = "\n".join(details[: cls.MAX_DETAIL_LINES])
        logger.error(f"{summary}\n{detail_text}")
        return False

    # ---- exact check: sampler worker side ----

    @torch.no_grad()
    def check_exact(self, name: str, target: torch.Tensor, received: torch.Tensor) -> None:
        """Compare ``received`` with ``target``; call before ``target`` is overwritten."""
        try:
            ref, new = target.detach().float(), received.detach().float()
            if ref.shape != new.shape:
                self.mismatched.append(f"{name}: shape target={tuple(ref.shape)} received={tuple(new.shape)}")
                return
            self.checked += 1
            diff = (ref - new).abs().max().item()
            if diff != 0.0:
                self.mismatched.append(
                    f"{name}: max|diff|={diff:.4g} ref_absmean={ref.abs().mean().item():.4g} "
                    f"recv_absmean={new.abs().mean().item():.4g}"
                )
        except Exception as e:  # never break the sync because of the check
            self.mismatched.append(f"{name}: parity check failed: {e}")

    def exact_result(self, rank: int, unresolved: list[str]) -> dict:
        return {
            "rank": rank,
            "checked": self.checked,
            "num_mismatched": len(self.mismatched),
            "mismatched": self.mismatched[: self.MAX_DETAIL_LINES],
            "unresolved": list(unresolved),
        }

    # ---- norm check: orchestrator side ----

    async def verify_norms(self, manager, global_steps: Optional[int] = None) -> None:
        """Compare distributed norms between Trainer Rank 0 and Sampler TP workers.

        TODO(tpu): Refactor parity verification to use rank-local scalar partitioning.
        Instead of collecting per-tensor dictionaries across all Sampler workers and checking
        replicated vs sharded heuristics over 200+ tensors, each worker can reduce its model
        locally into scalar metrics (numel, l1, l2_sq) before RPC return:
          - Rank 0 accumulates both sharded tensors and replicated 1D tensors (e.g. RMSNorms).
          - Ranks 1..N-1 accumulate only sharded tensors.
        The orchestrator can then perform verification in O(ranks) pure scalar arithmetic
        rather than O(tensors * ranks) loop aggregation.
        """
        step_key = global_steps if global_steps is not None else 0
        if step_key <= 0:
            return

        try:
            registry = ray.get_actor("TPUWeightRegistry", namespace="verl")
            num_trainer_ranks = manager.actor_wg.world_size
            trainer_entry = None
            for _ in range(25):
                entry = await registry.get_stats.remote(step_key)
                # Sharded export: every trainer rank posts partial stats, usable once all have arrived.
                if entry is not None and ("ranks" not in entry or len(entry["ranks"]) >= num_trainer_ranks):
                    trainer_entry = entry
                    break
                await asyncio.sleep(0.5)

            sampler_futures = [
                replica.server_handle.collective_rpc.remote(
                    method="get_model_weights_stats", kwargs={"include_shards": False}
                )
                for replica in manager.replicas
            ]
            sampler_entries = await asyncio.gather(*sampler_futures)

            if not trainer_entry or not sampler_entries:
                logger.warning(f"[RAIDEN PARITY] Incomplete stats data for step {step_key}")
                return

            # Unpacks and flattens the results collected from all Sampler rollout replicas into a single flat list
            # of worker dictionary objects.
            sampler_workers = [
                w
                for res in sampler_entries
                for w in (res if isinstance(res, list | tuple) else [res])
                if isinstance(w, dict)
            ]
            if not sampler_workers:
                return

            if "ranks" in trainer_entry:
                trainer_master = merge_rank_stats(trainer_entry["ranks"])
            else:
                trainer_master = trainer_entry.get("master", trainer_entry)
            trainer_per_tensor = trainer_master.get("per_tensor", {})
            # Gets master list of all model tensor names (e.g., "model.layers.0.self_attn.qkv_proj.weight").
            all_param_names = list(sampler_workers[0].get("per_tensor", {}).keys()) if sampler_workers else []

            total_trainer_numel = trainer_master.get("total_numel", 0)
            total_trainer_l1 = trainer_master.get("l1_norm", 0.0)
            global_trainer_l2 = trainer_master.get("l2_norm", 0.0)

            total_sampler_l1, total_sampler_l2_sq, total_sampler_numel = 0.0, 0.0, 0
            mismatches = []

            for name in all_param_names:
                s_numels = [w.get("per_tensor", {}).get(name, {}).get("numel", 0) for w in sampler_workers]
                s_l1s = [w.get("per_tensor", {}).get(name, {}).get("l1", 0.0) for w in sampler_workers]
                s_l2_sqs = [
                    w.get("per_tensor", {})
                    .get(name, {})
                    .get("l2_sq", w.get("per_tensor", {}).get(name, {}).get("l2", 0.0) ** 2)
                    for w in sampler_workers
                ]
                s_replicas = [w.get("per_tensor", {}).get(name, {}).get("replicas") for w in sampler_workers]

                is_replicated = (
                    len(set(s_numels)) == 1
                    and (name.endswith("layernorm.weight") or "norm" in name)
                    and len(s_numels[0:1]) > 0
                    and s_numels[0] < 10000
                )
                if all(r for r in s_replicas):
                    # Each slice is held by `replicas` ranks (GQA k/v heads, unsharded norms): count it once.
                    s_agg_numel = round(sum(n / r for n, r in zip(s_numels, s_replicas, strict=True)))
                    s_agg_l1 = sum(v / r for v, r in zip(s_l1s, s_replicas, strict=True))
                    s_agg_l2 = sum(v / r for v, r in zip(s_l2_sqs, s_replicas, strict=True)) ** 0.5
                elif is_replicated:
                    s_agg_numel = s_numels[0]
                    s_agg_l1 = s_l1s[0]
                    s_agg_l2 = s_l2_sqs[0] ** 0.5
                else:
                    s_agg_numel = sum(s_numels)
                    s_agg_l1 = sum(s_l1s)
                    s_agg_l2 = sum(s_l2_sqs) ** 0.5

                t_data = trainer_per_tensor.get(name, {})
                t_agg_numel = t_data.get("numel", 0)
                t_agg_l1 = t_data.get("l1", 0.0)

                total_sampler_numel += s_agg_numel
                total_sampler_l1 += s_agg_l1
                total_sampler_l2_sq += s_agg_l2**2

                delta_numel = abs(s_agg_numel - t_agg_numel)
                delta_l1 = abs(s_agg_l1 - t_agg_l1)
                rel_tol = 1e-3 * max(abs(t_agg_l1), 1.0)

                if delta_numel != 0 or delta_l1 > rel_tol:
                    mismatches.append(f"  * {name}: Trainer(L1={t_agg_l1:.4f}) vs Sampler(L1={s_agg_l1:.4f})")

            global_sampler_l2 = total_sampler_l2_sq**0.5
            total_l1_delta = abs(total_sampler_l1 - total_trainer_l1)
            total_l2_delta = abs(global_sampler_l2 - global_trainer_l2)
            total_numel_delta = abs(total_sampler_numel - total_trainer_numel)
            total_rel_tol = 1e-3 * max(abs(total_trainer_l1), 1.0)

            if not mismatches and total_numel_delta == 0 and total_l1_delta <= total_rel_tol:
                logger.info(
                    f"[RAIDEN PARITY VERIFIED | Step {step_key}] 100% DISTRIBUTED NORM PARITY CONFIRMED!\n"
                    f"  * Global L1 Norm: {total_sampler_l1:.6f} "
                    f"(Trainer={total_trainer_l1:.6f}, delta={total_l1_delta:.6f})\n"
                    f"  * Global L2 Norm: {global_sampler_l2:.6f} "
                    f"(Trainer={global_trainer_l2:.6f}, delta={total_l2_delta:.6f})\n"
                    f"  * Total Parameters: {total_sampler_numel} across {len(all_param_names)} tensors"
                )
            else:
                mismatches_summary = "\n".join(mismatches[:10])
                logger.error(
                    f"[RAIDEN PARITY MISMATCH | Step {step_key}] Norms do NOT match!\n"
                    f"  * Trainer: numel={total_trainer_numel}, L1={total_trainer_l1:.6f}, L2={global_trainer_l2:.6f}\n"
                    f"  * Sampler: numel={total_sampler_numel}, L1={total_sampler_l1:.6f}, L2={global_sampler_l2:.6f}\n"
                    f"  * Mismatched Tensors ({len(mismatches)} / {len(all_param_names)}):\n"
                    f"{mismatches_summary}"
                )
        except Exception as e:
            logger.warning(f"Error during parity verification for step {step_key}: {e}")


def setup_raiden_controller() -> tuple[Any, Any, str]:
    """Start embedded RaidenControllerServer on the head node and record its address in TPUWeightRegistry."""
    from tpu_sync.rpc import raiden_controller

    from verl.checkpoint_engine.tpu_weight_registry import get_tpu_weight_registry

    controller = raiden_controller.RaidenController(port=0)
    server = raiden_controller.RaidenControllerServer(controller)
    port = server.start()
    ip = ray.util.get_node_ip_address().strip("[]")
    address = f"{ip}:{port}"
    logger.info(f"RaidenControllerServer started on Headnode: {address}")

    try:
        registry = get_tpu_weight_registry()
        ray.get(registry.set_controller_address.remote(address))
        logger.info(f"Successfully stored RaidenController address ({address}) in TPUWeightRegistry")
    except Exception as reg_err:
        raise RuntimeError(
            f"Failed to store RaidenController address ({address}) in TPUWeightRegistry: {reg_err}"
        ) from reg_err

    return controller, server, address


def raiden_is_tile_aligned(local_shape: list) -> bool:
    """True when a 2D+ shard matches the TPU (8, 128) tile, so it can skip the CPU (de)tiling pass."""
    return (len(local_shape) >= 2) and (local_shape[-1] % 128 == 0) and (local_shape[-2] % 8 == 0)


def apply_raiden_skip_tiling(ws, skip_tiling_plan: list) -> None:
    """Set the per-tensor skip_tiling plan on a WeightSynchronizer (API name differs across tpu_sync builds)."""
    if hasattr(ws, "test_only_set_skip_tiling"):
        ws.test_only_set_skip_tiling(skip_tiling_plan)
    elif hasattr(ws, "set_skip_tiling"):
        ws.set_skip_tiling(skip_tiling_plan)


def _as_bool(value) -> bool:
    """Parse a bool flag that may arrive as a string (env var, CLI override) as well as a bool."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


@CheckpointEngineRegistry.register("raiden")
class RaidenCheckpointEngine(CheckpointEngine):
    """P2P Weight Synchronizer Checkpoint Engine for TPUs using Google Raiden (tpu-sync)."""

    # Receive the training engine in send_weights so each rank can register its local FSDP shard
    # (Raiden reshards to the sampler layout) instead of all-gathering full tensors.
    consumes_training_engine = True

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs) -> None:
        self.is_master = is_master
        self.bucket_size = bucket_size
        self.backend = "raiden"
        # Trainer ranks only post L1/L2 stats for the norm check; the exact check runs on the samplers.
        self.verify_parity = RaidenParityCheck(kwargs.get("verify_parity", False)).norm
        self.parallelism = kwargs.get("parallelism", 8)
        self.tp_size = kwargs.get("tp_size", kwargs.get("tensor_parallel_size", self.parallelism))
        self._trainer_raiden_ws = None
        self._trainer_chunks = []
        self._controller_addr = None
        self._registered_signature = None
        self._bound_tensors = None
        self._shard_info = {}
        # Free the device tensors bound for a transfer once the controller push has read them, instead of keeping
        # them until the next sync. With the local-shard export that is the bf16 copy of this rank's shards plus
        # the full q/k/v, which the fused-wqkv save hook still all-gathers to every rank: ~3.4 GiB per chip for
        # Qwen3-8B on 8 chips (~1.7 GiB of shards plus ~1.7 GiB of q/k/v, see get_per_tensor_param_shard). With
        # an all-gathered weights iterable it is the full bf16 model on every chip. The synchronizer pins the
        # device buffers of every bound tensor (dropping the Python references frees nothing), so the release
        # goes through WeightSynchronizer.unbind_weights(): the HBM holds are dropped while the synchronizer, its
        # pinned host staging buffers and its controller registration stay alive for the next sync to rebind.
        # On by default so the trainer keeps that headroom as models grow; opt out with the engine kwarg or
        # VERL_RAIDEN_RELEASE_BUFFERS=0 to keep the send tensors resident between syncs.
        release = _as_bool(kwargs.get("release_buffers_after_sync", True))
        env_release = os.environ.get("VERL_RAIDEN_RELEASE_BUFFERS", "")
        if env_release:
            release = _as_bool(env_release)
        self.release_buffers_after_sync = release
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = int(os.environ.get("RANK", "0"))

        from verl.checkpoint_engine.tpu_weight_registry import get_tpu_weight_registry

        self.registry = get_tpu_weight_registry()

    def prepare(self) -> dict:
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        return {}, {}

    def init_process_group(self, **kwargs):
        pass

    def finalize(self):
        if self._trainer_raiden_ws is not None:
            try:
                self._trainer_raiden_ws.close()
            except Exception:
                pass
            self._trainer_raiden_ws = None
        self._registered_signature = None
        self._bound_tensors = None

    async def _create_and_register(self, valid_weights: list[tuple[str, torch.Tensor]]) -> None:
        """(Re)create the trainer WeightSynchronizer on ``valid_weights`` and register it with the controller."""
        if self._trainer_raiden_ws is not None:
            try:
                self._trainer_raiden_ws.close()
            except Exception:
                pass
            self._trainer_raiden_ws = None

        logger.info(f"Trainer Rank {self.rank}: binding {len(valid_weights)} tensors to WeightSynchronizer ")
        bind_ip = ray.util.get_node_ip_address().strip("[]")
        self._trainer_raiden_ws = create_torch_weight_synchronizer(
            [[t] for _, t in valid_weights],
            local_port=0,
            listener_port=0,
            parallelism=getattr(self, "parallelism", 8),
            bind_ip=bind_ip,
        )

        def _full_shape(name: str, p: torch.Tensor) -> list[int]:
            shape = list(p.shape)
            if name in self._shard_info:
                shape[0] *= self._shard_info[name][0]
            return shape

        # Record global shapes to TPUWeightRegistry so Sampler can look them up
        # TODO(tpu): Move global_shapes registration to a one-time setup step during initialization
        # (e.g., in prepare or build_process_group/model_init) instead of per weight-sync step,
        # since tensor shapes are static across training iterations and only need to be communicated once.
        if self.registry is not None:
            try:
                global_shapes = {name: _full_shape(name, p) for name, p in valid_weights}
                ray.get(self.registry.set_global_shapes.remote(global_shapes))
            except Exception as e:
                logger.warning(f"Could not record global shapes in TPUWeightRegistry: {e}")

        # Build variable metadata protos for each dynamic tensor. Every variable carries its full shape.
        # - Local dim-0 shard: mesh_shape=[N, 1, ...] and global_shard_indices=[i] (this rank holds block i
        #   of N); Raiden reshards it to the sampler layout.
        # - Full tensor (replicated on every rank): mesh_shape=[1] * rank, unsharded.
        variable_protos = []
        from tpu_sync.rpc import raiden_service_pb2

        max_shards = 1
        for idx, (name, p) in enumerate(valid_weights):
            shape = _full_shape(name, p)
            itemsize = p.element_size()
            layout = list(range(len(shape) - 1, -1, -1))
            if name in self._shard_info:
                num_shards, shard_index = self._shard_info[name]
                max_shards = max(max_shards, num_shards)
                extra = {
                    "mesh_shape": [num_shards] + [1] * (len(shape) - 1),
                    "sharding_spec": ["fsdp"] + [""] * (len(shape) - 1),
                    "global_shard_indices": [shard_index],
                }
            else:
                extra = {"mesh_shape": [1] * len(shape), "sharding_spec": [""] * len(shape)}
            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=shape,
                    layout=layout,
                    item_size=itemsize,
                    layer_idx=idx,
                    **extra,
                )
            )
        num_sharded = sum(name in self._shard_info for name, _ in valid_weights)
        logger.info(
            f"Trainer Rank {self.rank}: registering {num_sharded} sharded and "
            f"{len(valid_weights) - num_sharded} full tensors"
        )

        # TODO(tpu): Consider passing controller_address directly during orchestration (e.g., via
        # CheckpointEngineManager / actor_wg.update_weights or init_process_group) rather than querying
        # the TPUWeightRegistry actor, eliminating cross-process registry lookups altogether.
        if self._controller_addr is None and self.registry is not None:
            for _ in range(30):
                try:
                    self._controller_addr = ray.get(self.registry.get_controller_address.remote())
                    if self._controller_addr:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.1)

        if self._controller_addr:
            from tpu_sync.rpc import raiden_controller

            try:
                ctrl_client = raiden_controller.RaidenControllerClientFacade(self._controller_addr)
                unit_id = raiden_controller.RaidenId("trainer", str(self.rank), "weights")
                ctrl_client.register_work_unit(
                    unit_id,
                    [f"{bind_ip}:{self._trainer_raiden_ws.local_port}"],
                    f"{bind_ip}:{self._trainer_raiden_ws.listener_port}",
                    mesh_shape=[max_shards, 1],
                    variables=variable_protos,
                    mesh_axes=["fsdp", "tp"],
                )
                logger.info(
                    f"Trainer Rank {self.rank} bound {len(valid_weights)} dynamic tensors and registered "
                    f"directly with RaidenController ({self._controller_addr}): "
                    f"data_port={self._trainer_raiden_ws.local_port}, "
                    f"listener_port={self._trainer_raiden_ws.listener_port}"
                )
            except Exception as reg_err:
                logger.error(
                    f"Trainer Rank {self.rank} failed to register with RaidenController "
                    f"({self._controller_addr}): {reg_err}"
                )
                raise
        else:
            raise RuntimeError(
                f"Trainer Rank {self.rank}: No RaidenController address found in TPUWeightRegistry after timeout"
            )

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None]
        | Iterable[tuple[str, torch.Tensor]]
        | dict[str, torch.Tensor],
        dst_peers: Optional[list[str]] = None,
        global_steps: Optional[int] = None,
        compute_stats: Optional[bool] = None,
        compute_checksum: Optional[bool] = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        """Register weights with RaidenController and prepare for coordinated P2P network transfer."""
        if compute_stats is None:
            compute_stats = compute_checksum if compute_checksum is not None else getattr(self, "verify_parity", False)
        step_key = global_steps if global_steps is not None else 0
        logger.info(f"RaidenCheckpointEngine: [Step {step_key}] Start send_weights...")

        try:
            from torch_tpu._internal import sync as torch_tpu_sync

            torch_tpu_sync.synchronize(wait=True)
        except Exception as e:
            logger.warning(f"Could not synchronize via torch_tpu: {e}")
            raise RuntimeError(f"TPU synchronization failed: {e}") from e

        if hasattr(weights, "get_per_tensor_param_shard"):
            # Handed the training engine: register each rank's local FSDP shard, no all-gather.
            hf_config = getattr(getattr(weights, "model_config", None), "hf_config", None)
            tie_word_embeddings = bool(getattr(hf_config, "tie_word_embeddings", tie_word_embeddings))
            named_weights, self._shard_info = export_local_shards(weights, tie_word_embeddings=tie_word_embeddings)
        else:
            named_weights = filter_tied_embeddings(
                weights.items() if hasattr(weights, "items") else weights,
                tie_word_embeddings=tie_word_embeddings,
            )
            self._shard_info = {}

        # Trainer sends pure canonical un-fused model weights directly
        unfused_weights = {k: _unwrap_tensor(v) for k, v in named_weights}

        sorted_weights = sorted(unfused_weights.items(), key=lambda x: x[0])
        valid_weights = validate_and_sanitize_tensors(sorted_weights, device=torch.device("tpu"))

        torch_tpu_sync.synchronize(wait=True)

        # Reuse the WeightSynchronizer across steps when the tensor signature is unchanged: rebinding keeps
        # the DMA-mapped host buffers, listener/threads, and controller registration, and only swaps the
        # device buffers the D2H reads from (release_sync_buffers() unbinds them between syncs). Rebuilding
        # every step re-allocates, pins, and first-touches a model-sized host buffer per rank (~1s/step at
        # 0.6B). Rebuild only on first use or if the (name, shape, dtype, shard) signature changes.
        signature = [(name, tuple(t.shape), t.dtype, self._shard_info.get(name)) for name, t in valid_weights]
        if self._trainer_raiden_ws is not None and signature == self._registered_signature:
            self._trainer_raiden_ws.bind_weights([[t] for _, t in valid_weights])
        else:
            await self._create_and_register(valid_weights)
            self._registered_signature = signature
        # Keep the bound device buffers alive until the next sync (or release_sync_buffers()): the
        # controller-driven push reads them after send_weights returns.
        self._bound_tensors = [t for _, t in valid_weights]

        # No explicit ws.d2h() here: PushWeightsResharded (triggered by the controller's start_transfer)
        # performs a pipelined D2H overlapped with H2H, using the per-transfer skip_tiling plan. An
        # explicit d2h() before the first transfer has no skip_tiling plan yet and stages the wrong
        # (detiled) layout; it was only ever masked by the push's own D2H.

        if not compute_stats or self.registry is None:
            return
        try:
            if self._shard_info:
                # Every rank posts partial stats: its local shards, plus the full tensors on rank 0 only.
                # The orchestrator sums them per tensor.
                items = [(n, t) for n, t in valid_weights if n in self._shard_info or self.rank == 0]
                trainer_stats = compute_tensor_stats(items)
                trainer_stats["rank"] = self.rank
                self.registry.set_rank_stats.remote(step_key, self.rank, trainer_stats)
            elif self.is_master:
                # Every rank holds the full model: rank 0's stats are the reference.
                trainer_stats = compute_tensor_stats(valid_weights)
                trainer_stats["rank"] = self.rank
                self.registry.set_stats.remote(step_key, trainer_stats)
            else:
                return
            logger.info(
                f"[RAIDEN PARITY] Successfully stored trainer rank {self.rank} stats for step {step_key}: "
                f"L1={trainer_stats['l1_norm']:.4f}, numel={trainer_stats['total_numel']}"
            )
        except Exception as e:
            logger.warning(f"Failed to record trainer rank {self.rank} stats in TPUWeightRegistry: {e}")

    def release_sync_buffers(self) -> dict:
        """Free this rank's bound send tensors while keeping the WeightSynchronizer for the next sync.

        Must only be called after the controller transfer that reads them has completed. Runs by default; a
        no-op when release_buffers_after_sync is disabled, in which case the bound tensors stay resident
        between syncs.

        WeightSynchronizer.unbind_weights() drops the synchronizer's holds on the bound device buffers, so
        dropping our own references returns the HBM. The synchronizer itself, its pinned host staging buffers
        and its controller registration survive, and D2H/H2D are rejected until the next send_weights rebinds
        through bind_weights(). Compared to destroying and re-creating the synchronizer each sync, this removes
        the per-step rebuild (host buffer allocation, pinning and first touch, plus controller
        re-registration: ~2 s "Trainer init" + ~2 s release on Qwen3-8B, v6e-8) from the sync critical path.
        """
        if not self.release_buffers_after_sync or self._trainer_raiden_ws is None:
            return {}
        if hasattr(self._trainer_raiden_ws, "unbind_weights"):
            self._trainer_raiden_ws.unbind_weights()
            self._bound_tensors = None
        else:
            ws, self._trainer_raiden_ws = self._trainer_raiden_ws, None
            self._bound_tensors = None
            self._registered_signature = None
            del ws
            gc.collect()
        # No empty_cache(): on TPU it clears the eager-op compilation cache (forcing recompiles) and the
        # TPU runtime reuses freed HBM without it.
        try:
            from torch_tpu._internal import sync as torch_tpu_sync

            torch_tpu_sync.synchronize(wait=True)
        except Exception:
            pass
        return {}

    @torch.no_grad()
    def receive_weights(self, global_steps: Optional[int] = None, **kwargs):
        return None


def _replica_num_workers(replica) -> int:
    """Number of Raiden work units a rollout replica registers: one per TP worker."""
    if getattr(replica, "world_size", None):
        return replica.world_size
    if getattr(replica, "workers", None):
        return len(replica.workers)
    return 1


def _sampler_job_name(replica_idx: int, num_replicas: int) -> str:
    """Raiden job name of a rollout replica: ``sampler`` with one replica, ``sampler<idx>`` with several.

    A rollout worker registers as (job name, rank within its replica). With several replicas the ranks
    ``0..TP-1`` would all collide under one name, and the controller would wait for ranks ``TP..N*TP-1`` that
    nothing ever registers. One job name per replica keeps every registration unique; a single replica keeps
    the historical name.
    """
    return "sampler" if num_replicas == 1 else f"sampler{replica_idx}"


async def update_raiden_weights(
    manager,
    global_steps: Optional[int] = None,
) -> dict:
    """Orchestrator coordination for Raiden TPU P2P weight synchronization via central RaidenController."""
    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning(f"Failed to abort replicas at step {global_steps}: {e}")
    t_abort = time.perf_counter() - t_abort_start
    parity = RaidenParityCheck.from_config(manager.config.engine_kwargs)

    t_total_start = time.perf_counter()

    # 1. Trigger Trainer ranks to register their tensors with central RaidenController and record global shapes
    t_init_trainer_start = time.perf_counter()

    # Blocks untill all Trainer ranks complete their send_weights, but do so asynchronously
    # in a background thread so we don't freeze the orchestrator's event loop.
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode="raiden")
    if actor_refs is not None:
        await asyncio.to_thread(ray.get, actor_refs)
    t_init_trainer = time.perf_counter() - t_init_trainer_start

    parallelism = 8
    if hasattr(manager, "config") and hasattr(manager.config, "engine_kwargs"):
        parallelism = manager.config.engine_kwargs.get("raiden", {}).get(
            "parallelism", manager.config.engine_kwargs.get("parallelism", 8)
        )

    # 2. Initialize and register Sampler rollout workers with central RaidenController
    t_init_sampler_start = time.perf_counter()
    sampler_init_futures = [
        replica.server_handle.collective_rpc.remote(
            method="init_raiden_sync_on_worker",
            kwargs={"parallelism": parallelism, "job_name": _sampler_job_name(replica_idx, len(manager.replicas))},
        )
        for replica_idx, replica in enumerate(manager.replicas)
    ]
    await asyncio.gather(*sampler_init_futures)
    t_init_sampler = time.perf_counter() - t_init_sampler_start

    # 3. Explicit Registration Barrier on Central RaidenController
    # TODO(tpu): Move worker registration and barrier verification to a one-time setup step during
    # initialization (e.g. in prepare/build_process_group), since shard registration is persistent on the
    # controller and does not need to be repeated on every weight sync iteration.
    t_barrier_start = time.perf_counter()

    # One group of destination units per rollout replica: ranks '0'..'TP-1' under the replica's own job name.
    # Examples:
    #   - 1 replica with TP=8: one group ['0'..'7'] under job "sampler".
    #   - 3 replicas with TP=8: groups ['0'..'7'] under "sampler0", "sampler1", "sampler2".
    num_replicas = len(manager.replicas)
    trainer_replica_ids = [str(i) for i in range(manager.actor_wg.world_size)]

    from tpu_sync.api.common import RaidenId
    from tpu_sync.rpc.raiden_controller import RaidenMemoryType

    src_units = [RaidenId(job_name="trainer", job_replica_id=r_id, data_name="weights") for r_id in trainer_replica_ids]
    dst_unit_groups = [
        [
            RaidenId(job_name=_sampler_job_name(replica_idx, num_replicas), job_replica_id=str(k), data_name="weights")
            for k in range(_replica_num_workers(replica))
        ]
        for replica_idx, replica in enumerate(manager.replicas)
    ]
    dst_units = [unit for group in dst_unit_groups for unit in group]

    if not hasattr(manager, "raiden_controller") or manager.raiden_controller is None:
        raise RuntimeError(
            "No raiden_controller found on CheckpointEngineManager! "
            "Embedded RaidenControllerServer must be initialized on the head node before weight sync."
        )

    barrier_timeout = 60.0
    while True:
        with manager.raiden_controller._lock:
            registered = set(manager.raiden_controller._registered_shards.keys())
        src_registered = all(u in registered for u in src_units)
        dst_registered = all(u in registered for u in dst_units)
        if src_registered and dst_registered:
            logger.info(
                f"[RAIDEN CONTROLLER] All {len(src_units)} Trainer and {len(dst_units)} Sampler units "
                "verified and registered."
            )
            break
        if time.perf_counter() - t_barrier_start > barrier_timeout:
            missing_src = [u for u in src_units if u not in registered]
            missing_dst = [u for u in dst_units if u not in registered]
            raise RuntimeError(
                f"Timeout ({barrier_timeout}s) waiting for workers to register with RaidenController! "
                f"Missing Trainer: {missing_src}, Missing Sampler: {missing_dst}"
            )
        await asyncio.sleep(0.1)
    t_barrier = time.perf_counter() - t_barrier_start

    # 4. Trigger coordinated P2P network transfers via central RaidenController
    t_transfer_start = time.perf_counter()
    # One transfer per replica. Each has the destination mesh of the validated single-replica case ([1, TP]),
    # instead of one transfer whose destination mixes the ranks of several meshes. The transfers are issued
    # together and awaited together: the controller tracks each by its own req_id / uuid and the trainer ranks
    # keep their D2H and skip-tiling state per uuid, so the replicas overlap instead of paying the per-transfer
    # latency one after another. The trainer buffers are only released after every transfer has finished.
    transfer_futures = []
    for replica_idx, dst_group in enumerate(dst_unit_groups):
        req_id = f"verl_step_{global_steps or 0}" + (f"_replica{replica_idx}" if num_replicas > 1 else "")
        transfer_futures.append(
            manager.raiden_controller.start_transfer(
                src_units=src_units,
                dst_units=dst_group,
                dst_mem_type=RaidenMemoryType.DRAM,
                use_block_chunks=True,
                is_sender=True,
                expected_block_count=0,
                parallelism=parallelism,
                req_id=req_id,
            )
        )
    await asyncio.gather(*[future.wait() for future in transfer_futures])
    t_transfer = time.perf_counter() - t_transfer_start

    # The trainer buffers are no longer read once the transfer is done; free them (unless disabled)
    # while the samplers install, so the HBM is back before the next training step.
    release_refs = None
    if hasattr(manager.actor_wg, "release_raiden_sync_buffers"):
        release_refs = manager.actor_wg.release_raiden_sync_buffers()

    # 5. Sampler replicas install received weights to TPU HBM via H2D DMA
    t_install_start = time.perf_counter()
    install_kwargs = parity.exact_install_kwargs(global_steps)
    install_futures = [
        replica.server_handle.collective_rpc.remote(method="install_raiden_weights", kwargs=install_kwargs)
        for replica in manager.replicas
    ]
    install_results = await asyncio.gather(*install_futures)
    t_install = time.perf_counter() - t_install_start
    # collective_rpc returns one result per TP worker of each replica; each is the timing dict returned by
    # vLLMRaidenWorkerExtension.install_raiden_weights. The slowest worker gates the sync, so report the max.
    worker_install_stats = [
        r for per_replica in install_results for r in (per_replica or []) if isinstance(r, dict) and "h2d" in r
    ]
    t_h2d_pure = max((r["h2d"] for r in worker_install_stats), default=None)
    # Tag new generations with this weight version (the trajectory staleness metrics read it), as the other
    # checkpoint backends do once the new weights are loaded.
    if global_steps is not None:
        await asyncio.gather(
            *[replica.server_handle.set_global_steps.remote(global_steps) for replica in manager.replicas]
        )
    if release_refs is not None:
        await asyncio.to_thread(ray.get, release_refs)

    t_total = time.perf_counter() - t_total_start

    # 5. Parity Verification (Optional, default=off; see RaidenParityCheck)
    if install_kwargs:
        parity.report_exact(global_steps, install_results)
    if parity.norm:
        try:
            await parity.verify_norms(manager, global_steps)
        except Exception as e:
            logger.warning(f"Failed to execute parity verification: {e}")

    logger.info(
        f"[RAIDEN TELEMETRY | Orchestrator] Step {global_steps} Completed in {t_total:.4f}s:\n"
        f"  * Sampler Quiesce/Pause  : {t_abort:.4f}s\n"
        f"  * Trainer Raiden Init    : {t_init_trainer:.4f}s\n"
        f"  * Sampler Raiden Init    : {t_init_sampler:.4f}s\n"
        f"  * Raiden Barrier Check   : {t_barrier:.4f}s\n"
        f"  * RaidenController P2P   : {t_transfer:.4f}s\n"
        f"  * Sampler H2D DMA        : {t_install:.4f}s\n"
        f"  * Total End-to-End Sync  : {t_total:.4f}s"
    )

    # 6. Resume generation immediately
    await manager.resume_generation_replicas()

    # Surface the phase timers as step metrics. The trainer stashes this dict in _pending_sync_metrics and
    # merges it into the same step's metrics, so they are logged next to timing_s/update_weights by every
    # configured backend (console / tensorboard / wandb). timing_s/update_weights ~= quiesce + total_sync.
    metrics = {
        "timing_s/tpu-sync/quiesce": t_abort,
        "timing_s/tpu-sync/trainer_init": t_init_trainer,
        "timing_s/tpu-sync/sampler_init": t_init_sampler,
        "timing_s/tpu-sync/barrier": t_barrier,
        "timing_s/tpu-sync/p2p_transfer": t_transfer,
        "timing_s/tpu-sync/sampler_h2d": t_install,
        "timing_s/tpu-sync/total_sync": t_total,
    }
    if t_h2d_pure is not None:
        # Pure _raiden_ws.h2d() time on the slowest sampler worker (sampler_h2d above also includes the
        # fuse/transpose into vLLM params, the TPU sync barrier and the RPC round trip).
        metrics["timing_s/tpu-sync/sampler_h2d_pure"] = t_h2d_pure
    return metrics

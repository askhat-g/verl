# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import importlib
import logging
import re
from collections import defaultdict
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.models.common.attention import (
    AttentionMasksType,
    VarlenMetadata,
    create_attention_mask,
    get_causal_mask_mod,
)

logger = logging.getLogger(__name__)


class NoOpDataLoader(BaseDataLoader):
    """A no-op dataloader for use when verl manages its own data loading.

    Satisfies the BaseDataLoader interface required by torchtitan's Trainer
    but does nothing. Its __iter__ yields nothing, and state_dict /
    load_state_dict are no-ops.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        pass

    def __init__(self, **kwargs):
        pass

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        return iter([])

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# Mapping from HuggingFace model_type to torchtitan model name.
# Torchtitan models not mapped here:
#   - flux: diffusion model, not applicable to verl's RL/SFT workflows
#   - llama3_ft: fault-tolerant variant of llama3, same HF models (mapped via "llama")
_HF_MODEL_TYPE_TO_TORCHTITAN_NAME = {
    "qwen2": "qwen3",
    "qwen3": "qwen3",
    "qwen2_moe": "qwen3",
    "qwen3_moe": "qwen3",
    "llama": "llama3",
    "llama4": "llama4",
    "deepseek_v3": "deepseek_v3",
    "gpt_oss": "gpt_oss",
}


def derive_torchtitan_name_and_flavor(hf_config) -> tuple[str, str]:
    """Derive torchtitan model name and flavor from a HuggingFace config.

    The name is mapped from ``hf_config.model_type``. The flavor is found by
    matching architecture parameters (dim, n_layers, vocab_size) against the
    known flavors registered in the torchtitan model package.

    Args:
        hf_config: A HuggingFace AutoConfig object.

    Returns:
        A ``(name, flavor)`` tuple.

    Raises:
        ValueError: If model_type is unsupported or no matching flavor is found.
    """
    model_type = getattr(hf_config, "model_type", None)
    if model_type is None:
        raise ValueError("HuggingFace config does not have 'model_type' field")

    name = _HF_MODEL_TYPE_TO_TORCHTITAN_NAME.get(model_type)
    if name is None:
        raise ValueError(
            f"Cannot derive torchtitan model name from HF model_type '{model_type}'. "
            f"Supported types: {list(_HF_MODEL_TYPE_TO_TORCHTITAN_NAME.keys())}."
        )

    # Import the model package and use model_registry to build each flavor's config.
    # model_registry has sensible defaults for all optional params (attn_backend, etc.).
    model_module = importlib.import_module(f"torchtitan.models.{name}")
    model_registry = model_module.model_registry

    # The configs dict name isn't derivable from the model name
    # (e.g. gpt_oss -> gptoss_configs), so we find it by convention.
    flavor_names = None
    for attr, obj in vars(model_module).items():
        if attr.endswith("_configs") and isinstance(obj, dict):
            flavor_names = list(obj.keys())
            break

    if flavor_names is None:
        raise ValueError(
            f"Could not find model configs dict in torchtitan.models.{name}. "
            f"Expected a dict attribute ending with '_configs'."
        )

    hidden_size = hf_config.hidden_size
    num_layers = hf_config.num_hidden_layers
    vocab_size = hf_config.vocab_size

    for flavor_name in flavor_names:
        cfg = model_registry(flavor_name).model
        n_layers = getattr(cfg, "n_layers", None) or len(getattr(cfg, "layers", []))
        if (
            getattr(cfg, "dim", None) == hidden_size
            and n_layers == num_layers
            and getattr(cfg, "vocab_size", None) == vocab_size
        ):
            logger.info(
                f"Auto-derived torchtitan name='{name}', flavor='{flavor_name}' from HF model_type='{model_type}'"
            )
            return name, flavor_name

    raise ValueError(
        f"No matching torchtitan flavor found for model_type='{model_type}' "
        f"(hidden_size={hidden_size}, num_hidden_layers={num_layers}, "
        f"vocab_size={vocab_size}). "
        f"Available flavors for '{name}': {flavor_names}."
    )


# Row tile of the MegaBlox grouped matmul that runs the routed experts on TPU. MegaBlox requires the routed
# row count (tokens * top_k) to be a multiple of it.
_TPU_PALLAS_GROUPED_MM_TILE_M = 256


def configure_qwen3_moe_for_rl(
    model_config,
    *,
    use_pallas_grouped_mm: bool,
    expert_parallel_size: int = 1,
    token_multiple: int = 1,
):
    """Adapt a torchtitan Qwen3 MoE model config for RL against an HF-format rollout engine.

    Returns ``model_config`` unchanged for dense flavors (no ``MoE.Config`` in the tree).

    Load balancing: torchtitan's Qwen3 MoE flavors balance experts without an aux loss
    (``MoE.Config.load_balance_coeff=1e-3``): the ModelSpec's ``post_optimizer_build_fn``
    (``register_moe_load_balancing_hook``, registered by torchtitan's Trainer) nudges a per-layer
    ``expert_bias_E`` before every optimizer step, and the router adds that bias to the scores it picks the
    top-k experts from. HF Qwen2/Qwen3 MoE checkpoints have no such bias and ``Qwen3StateDictAdapter.to_hf``
    drops it, so the rollout engine (and the ref model) never see it: from the second step on the actor would
    route tokens to other experts than the rollout that sampled them. The ``*-global-aux`` flavors instead
    inject an aux-loss gradient through a custom autograd function, outside verl's policy loss and scaled for
    torchtitan's batch accounting. Both are cleared so the trainer routes exactly like the HF model.
    ``model_registry`` does not forward flavor kwargs, so the built config is edited, as torchtitan's TPU
    235B-A22B recipe does (``experiments/tpu/qwen3/config_registry.py``).

    Routed experts on TPU (``use_pallas_grouped_mm``): swaps ``GroupedExperts`` for torchtitan's
    ``TpuGroupedExperts`` with the MegaBlox Pallas grouped matmul, as torchtitan's Qwen3-30B-A3B TPU recipe
    does, because the native ``torch._grouped_mm`` backward computes a wrong expert dWeight on TPU
    (torch_tpu#2668, see ``experiments/tpu/grouped_experts.py``). The parameters keep the canonical
    ``w1_EFD``/``w2_EDF``/``w3_EFD`` layout (no state-dict hooks, so checkpoint, HF conversion and weight
    sync are unchanged) and use the unpaired kernel with full K/N dWeight tiles (fp32 accumulation), the
    torchtitan TPU recipe's setting before its paired W1/W3 variant. Needs EP=1 (exact routed-row dispatch; EP>1
    raises, since the native fallback's dWeight is wrong) and a routed row count that is a multiple of the 256-row
    tile.

    Args:
        model_config: torchtitan model config (``ModelSpec.model``), edited in place.
        use_pallas_grouped_mm: Run the routed experts with the TPU Pallas grouped matmul.
        expert_parallel_size: Expert parallel degree; must be 1 when ``use_pallas_grouped_mm`` (raises otherwise).
        token_multiple: Every micro-batch's token count is a multiple of this (the TPU sequence bucket).

    Returns:
        The edited model config.
    """
    from torchtitan.models.common.moe import MoE

    moe_configs = [moe for _, moe, _, _ in model_config.traverse(MoE.Config)]
    if not moe_configs:
        return model_config
    for moe in moe_configs:
        moe.load_balance_coeff = None
        moe.aux_loss = None
    logger.warning("MoE: disabled expert-bias load balancing and aux loss on %d layers", len(moe_configs))

    if not use_pallas_grouped_mm:
        return model_config
    if expert_parallel_size > 1:
        # EP dispatch yields ragged routed-row counts that are not 256-tile aligned, so the Pallas kernel cannot
        # run (torchtitan's TPU TpuGroupedExperts also rejects EP for its paired variant), and the native
        # torch._grouped_mm fallback computes a wrong expert dWeight on TPU: refuse instead of training silently
        # with wrong expert gradients.
        raise ValueError(
            f"expert_parallel_size={expert_parallel_size} is not supported for Qwen3 MoE on TPU: the Pallas grouped "
            "matmul needs EP=1 (exact, tile-aligned routed rows), and the native torch._grouped_mm fallback computes "
            "a wrong expert dWeight on TPU (torch_tpu#2668). Use expert_parallel_size=1 with FSDP "
            "(data_parallel_shard_size) instead."
        )

    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts, TpuGroupedExpertsConverter

    tile_m = _TPU_PALLAS_GROUPED_MM_TILE_M
    token_multiple = max(int(token_multiple), 1)  # bucket sizes <= 1 disable bucketing
    for moe in moe_configs:
        if (token_multiple * moe.router.top_k) % tile_m:
            raise ValueError(
                f"The Pallas grouped matmul needs tokens * top_k to be a multiple of {tile_m}, but token counts "
                f"are only guaranteed to be multiples of {token_multiple} (top_k={moe.router.top_k}). Set "
                f"VERL_TPU_SEQ_BUCKET_SIZE so that it times top_k is a multiple of {tile_m} (the default 256 is)."
            )
    model_config = TpuGroupedExpertsConverter.Config().build().convert(model_config)
    n_experts = 0
    for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config):
        experts.use_pallas_grouped_mm = True
        experts.use_pallas_paired_grouped_mm = False  # paired W1/W3 is a later perf variant
        experts.use_kernel_native_weight_layout = False  # keep canonical param names and layout
        experts.pallas_grouped_mm_tile_m = tile_m
        experts.pallas_grouped_mm_tgmm_tiling = (tile_m, 0, 0)  # full K/N dWeight tiles
        n_experts += 1
    logger.warning("MoE: routed experts on %d layers use the Pallas grouped matmul (tile_m=%d)", n_experts, tile_m)
    return model_config


def enable_fsdp_gradient_division(model: nn.Module, dp_size: int) -> None:
    """
    Re-enable FSDP's automatic gradient division.

    TorchTitan calls disable_fsdp_gradient_division() which sets gradient_divide_factor=1.0.
    This re-enables it by setting the factor to the specified dp_size, so gradients are
    averaged across FSDP ranks. This is needed for verl's loss scaling (loss * dp_size)
    to work correctly.

    Args:
        model: The model (or model part) to enable gradient division on.
        dp_size: The data parallel size to use as the gradient divide factor.
    """

    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(float(dp_size))


def make_simple_fsdp_parallelize_fn(parallelize_fn, *, spmd_safe_blocks: bool = False):
    """Wrap a torchtitan ``parallelize_fn`` so it shards with SimpleFSDP instead of FSDP2.

    Mirrors torchtitan's TPU recipes (``experiments/tpu/qwen3/infra/parallelize.py`` with
    ``parallelism.use_simple_fsdp``): AC and per-block compile are applied by the model's own
    ``parallelize_fn`` with ``skip_dp=True``, then graph_trainer's ``apply_simple_fsdp`` turns every parameter
    into a ``Shard(0)`` DTensor behind a parametrization. Accessing ``module.weight`` all-gathers it (cast to the
    param dtype) and its backward reduce-scatters the gradient (in the reduce dtype), so under ``torch.compile``
    the FSDP collectives live inside each compiled TransformerBlock, and under activation checkpointing the
    all-gather is recomputed in backward (reshard-after-forward semantics).

    Differences from the TPU recipe handled here:
      * Weight tying: ``data_parallel`` re-registers each module's parameter separately, which would untie
        ``tok_embeddings.weight``/``lm_head.weight``; both modules are pointed back at the same sharded parameter.
      * ``init_weights`` runs under ``disable_active_parametrization`` so initialization (and the decoder's
        re-tie) sees the sharded parameters instead of all-gathered temporaries.

    Gradients are summed, not averaged, across the data-parallel ranks (no FSDP2 gradient divide factor):
    the engine scales the loss by ``1 / dp_size`` before backward.

    Args:
        parallelize_fn: The model's torchtitan ``parallelize_fn`` (must accept ``skip_dp``).
        spmd_safe_blocks: Mark each dense TransformerBlock ``spmd_safe`` (torch_tpu) so the collectives traced
            into the compiled blocks run as SPMD collectives. Needed on TPU with ``torch.compile``. Blocks that
            contain an ``MoE`` stay unmarked, as in torchtitan's TPU Qwen3 recipe: their routing is data
            dependent (with EP, host-synced all-to-all splits), so ranks are not guaranteed identical programs.
    """
    from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
    from torchtitan.experiments.graph_trainer.simple_fsdp import disable_active_parametrization

    def _parallelize(model: nn.Module, *, parallel_dims, training, compile_config, **kwargs) -> nn.Module:
        n_spmd_safe = 0
        if spmd_safe_blocks and compile_config.enable and hasattr(model, "layers"):
            from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
            from torchtitan.models.common.moe import MoE

            for block in model.layers.values():
                if not any(isinstance(m, MoE) for m in block.modules()):
                    apply_spmd_safe(block)
                    n_spmd_safe += 1

        kwargs.pop("skip_dp", None)
        model = parallelize_fn(
            model, parallel_dims=parallel_dims, training=training, compile_config=compile_config, skip_dp=True, **kwargs
        )
        tied = bool(getattr(model, "enable_weight_tying", False))
        model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)
        if tied:
            model.tok_embeddings._parameters["weight"] = model.lm_head._parameters["weight"]

        init_weights = model.init_weights

        def _init_weights(*args, **init_kwargs):
            with disable_active_parametrization():
                return init_weights(*args, **init_kwargs)

        model.init_weights = _init_weights
        logger.info("Applied SimpleFSDP (weight tying preserved: %s, spmd_safe blocks: %d)", tied, n_spmd_safe)
        return model

    return _parallelize


def get_attention_masks(
    input_batch: torch.Tensor,
    positions: torch.Tensor,
    attn_type: str,
) -> AttentionMasksType:
    match attn_type:
        case "flex" | "flex_flash":
            # flex_flash is FlexAttention with the FLASH kernel backend; it uses
            # the same BlockMask as flex.
            return _get_flex_attention_masks(
                input_batch,
                positions,
            )
        case "varlen":
            return _create_varlen_metadata_for_document(
                input_batch,
                positions,
            )
        case _:
            raise TypeError("Only varlen and flex attn masks are supported")


def _get_document_mask_mod(positions: torch.Tensor) -> _mask_mod_signature:
    # Detect boundaries from position resets
    first_dummy_value = positions[:, :1] - 1
    position_diff = torch.diff(positions, prepend=first_dummy_value, dim=-1)
    sequence_indices = (position_diff != 1).cumsum(-1)  # [batch, seq]

    def document_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]

    return document_mask


def _get_flex_attention_masks(
    input_batch: torch.Tensor,
    positions: torch.Tensor,
) -> AttentionMasksType:
    mask_mods = [get_causal_mask_mod()]
    B = input_batch.shape[0]
    mask_mods.append(_get_document_mask_mod(positions=positions))
    return create_attention_mask(and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1])


def _create_varlen_metadata_for_document(input_batch: torch.Tensor, positions: torch.Tensor) -> VarlenMetadata:
    """
    Creates cumulative sequence length indices needed for variable length attention

    Args:
        input_batch: Input token IDs with shape [batch, seq].
        positions: Position IDs with shape [batch, seq]. Boundaries detected where
            position diff != 1 (i.e., position resets).

    Returns:
        VarlenMetadata containing cumulative sequence length indices for q, k, and max_seq_len
    """
    batch_size, seq_len = input_batch.shape
    device = input_batch.device

    # Detect boundaries from position resets (where diff != 1)
    first_dummy_value = positions[:, :1] - 1
    position_diff = torch.diff(positions, prepend=first_dummy_value, dim=-1)
    # boundary_mask[b, i] is True if position i starts a new document
    boundary_mask = position_diff != 1  # [batch, seq]
    boundary_mask[:, 0] = True

    cu_seqlens_list, all_seq_lengths = [], []
    offset = 0

    for b in range(batch_size):
        # Find positions where new documents start
        boundary_positions = boundary_mask[b].nonzero(as_tuple=True)[0].to(torch.int32)
        sample_cu_seqlens = torch.cat(
            [
                boundary_positions,
                torch.tensor([seq_len], dtype=torch.int32, device=device),
            ]
        )
        if sample_cu_seqlens.device.type == "tpu":
            # aten::unique_consecutive is not implemented in torch_tpu; use mask indexing instead.
            mask = torch.cat(
                [
                    torch.tensor([True], device=sample_cu_seqlens.device),
                    sample_cu_seqlens[1:] != sample_cu_seqlens[:-1],
                ]
            )
            sample_cu_seqlens = sample_cu_seqlens[mask]
        else:
            sample_cu_seqlens = torch.unique_consecutive(sample_cu_seqlens)

        seq_lengths = torch.diff(sample_cu_seqlens)
        all_seq_lengths.append(seq_lengths)

        cu_seqlens_adjusted = sample_cu_seqlens[:-1] + offset
        cu_seqlens_list.append(cu_seqlens_adjusted)

        offset += seq_len

    packed_cu_seqlens = torch.cat(cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)])

    max_seqlen = 0
    if len(all_seq_lengths) > 0:
        all_seq_lengths = torch.cat(all_seq_lengths)
        # device to host sync but only done once per model forward
        max_seqlen = all_seq_lengths.max().item()

    return VarlenMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )


# Regex to parse: model.layers.{L}.mlp.experts.{E}.{weight_suffix}
_EXPERT_PATTERN = re.compile(r"\.layers\.(\d+)\..*\.experts\.(\d+)\.(.*)")


def _parse_expert_name(name: str) -> tuple[int, int, str] | None:
    """Parse layer_id, expert_id, weight_suffix from expert param name."""
    match = _EXPERT_PATTERN.search(name)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None


def _make_expert_name_template(name: str) -> str:
    """Convert 'model.layers.0.mlp.experts.3.w1' -> 'model.layers.0.mlp.experts.{}.w1'"""
    return _EXPERT_PATTERN.sub(lambda m: f".layers.{m.group(1)}.mlp.experts.{{}}.{m.group(3)}", name)


def iter_per_tensor_params_ep(
    params: dict[str, Any],
    device: int,
    ep_group: torch.distributed.ProcessGroup,
    ep_size: int,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield (name, tensor) pairs for weight sync with Expert Parallel.

    Gathers expert weights across EP ranks one (layer, weight_type) group
    at a time to avoid OOM from materializing all experts simultaneously.

    Non-expert params are yielded first (with FSDP full_tensor() if needed),
    then expert params are all-gathered per group and yielded individually.

    Args:
        params: HF-format state dict with per-expert keys. Expert keys must
            follow the pattern ``model.layers.{L}.mlp.experts.{E}.{suffix}``.
        device: device ID to place tensors on.
        ep_group: The EP process group for all-gather.
        ep_size: Number of EP ranks.
    """
    expert_params: dict[tuple[int, str], dict[int, tuple[str, Any]]] = defaultdict(dict)
    non_expert_params: list[tuple[str, Any]] = []

    for name, param in params.items():
        parsed = _parse_expert_name(name) if "mlp.experts." in name else None
        if parsed is None:
            non_expert_params.append((name, param))
        else:
            layer_id, expert_id, weight_suffix = parsed
            expert_params[(layer_id, weight_suffix)][expert_id] = (name, param)

    params.clear()

    # Yield non-expert params
    for name, param in non_expert_params:
        if isinstance(param, DTensor):
            yield name, param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
        else:
            yield name, param
    del non_expert_params

    # Yield expert params with all-gather
    for (layer_id, weight_suffix), experts_dict in sorted(expert_params.items()):
        sorted_expert_ids = sorted(experts_dict.keys())

        # Stack local expert weights
        local_weights = []
        for eid in sorted_expert_ids:
            _, param = experts_dict[eid]
            if isinstance(param, DTensor):
                param = param.to(device, non_blocking=True).full_tensor()
            else:
                param = param.to(device, non_blocking=True)
            local_weights.append(param)

        name_template = _make_expert_name_template(experts_dict[sorted_expert_ids[0]][0])
        local_stacked = torch.stack(local_weights, dim=0)

        # All-gather across EP ranks
        gathered_list = [torch.empty_like(local_stacked) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered_list, local_stacked, group=ep_group)
        all_experts = torch.cat(gathered_list, dim=0)

        for expert_id in range(all_experts.shape[0]):
            yield name_template.format(expert_id), all_experts[expert_id].to(torch.bfloat16).clone()

        del local_weights, local_stacked, gathered_list, all_experts
        torch.cuda.empty_cache()

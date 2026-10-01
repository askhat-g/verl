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
"""TPU sequence packing utilities, attention monkey patches, and memory layout helpers."""

import logging
import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def unwrap_metadata(val):
    """Recursively unwraps metadata values (lists, single-element tensors) to standard Python types."""
    if isinstance(val, list):
        if len(val) > 0:
            return unwrap_metadata(val[0])
        return None
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        elif val.numel() > 1:
            return val.flatten()[0].item()
    return val


def monkey_patch_varlen_attention_tpu():
    """Patches TorchTitan's VarlenAttention forward method to use native scaled_dot_product_attention on TPU."""
    try:
        from torchtitan.models.common.attention import VarlenAttention

        def tpu_varlen_forward(
            self,
            q_BLNH,
            k_BLNH,
            v_BLNH,
            *,
            attention_masks,
            scale=None,
            out_transform=None,
            **kwargs,
        ):
            xq, xk, xv = q_BLNH, k_BLNH, v_BLNH
            if (
                hasattr(attention_masks, "cu_seq_q")
                or hasattr(attention_masks, "cu_seqlens_q")
                or hasattr(attention_masks, "cu_seqlens")
            ):
                cu_seqs = getattr(
                    attention_masks,
                    "cu_seq_q",
                    getattr(attention_masks, "cu_seqlens_q", getattr(attention_masks, "cu_seqlens", None)),
                )
                total_tokens = xq.shape[1]
                positions = torch.arange(total_tokens, device=xq.device)
                seq_indices = (positions.unsqueeze(1) >= cu_seqs.unsqueeze(0)).sum(dim=1) - 1
                same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
                causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)

                mask = same_seq_mask & causal_mask
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif isinstance(attention_masks, torch.Tensor):
                if attention_masks.dim() == 4:
                    mask = attention_masks.to(torch.bool)
                else:
                    seq_len = xq.shape[1]
                    positions = torch.arange(seq_len, device=xq.device)
                    causal_mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

                    padding_mask = attention_masks.unsqueeze(1).unsqueeze(2).to(torch.bool)
                    mask = causal_mask & padding_mask
            else:
                seq_len = xq.shape[1]
                positions = torch.arange(seq_len, device=xq.device)
                mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

            q = xq.transpose(1, 2)
            k = xk.transpose(1, 2)
            v = xv.transpose(1, 2)

            if q.shape[1] != k.shape[1]:
                num_repeat = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(num_repeat, dim=1)
                v = v.repeat_interleave(num_repeat, dim=1)

            # Eager only: TorchTitanEngine rejects use_torch_compile without use_splash_attention on TPU,
            # because F.scaled_dot_product_attention traced by torch.compile(backend="tpu") yields NaN q/k/v
            # gradients. With splash attention enabled this module is replaced by TPUSplashAttention.
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
            return attn_out.transpose(1, 2)

        VarlenAttention.forward = tpu_varlen_forward
        logger.info("Successfully patched VarlenAttention.forward for TPU execution.")
    except Exception as e:
        logger.warning(f"Failed to patch VarlenAttention: {e}")


def configure_torch_compile_for_tpu(recompile_limit: int = 64) -> None:
    """Dynamo settings for per-TransformerBlock ``torch.compile(backend="tpu")``.

    verl feeds packed sequences whose length is bucketed to a multiple of
    ``VERL_TPU_SEQ_BUCKET_SIZE``, so each block sees a small set of static shapes.
    - ``automatic_dynamic_shapes=False``: after the second distinct shape Dynamo
      would otherwise retrace with symbolic shapes, which the XLA backend handles
      badly (padding + recompiles, no fusion). Keep every bucket static.
    - ``recompile_limit``: one graph per (bucket, grad-mode) pair; the default
      limit (8) is hit quickly and then Dynamo fails under ``fullgraph=True``.

    Note: ``torch._dynamo.config`` stores overrides in a ``ContextVar``, so values set
    during ``init_model`` do not carry over to later Ray actor RPCs unless
    ``_config[name].default`` is updated too.
    """
    import torch._dynamo

    entries = getattr(torch._dynamo.config, "_config", {})

    def _set(name: str, value: Any) -> None:
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, value)
        if name in entries:
            entries[name].default = value

    _set("automatic_dynamic_shapes", False)
    _set("assume_static_by_default", True)
    _set("capture_scalar_outputs", True)
    _set("skip_fwd_side_effects_in_bwd_under_checkpoint", True)
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            _set(name, max(getattr(torch._dynamo.config, name), recompile_limit))
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        _set(
            "accumulated_recompile_limit",
            max(torch._dynamo.config.accumulated_recompile_limit, recompile_limit * 16),
        )


def splash_block_size_for(seq_len: int, max_block_size: int = 512, min_block_size: int = 128) -> int:
    """Largest power-of-two splash block size <= ``max_block_size`` that divides ``seq_len``.

    The splash kernel requires every block size to divide the sequence length. verl pads
    packed sequences to a multiple of ``VERL_TPU_SEQ_BUCKET_SIZE`` (256), so a fixed 512
    block (torchtitan's default) fails for e.g. 768 tokens. Returns 0 if no block
    >= ``min_block_size`` divides ``seq_len``.
    """
    block = max_block_size
    while block >= min_block_size:
        if seq_len % block == 0:
            return block
        block //= 2
    return 0


class TPUSplashAttention(torch.nn.Module):
    """TorchTitan inner attention backed by the TPU splash attention Pallas kernel.

    Same contract as ``torchtitan.experiments.tpu.kernels.splash_attention.SplashAttention``
    ((B, S, H, D) in and out, causal + packed-document masking from ``segment_ids``), but the
    block sizes are chosen per call from the (static, bucketed) sequence length instead of
    being fixed at 512. Falls back to the original module if no block size fits.
    """

    def __init__(self, original_module: torch.nn.Module, local_window_size: int | None = None):
        super().__init__()
        self.original_module = original_module
        self.local_window_size = local_window_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        attention_masks: Any = None,
        positions: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from torch.distributed.tensor import DTensor
        from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa
        from torchtitan.models.common.attention import segment_ids_from_positions

        block = splash_block_size_for(int(q.shape[1]))
        if block == 0 or q.device.type == "cpu":
            return self.original_module(
                q,
                k,
                v,
                scale=scale,
                enable_gqa=enable_gqa,
                attention_masks=attention_masks,
                positions=positions,
                segment_ids=segment_ids,
                **kwargs,
            )

        q_mesh = q_placements = None
        if isinstance(q, DTensor):
            q_mesh, q_placements = q.device_mesh, q.placements
            q = q.to_local()
        k = k.to_local() if isinstance(k, DTensor) else k
        v = v.to_local() if isinstance(v, DTensor) else v
        positions = positions.to_local() if isinstance(positions, DTensor) else positions
        segment_ids = segment_ids.to_local() if isinstance(segment_ids, DTensor) else segment_ids
        if segment_ids is None and positions is not None:
            segment_ids = segment_ids_from_positions(positions)

        out = splash_sdpa(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            segment_ids=segment_ids,
            scale=scale,
            is_causal=True,
            local_window_size=self.local_window_size,
            enable_gqa=enable_gqa,
            block_q=block,
            block_kv=block,
            block_dkv=block,
            block_kv_compute=block,
            block_q_dkv=block,
            block_kv_dkv=block,
            block_kv_dkv_compute=block,
        ).transpose(1, 2)
        if q_mesh is not None:
            out = DTensor.from_local(out, q_mesh, q_placements)
        return out


def apply_splash_attention_tpu(model_parts: list[torch.nn.Module]) -> int:
    """Swaps every attention module's ``inner_attention`` for ``TPUSplashAttention``.

    The splash kernel takes the per-token ``segment_ids`` that ``Decoder.forward`` derives
    from ``positions`` (which restart at 0 for every packed sequence), so it applies the
    same causal + document-boundary mask as the dense [1, 1, S, S] mask built in
    ``pad_packed_inputs_for_tpu`` without materializing it. Bucket padding tokens have
    position 0, so each one becomes its own segment and only attends to itself; their
    outputs are discarded.

    Must run before the first forward: per-block ``torch.compile`` traces lazily, so the
    swapped module is what gets compiled. Returns the number of replaced modules.
    """
    replaced = 0
    for model in model_parts:
        targets = [
            m
            for m in model.modules()
            if isinstance(getattr(m, "inner_attention", None), torch.nn.Module)
            and not isinstance(m.inner_attention, TPUSplashAttention)
        ]
        for module in targets:
            inner = module.inner_attention
            window = getattr(inner, "window_size", None)
            local_window_size = None
            if isinstance(window, tuple) and len(window) == 2 and window[0] >= 0:
                local_window_size = int(window[0])
            module.inner_attention = TPUSplashAttention(inner, local_window_size=local_window_size)
            replaced += 1
    if replaced == 0:
        raise ValueError("use_splash_attention=True but no module with an `inner_attention` was found")
    logger.warning(f"Splash attention enabled on TPU: replaced {replaced} inner attention modules")
    return replaced


def compute_global_batch_num_tokens(data: TensorDict, dp_group, tp_size: int) -> Any:
    """Computes global batch token count for loss normalization on TPU.

    On TPU, performing CPU-side all-reduce for loss normalization avoids graph desynchronization
    and JIT compilation lockups across ranks.
    """
    batch_num_tokens = data["loss_mask"].sum().cpu()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
        batch_num_tokens = batch_num_tokens / tp_size
    return batch_num_tokens.item()


def align_micro_batch_shapes_across_ranks(micro_batches: list[TensorDict]) -> None:
    """Make every rank pad micro-batch ``i`` to the same packed length and response-length bucket.

    Each rank buckets its own packed token count, so ranks normally run differently shaped programs for the
    same micro-batch. That is fine while FSDP collectives run as their own eager programs (FSDP2 hooks), but
    SimpleFSDP traces the all-gather/reduce-scatter into each compiled (``spmd_safe``) TransformerBlock, and
    collectives inside a program require every participant to run the same program: mismatched shapes halt
    the TPU (``sync_flag_public_access_error``). One CPU all-reduce(MAX) over all micro-batches' buckets (the
    same CPU-side collective ``compute_global_batch_num_tokens`` uses) gives each micro-batch a common
    ``tpu_padded_seq_len`` / ``tpu_max_response_len`` that :func:`pad_packed_inputs_for_tpu` pads up to.
    Requires the same number of micro-batches on every rank (``same_micro_num_in_dp``).
    """
    from verl.utils import tensordict_utils as tu

    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return
    bucket_size = get_tpu_seq_bucket_size()
    lens = []
    for mb in micro_batches:
        ids = mb["input_ids"]
        n_tokens = ids.values().numel() if getattr(ids, "is_nested", False) else ids.numel()
        resp = mb["responses"] if "responses" in mb.keys() else None
        max_resp = int(resp.offsets().diff().max().item()) if getattr(resp, "is_nested", False) else 0
        lens += [bucket_length(n_tokens, bucket_size), bucket_length(max_resp, bucket_size) if max_resp else 0]
    lens = torch.tensor(lens, dtype=torch.int64)
    torch.distributed.all_reduce(lens, op=torch.distributed.ReduceOp.MAX)
    lens = lens.tolist()
    for i, mb in enumerate(micro_batches):
        tu.assign_non_tensor_data(mb, "tpu_padded_seq_len", int(lens[2 * i]))
        tu.assign_non_tensor_data(mb, "tpu_max_response_len", int(lens[2 * i + 1]))


def synchronize_tpu_loss(loss: torch.Tensor):
    """Materializes forward graph loss without blocking to split XLA forward and backward compilation passes."""
    try:
        from torch_tpu._internal.sync import synchronize

        synchronize(loss, wait=False)
    except ImportError:
        pass


def tpu_eager_mode_context(mode: str | None):
    """Context manager running the enclosed ops under the torch_tpu eager ``mode`` (no-op for None).

    Mirrors torchtitan's ``tpu_config.eager_mode`` (``torchtitan/experiments/tpu/gmain.py``), which the
    afmv7 / qwen3 TPU recipes set to ``DEFER_AND_FUSE``: ops outside the compiled blocks are deferred
    and compiled into fused XLA programs at the next materialization point instead of being launched
    one XLA program per op (the torch_tpu default, ``DEFER_NEVER``). A context manager rather than a
    global setting so it applies to whichever Ray actor thread runs the RPC.
    """
    if mode is None:
        return nullcontext()
    from torch_tpu._internal import execution_mode

    return execution_mode.set_eager_mode(getattr(execution_mode.EagerMode, mode))


def get_tpu_seq_bucket_size() -> int:
    """Returns the token bucketing multiple for TPU sequence packing (default 256)."""
    return int(os.getenv("VERL_TPU_SEQ_BUCKET_SIZE", "256"))


def bucket_length(length: int, bucket_size: int | None = None) -> int:
    """Rounds a positive sequence length up to the nearest multiple of `bucket_size`."""
    if bucket_size is None:
        bucket_size = get_tpu_seq_bucket_size()
    if bucket_size <= 1:
        return max(1, int(length))
    return max(bucket_size, ((int(length) + bucket_size - 1) // bucket_size) * bucket_size)


def pad_packed_inputs_for_tpu(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    micro_batch: TensorDict,
    device: Any,
    build_attention_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """Pads 1D packed sequence inputs on CPU to a fixed bucket multiple and transfers to TPU.

    On `torch_tpu` (XLA), passing or padding unbucketed packed token counts `orig_seq_len` on the
    TPU device causes XLA to compile and cache a new HLO executable in host CPU RAM for every
    distinct sequence length. Padding `input_ids`, `position_ids`, `labels`, and the 4D causal
    document mask on CPU to multiples of `bucket_size` (default 256) before H2D transfer ensures
    the TPU only ever sees static bucket shapes across the entire training run.

    With ``build_attention_mask=False`` (splash attention, which derives document boundaries
    from ``positions``) the dense mask is skipped and ``None`` is returned in its place.
    """
    from verl.utils import tensordict_utils as tu

    bucket_size = get_tpu_seq_bucket_size()
    input_ids_cpu = input_ids.values().detach().cpu().unsqueeze(0)
    if position_ids.dim() == 3:
        position_ids_cpu = position_ids.values().detach().cpu().unsqueeze(1)
    else:
        position_ids_cpu = position_ids.values().detach().cpu().unsqueeze(0)

    labels_cpu = torch.roll(input_ids_cpu, shifts=-1, dims=1)

    orig_seq_len = int(input_ids_cpu.shape[1])
    # tpu_padded_seq_len: common length across ranks from align_micro_batch_shapes_across_ranks (SimpleFSDP).
    padded_seq_len = max(
        bucket_length(orig_seq_len, bucket_size),
        int(tu.get_non_tensor_data(data=micro_batch, key="tpu_padded_seq_len", default=0) or 0),
    )
    pad_len = padded_seq_len - orig_seq_len

    pos_2d_cpu = position_ids_cpu[0] if position_ids_cpu.dim() == 3 else position_ids_cpu
    if pad_len > 0:
        input_ids_cpu = F.pad(input_ids_cpu, (0, pad_len), value=0)
        labels_cpu = F.pad(labels_cpu, (0, pad_len), value=0)
        position_ids_cpu = F.pad(position_ids_cpu, (0, pad_len), value=0)
        pos_2d_cpu = F.pad(pos_2d_cpu, (0, pad_len), value=0)

    # Build static 4D causal + document-boundary mask [1, 1, padded_seq_len, padded_seq_len] on CPU
    attention_mask_cpu = None
    if build_attention_mask:
        if getattr(input_ids, "is_nested", False):
            seq_lens = input_ids.offsets().diff().detach().cpu()
            seq_ids_1d = torch.repeat_interleave(torch.arange(1, len(seq_lens) + 1, dtype=torch.int64), seq_lens)
            if pad_len > 0:
                seq_ids_1d = F.pad(seq_ids_1d, (0, pad_len), value=0)
            seq_ids = seq_ids_1d.unsqueeze(0)
        else:
            first_dummy = pos_2d_cpu[:, :1] - 1
            boundary = torch.diff(pos_2d_cpu, prepend=first_dummy, dim=-1) != 1
            boundary[:, 0] = True
            seq_ids = boundary.cumsum(dim=-1)
        idx = torch.arange(padded_seq_len, dtype=seq_ids.dtype).unsqueeze(0)
        valid = idx < orig_seq_len
        seq_ids = torch.where(valid, seq_ids, -idx - 1)
        same_seq_mask = seq_ids.unsqueeze(2) == seq_ids.unsqueeze(1)
        causal_mask = idx.unsqueeze(2) >= idx.unsqueeze(1)
        attention_mask_cpu = (same_seq_mask & causal_mask).unsqueeze(1)

    # Bucket max_response_len on micro_batch so ppo_loss operates on static bucketed shapes.
    if "responses" in micro_batch.keys() and getattr(micro_batch["responses"], "is_nested", False):
        resp_lens = micro_batch["responses"].offsets().diff().cpu()
        raw_max_resp = int(resp_lens.max().item())
        max_resp = max(
            bucket_length(raw_max_resp, bucket_size),
            int(tu.get_non_tensor_data(data=micro_batch, key="tpu_max_response_len", default=0) or 0),
        )
        tu.assign_non_tensor_data(micro_batch, "max_response_len", max_resp)

    return (
        input_ids_cpu.to(device=device).contiguous(),
        position_ids_cpu.to(device=device).contiguous(),
        labels_cpu.to(device=device).contiguous(),
        attention_mask_cpu.to(device=device).contiguous() if attention_mask_cpu is not None else None,
        orig_seq_len,
    )


def tpu_no_padding_2_padding(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Extracts and pads response tokens from a bucketed 1D tensor using static-shape index gather on TPU."""
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding

    padded_values = getattr(tensor, "_tpu_padded_values", None)
    if padded_values is None:
        return no_padding_2_padding(tensor, data)

    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    if not (getattr(prompt_ids, "is_nested", False) and getattr(response_ids, "is_nested", False)):
        return no_padding_2_padding(tensor, data)

    prompt_lens = prompt_ids.offsets().diff().cpu()
    response_lens = response_ids.offsets().diff().cpu()
    seq_offsets = (prompt_lens + response_lens).cumsum(dim=0)

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
    if max_response_len < 0:
        max_response_len = bucket_length(int(response_lens.max().item()))
    else:
        max_response_len = bucket_length(int(max_response_len))
        tu.assign_non_tensor_data(data, "max_response_len", max_response_len)

    bsz = int(response_lens.shape[0])
    col_idx = torch.arange(max_response_len, dtype=torch.int64).unsqueeze(0)  # [1, max_response_len]
    starts = (seq_offsets - response_lens - 1).to(torch.int64).unsqueeze(1)  # [bsz, 1]
    valid_mask_cpu = col_idx < response_lens.unsqueeze(1)  # [bsz, max_response_len]
    gather_idx_cpu = torch.where(valid_mask_cpu, starts + col_idx, torch.zeros_like(col_idx)).clamp(
        min=0, max=max(0, int(padded_values.shape[0]) - 1)
    )

    device = padded_values.device
    gather_idx = gather_idx_cpu.to(device=device)
    valid_mask = valid_mask_cpu.to(device=device, dtype=padded_values.dtype)
    values_2d = padded_values.unsqueeze(0).expand(bsz, -1)
    return torch.gather(values_2d, 1, gather_idx) * valid_mask


def safe_to_padded_tensor(nt: Any, padding: Any = 0, output_size: Any = None) -> torch.Tensor:
    """Safely converts a NestedTensor to a padded dense tensor on TPU using CPU-side assembly.

    Assembling the small padded response tensors on CPU avoids compiling a new
    `tt_jit_jagged_to_padded_dense_forward` HLO executable for every unbucketed jagged length.
    """
    from verl.utils.device import get_device_id

    if not getattr(nt, "is_nested", False):
        if isinstance(nt, torch.Tensor) and nt.device.type == "cpu":
            return nt.to(device=get_device_id())
        return nt
    target_device = get_device_id() if nt.device.type == "cpu" else nt.device
    values_cpu = nt.values().detach().cpu()
    offsets_cpu = nt.offsets().detach().cpu()
    batch_size = int(offsets_cpu.shape[0]) - 1
    if batch_size <= 0:
        return torch.empty(output_size if output_size is not None else (0,), device=target_device, dtype=nt.dtype)
    lengths = offsets_cpu.diff().tolist()
    trailing_dims = tuple(values_cpu.shape[1:])
    if output_size is None:
        max_len = bucket_length(max(lengths))
        output_size = (batch_size, max_len, *trailing_dims)
    out_cpu = torch.full(output_size, padding, dtype=values_cpu.dtype)
    for i in range(batch_size):
        start = int(offsets_cpu[i].item())
        length = int(lengths[i])
        if length > 0:
            out_cpu[i, :length] = values_cpu[start : start + length]
    return out_cpu.to(device=target_device)


def select_and_to_padded_tensor(data: TensorDict, *fields: str) -> TensorDict:
    """Selects fields from a TensorDict and converts NestedTensors to bucket-padded dense tensors on TPU."""
    from verl.utils import tensordict_utils as tu
    from verl.utils.device import get_device_id

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
    if max_response_len is not None and int(max_response_len) > 0:
        max_response_len = bucket_length(int(max_response_len))
    else:
        max_response_len = None

    target_device = get_device_id()
    padded_dict = {}
    for k in fields:
        if k in data.keys():
            val = data[k]
            if getattr(val, "is_nested", False):
                output_size = None
                if max_response_len is not None:
                    trailing_dims = tuple(val.values().shape[1:])
                    output_size = (int(data.batch_size[0]), max_response_len, *trailing_dims)
                padded_dict[k] = safe_to_padded_tensor(val, output_size=output_size)
            elif isinstance(val, torch.Tensor) and val.device.type == "cpu":
                padded_dict[k] = val.to(device=target_device)
            else:
                padded_dict[k] = val
    return TensorDict(padded_dict, batch_size=data.batch_size)


class _VocabParallelLogprobsFn(torch.autograd.Function):
    """Computes log-probabilities from Shard(-1) vocab-parallel logits without full-vocab backward all_gather."""

    @staticmethod
    def forward(
        ctx,
        local_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        tp_group: torch.distributed.ProcessGroup,
        tp_rank: int,
    ) -> torch.Tensor:
        orig_dtype = local_logits.dtype
        logits_fp32 = local_logits.float()
        if temperature != 1.0:
            logits_fp32 = logits_fp32 / float(temperature)

        vocab_per_rank = int(logits_fp32.size(-1))
        vocab_start = tp_rank * vocab_per_rank
        vocab_end = vocab_start + vocab_per_rank

        # 1. Global max logit across TP group
        max_logits = torch.max(logits_fp32, dim=-1, keepdim=True).values
        torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        # 2. Shifted exponentials and global partition function Z
        shifted_logits = logits_fp32 - max_logits
        exp_logits = torch.exp(shifted_logits)
        sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
        torch.distributed.all_reduce(sum_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        # 3. Target logit on the owning TP shard
        target_mask = (labels >= vocab_start) & (labels < vocab_end)
        local_labels = torch.where(target_mask, labels - vocab_start, torch.zeros_like(labels)).clamp(
            0, vocab_per_rank - 1
        )
        gathered_shifted = torch.gather(shifted_logits, dim=-1, index=local_labels.unsqueeze(-1)).squeeze(-1)
        target_shifted = torch.where(target_mask, gathered_shifted, torch.zeros_like(gathered_shifted))
        torch.distributed.all_reduce(target_shifted, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        log_probs = target_shifted - torch.log(sum_exp.squeeze(-1))
        softmax_local = (exp_logits / sum_exp).to(orig_dtype)
        ctx.save_for_backward(softmax_local, local_labels, target_mask)
        ctx.temperature = float(temperature)
        # Keep fp32 like the non-TP path; bf16 log-probs add noise to the PPO ratio.
        return log_probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        softmax_local, local_labels, target_mask = ctx.saved_tensors
        grad_logits = -softmax_local.clone()
        one_hot_update = target_mask.to(grad_logits.dtype).unsqueeze(-1)
        grad_logits.scatter_add_(-1, local_labels.unsqueeze(-1), one_hot_update)
        scale = grad_output.unsqueeze(-1).to(grad_logits.dtype)
        if ctx.temperature != 1.0:
            scale = scale / ctx.temperature
        grad_logits.mul_(scale)
        return grad_logits, None, None, None, None


def vocab_parallel_logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Computes token logprobs directly from a Shard(-1) DTensor on TPU without calling full_tensor()."""
    from torch.distributed.tensor import DTensor

    if isinstance(logits, DTensor):
        if tp_group is None:
            tp_mesh = logits.device_mesh
            tp_group = tp_mesh.get_group()
        tp_rank = torch.distributed.get_rank(tp_group)
        local_logits = logits.to_local()
    else:
        assert tp_group is not None
        tp_rank = torch.distributed.get_rank(tp_group)
        local_logits = logits
    return _VocabParallelLogprobsFn.apply(local_logits, labels, temperature, tp_group, tp_rank)

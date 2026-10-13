# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import os
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tensordict import TensorDict

import verl.workers.engine.torchtitan.tpu_utils as tpu_utils
from verl.workers.config.engine import TorchtitanEngineConfig
from verl.workers.engine.torchtitan.tpu_utils import (
    TPUSplashAttention,
    _all_to_all_head_to_seq_blnh,
    _all_to_all_seq_to_head_bnlh,
    apply_splash_attention_tpu,
    compute_global_batch_num_tokens,
    gather_cp_1d,
    pad_packed_inputs_for_tpu,
)


def _reference_splash_sdpa(
    q_bnlh: torch.Tensor,
    k_bnlh: torch.Tensor,
    v_bnlh: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Reference causal + document-segment SDPA on (B, H_q, S, D) with GQA support."""
    bsz, h_q, seq_len, head_dim = q_bnlh.shape
    h_kv = k_bnlh.shape[1]
    if h_q != h_kv:
        k_bnlh = k_bnlh.repeat_interleave(h_q // h_kv, dim=1)
        v_bnlh = v_bnlh.repeat_interleave(h_q // h_kv, dim=1)
    idx = torch.arange(seq_len, device=q_bnlh.device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # [S, S]
    if segment_ids is not None:
        same_seg = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)  # [B, S, S]
        mask = (same_seg & causal.unsqueeze(0)).unsqueeze(1)  # [B, 1, S, S]
    else:
        mask = causal.unsqueeze(0).unsqueeze(0)
    return F.scaled_dot_product_attention(q_bnlh, k_bnlh, v_bnlh, attn_mask=mask, scale=scale)


def _cp_worker(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        tpu_utils.splash_sdpa = _reference_splash_sdpa
        torch.manual_seed(42)

        bsz, seq_full, n_q_heads, n_kv_heads, head_dim = 1, 512, 8, 4, 16
        seq_local = seq_full // world_size

        q_full = torch.randn(bsz, seq_full, n_q_heads, head_dim, dtype=torch.float32, requires_grad=True)
        k_full = torch.randn(bsz, seq_full, n_kv_heads, head_dim, dtype=torch.float32, requires_grad=True)
        v_full = torch.randn(bsz, seq_full, n_kv_heads, head_dim, dtype=torch.float32, requires_grad=True)
        grad_out_full = torch.randn(bsz, seq_full, n_q_heads, head_dim, dtype=torch.float32)

        # Two packed documents of lengths 200 and 312
        seg_ids = torch.zeros(bsz, seq_full, dtype=torch.int32)
        seg_ids[:, 200:] = 1

        # 1. Single-rank (CP=1) reference
        out_cp1 = _reference_splash_sdpa(
            q_full.transpose(1, 2).contiguous(),
            k_full.transpose(1, 2).contiguous(),
            v_full.transpose(1, 2).contiguous(),
            segment_ids=seg_ids,
        ).transpose(1, 2)
        loss_cp1 = (out_cp1 * grad_out_full).sum()
        loss_cp1.backward()
        dq_ref = q_full.grad.chunk(world_size, dim=1)[rank].detach().clone()
        dk_ref = k_full.grad.chunk(world_size, dim=1)[rank].detach().clone()
        dv_ref = v_full.grad.chunk(world_size, dim=1)[rank].detach().clone()

        # 2. Multi-rank Ulysses CP (cp_size = world_size)
        q_local = q_full.detach().chunk(world_size, dim=1)[rank].clone().requires_grad_(True)
        k_local = k_full.detach().chunk(world_size, dim=1)[rank].clone().requires_grad_(True)
        v_local = v_full.detach().chunk(world_size, dim=1)[rank].clone().requires_grad_(True)

        q_bnlh = _all_to_all_seq_to_head_bnlh(q_local, dist.group.WORLD, world_size)
        k_bnlh = _all_to_all_seq_to_head_bnlh(k_local, dist.group.WORLD, world_size)
        v_bnlh = _all_to_all_seq_to_head_bnlh(v_local, dist.group.WORLD, world_size)
        assert q_bnlh.shape == (bsz, n_q_heads // world_size, seq_full, head_dim)
        assert k_bnlh.shape == (bsz, n_kv_heads // world_size, seq_full, head_dim)

        out_bnlh = _reference_splash_sdpa(q_bnlh, k_bnlh, v_bnlh, segment_ids=seg_ids)
        out_local = _all_to_all_head_to_seq_blnh(out_bnlh, dist.group.WORLD, world_size)
        assert out_local.shape == (bsz, seq_local, n_q_heads, head_dim)

        # Verify 1D log_prob gather across CP group preserves forward values and backward local gradients
        token_scores_local = out_local.sum(dim=(-1, -2)).squeeze(0)  # [seq_local]
        token_scores_full = gather_cp_1d(token_scores_local, dist.group.WORLD)
        assert token_scores_full.shape == (seq_full,)

        out_cp1_local = out_cp1.detach().chunk(world_size, dim=1)[rank]
        assert torch.allclose(out_local, out_cp1_local, atol=1e-5, rtol=1e-5)
        assert torch.allclose(token_scores_full, out_cp1.detach().sum(dim=(-1, -2)).squeeze(0), atol=1e-5, rtol=1e-5)

        grad_local = grad_out_full.chunk(world_size, dim=1)[rank]
        (out_local * grad_local).sum().backward()

        assert torch.allclose(q_local.grad, dq_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(k_local.grad, dk_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(v_local.grad, dv_ref, atol=1e-5, rtol=1e-5)

        # Also verify apply_splash_attention_tpu wires cp_mesh onto inner_attention
        dummy_part = torch.nn.Module()
        dummy_part.attn = torch.nn.Module()
        dummy_part.attn.inner_attention = torch.nn.Identity()
        fake_mesh = SimpleNamespace(size=lambda: world_size, get_group=lambda: dist.group.WORLD)
        replaced = apply_splash_attention_tpu([dummy_part], cp_mesh=fake_mesh)
        assert replaced == 1
        assert isinstance(dummy_part.attn.inner_attention, TPUSplashAttention)
        assert dummy_part.attn.inner_attention.cp_mesh is fake_mesh
    finally:
        dist.destroy_process_group()


class TestTPUSplashUlyssesCP(unittest.TestCase):
    def test_torchtitan_engine_config_allows_cp_with_simple_fsdp(self):
        cfg = TorchtitanEngineConfig(use_simple_fsdp=True, context_parallel_size=8, data_parallel_shard_size=4)
        self.assertEqual(cfg.context_parallel_size, 8)
        self.assertTrue(cfg.use_simple_fsdp)

        with self.assertRaisesRegex(AssertionError, "use_simple_fsdp supports FSDP/HSDP"):
            TorchtitanEngineConfig(use_simple_fsdp=True, tensor_parallel_size=2, context_parallel_size=4)

    def test_pad_packed_inputs_for_tpu_cp_slicing_and_bucket_alignment(self):
        with patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "6144"}):
            doc1_len, doc2_len = 10000, 7500
            total_len = doc1_len + doc2_len
            vals = torch.arange(1, total_len + 1, dtype=torch.int64)
            pos = torch.cat([torch.arange(doc1_len), torch.arange(doc2_len)]).to(torch.int64)
            offsets = torch.tensor([0, doc1_len, total_len], dtype=torch.int64)

            input_ids_nt = torch.nested.nested_tensor_from_jagged(vals, offsets)
            pos_nt = torch.nested.nested_tensor_from_jagged(pos, offsets)
            mb = TensorDict({}, batch_size=[2])

            cp_size = 8
            chunks_ids, chunks_pos, chunks_labels = [], [], []
            seg_ids_ref = None
            for cp_rank in range(cp_size):
                ids_r, pos_r, labels_r, seg_r, orig_len = pad_packed_inputs_for_tpu(
                    input_ids=input_ids_nt,
                    position_ids=pos_nt,
                    micro_batch=mb,
                    device="cpu",
                    build_attention_mask=False,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                )
                self.assertEqual(orig_len, total_len)
                self.assertEqual(tuple(ids_r.shape), (1, 18432 // cp_size))
                self.assertEqual(tuple(pos_r.shape), (1, 18432 // cp_size))
                self.assertEqual(tuple(labels_r.shape), (1, 18432 // cp_size))
                self.assertIsNotNone(seg_r)
                self.assertEqual(seg_r.dtype, torch.int32)
                self.assertEqual(tuple(seg_r.shape), (1, 18432))
                if seg_ids_ref is None:
                    seg_ids_ref = seg_r
                else:
                    self.assertTrue(torch.equal(seg_r, seg_ids_ref))
                chunks_ids.append(ids_r)
                chunks_pos.append(pos_r)
                chunks_labels.append(labels_r)

            full_ids = torch.cat(chunks_ids, dim=1)
            full_pos = torch.cat(chunks_pos, dim=1)
            full_labels = torch.cat(chunks_labels, dim=1)

            ids_cp1, pos_cp1, labels_cp1, mask_cp1, _ = pad_packed_inputs_for_tpu(
                input_ids=input_ids_nt,
                position_ids=pos_nt,
                micro_batch=mb,
                device="cpu",
                build_attention_mask=False,
                cp_size=1,
                cp_rank=0,
            )
            self.assertIsNone(mask_cp1)
            self.assertTrue(torch.equal(full_ids, ids_cp1))
            self.assertTrue(torch.equal(full_pos, pos_cp1))
            self.assertTrue(torch.equal(full_labels, labels_cp1))
            self.assertTrue(torch.all(seg_ids_ref[0, :doc1_len] == 0))
            self.assertTrue(torch.all(seg_ids_ref[0, doc1_len:total_len] == 1))

    def test_compute_global_batch_num_tokens_scales_by_tp_and_cp(self):
        data = TensorDict({"loss_mask": torch.ones(120, dtype=torch.int64)}, batch_size=[120])

        def _fake_all_reduce(tensor, op=None):
            tensor.mul_(8)

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "all_reduce", side_effect=_fake_all_reduce),
        ):
            tokens = compute_global_batch_num_tokens(data, dp_group=None, tp_size=1, cp_size=4)
            self.assertEqual(tokens, 240.0)

    def test_ulysses_cp_forward_and_backward_parity_gloo(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        world_size = 4
        mp.spawn(_cp_worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    unittest.main()

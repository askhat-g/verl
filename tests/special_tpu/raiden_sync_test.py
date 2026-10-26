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
"""Raiden weight-sync correctness on TPU: real TorchTitan trainer -> Raiden -> fake samplers (no vLLM).

The TPU counterpart of tests/checkpoint_engine/test_correctness_on_gpu.py. A forward-only TorchTitan trainer on the
first TPU slice sends its FSDP shards with the real RaidenCheckpointEngine / RaidenController / tpu-sync transport.
Fake rollout replicas on the second slice run the real vLLMRaidenWorkerExtension.init_raiden_sync_on_worker (staging
buffers, TP sharding specs, controller registration) in single-chip processes, without vLLM. After every sync each
sampler rank compares the tensors it received bit for bit with its TP slice of the HF checkpoint (the trainer does not
train, so the weights must equal the checkpoint); the engine's own parity checks (verify_parity=all) run as well.

Submitted as a Ray job to a cluster with two TPU slices (TPU CI tier v6e-8chip: 2 x v6e-4):
    python3 tests/special_tpu/raiden_sync_test.py
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from types import SimpleNamespace

import ray
import torch

from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineRegistry
from verl.checkpoint_engine.raiden_checkpoint_engine import RaidenParityCheck
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.config import (
    CheckpointEngineConfig,
    HFModelConfig,
    TorchtitanEngineConfig,
    TorchtitanOptimizerConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine_workers import TrainingWorker

MODEL_PATH = os.environ.get("RAIDEN_TEST_MODEL_PATH", "/data/jialei/assets/hf/Qwen3-0.6B")
NUM_SYNCS = 3
# (num_replicas, tp_size, release_buffers_after_sync)
CASES = [(1, 4, True), (2, 2, True), (1, 4, False)]
ROW_PARALLEL_SUFFIXES = (".o_proj.weight", ".down_proj.weight")
KV_PROJ_SUFFIXES = (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")


def log(msg: str) -> None:
    print(f"[raiden test] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# Trainer: forward-only TorchTitan engine + RaidenCheckpointEngine
# ---------------------------------------------------------------------------------------------


class RaidenTrainerWorker(TrainingWorker):
    """TrainingWorker that sends its local FSDP shards with RaidenCheckpointEngine, as ActorRolloutRefWorker does."""

    def __init__(self, config: TrainingWorkerConfig) -> None:
        super().__init__(config)
        self.checkpoint_engine = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_raiden(self, **engine_kwargs):
        """(Re)create the Raiden engine so that every case registers with its own RaidenController."""
        if self.checkpoint_engine is not None:
            self.checkpoint_engine.finalize()
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            "raiden", bucket_size=0, is_master=torch.distributed.get_rank() == 0, **engine_kwargs
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        # Same call as ActorRolloutRefWorker.update_weights for engines with consumes_training_engine=True.
        return await self.checkpoint_engine.send_weights(self.engine, global_steps=global_steps) or {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def release_raiden_sync_buffers(self):
        return self.checkpoint_engine.release_sync_buffers() or {}


# ---------------------------------------------------------------------------------------------
# Fake sampler: one single-chip process per TP rank, running the real Raiden sampler code
# ---------------------------------------------------------------------------------------------


def _pin_single_chip_env() -> None:
    """Restrict this process to the one TPU chip Ray assigned to it (same recipe as tpu_smoke_test.py)."""
    chip_ids = ray.get_runtime_context().get_accelerator_ids().get("TPU", [])
    assert len(chip_ids) == 1, f"expected exactly one assigned TPU chip, got {chip_ids}"
    port = 8471 + int(chip_ids[0])
    os.environ.update(
        {
            "TPU_VISIBLE_CHIPS": str(chip_ids[0]),
            "TPU_VISIBLE_DEVICES": str(chip_ids[0]),
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
            "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_PROCESS_BOUNDS": "1,1,1",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "localhost",
            "CLOUD_TPU_TASK_ID": "0",
            "TORCH_TPU_TOPOLOGY": "1,1,1",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"localhost:{port}",
            "TPU_PROCESS_ADDRESSES": f"localhost:{port}",
            "TPU_PROCESS_PORT": str(port),
        }
    )


def expected_tp_slice(name: str, full: torch.Tensor, tp_size: int, tp_rank: int, num_kv_heads: int) -> torch.Tensor:
    """The slice of an HF tensor that vLLM's TP rank ``tp_rank`` holds, i.e. what Raiden must deliver to it."""
    if full.dim() != 2:
        return full
    if name.endswith(ROW_PARALLEL_SUFFIXES):
        return full.chunk(tp_size, dim=1)[tp_rank]
    if num_kv_heads < tp_size and name.endswith(KV_PROJ_SUFFIXES):
        return full.chunk(num_kv_heads, dim=0)[tp_rank // (tp_size // num_kv_heads)]
    if full.shape[0] % tp_size == 0:
        return full.chunk(tp_size, dim=0)[tp_rank]
    return full


@ray.remote(num_cpus=1)
class RaidenSamplerWorker:
    """Stands in for one vLLM TP worker.

    ``init_raiden_sync_on_worker`` and ``get_model_weights_stats`` run the real vLLMRaidenWorkerExtension code on
    this object. ``install_raiden_weights`` replaces only the copy into vLLM parameters: it runs the H2D and then
    compares the received staging tensors bit for bit with this rank's slice of the checkpoint.
    """

    def __init__(self, tp_rank: int, tp_size: int, model_path: str) -> None:
        _pin_single_chip_env()  # before anything initializes the TPU runtime
        import torch_tpu  # noqa: F401
        import vllm.distributed
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path)
        self.rank = tp_rank
        self.tp_size = tp_size
        self.model_path = model_path
        self.num_kv_heads = hf_config.num_key_value_heads
        self.tie_word_embeddings = bool(getattr(hf_config, "tie_word_embeddings", False))
        # What vLLMRaidenWorkerExtension reads from the vLLM worker.
        self.model_runner = SimpleNamespace(model=torch.nn.Module())
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
            model_config=SimpleNamespace(get_total_num_kv_heads=lambda: self.num_kv_heads),
        )
        vllm.distributed.get_tensor_model_parallel_rank = lambda: tp_rank
        self._expected = None
        self.last_exact = None

    @staticmethod
    def _extension():
        from verl.workers.rollout.vllm_rollout.tpu_utils import vLLMRaidenWorkerExtension

        return vLLMRaidenWorkerExtension

    def _get_vllm_model(self):
        return self._extension()._get_vllm_model(self)

    def init_raiden_sync_on_worker(self, parallelism: int = 8, job_name: str = "sampler") -> bool:
        return self._extension().init_raiden_sync_on_worker(self, parallelism=parallelism, job_name=job_name)

    def get_model_weights_stats(self, include_shards: bool = False) -> dict:
        return self._extension().get_model_weights_stats(self, include_shards=include_shards)

    def _expected_tensors(self) -> dict:
        """This rank's slice of every checkpoint tensor the trainer sends (lm_head is not sent for tied models)."""
        if self._expected is None:
            from safetensors import safe_open

            index = os.path.join(self.model_path, "model.safetensors.index.json")
            if os.path.exists(index):
                with open(index) as f:
                    files = sorted(set(json.load(f)["weight_map"].values()))
            else:
                files = ["model.safetensors"]
            expected = {}
            for file in files:
                with safe_open(os.path.join(self.model_path, file), framework="pt") as f:
                    for name in f.keys():
                        if self.tie_word_embeddings and name == "lm_head.weight":
                            continue
                        full = f.get_tensor(name)
                        expected[name] = expected_tp_slice(name, full, self.tp_size, self.rank, self.num_kv_heads)
            self._expected = expected
        return self._expected

    def install_raiden_weights(self, exact_parity: bool = False) -> dict:
        from torch_tpu._internal import sync as torch_tpu_sync

        t0 = time.perf_counter()
        self._raiden_ws.h2d()
        torch_tpu_sync.synchronize(wait=True)
        t_h2d = time.perf_counter() - t0

        expected = self._expected_tensors()
        parity = RaidenParityCheck("exact")
        for name, received in self._raiden_staging.items():
            if name in expected:
                parity.check_exact(name, expected[name], received.cpu())
        unresolved = sorted(set(expected) ^ set(self._raiden_staging))  # missing or unexpected tensors
        self.last_exact = parity.exact_result(self.rank, unresolved)
        return {"total": time.perf_counter() - t0, "h2d": t_h2d, "exact_parity": self.last_exact}

    def last_exact_result(self) -> dict:
        return {"exact_parity": self.last_exact}


class _AsyncRemote:
    """``.remote(**kwargs)`` returning an awaitable, like a Ray actor method."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeRaidenReplica:
    """The parts of a RolloutReplica that CheckpointEngineManager / update_raiden_weights use."""

    def __init__(self, workers: list) -> None:
        self.workers = workers
        self.world_size = len(workers)
        self.global_steps = None
        self.server_handle = SimpleNamespace(
            collective_rpc=_AsyncRemote(self._collective_rpc), set_global_steps=_AsyncRemote(self._set_global_steps)
        )

    async def _collective_rpc(self, method: str, kwargs: dict = None):
        return await asyncio.gather(*[getattr(w, method).remote(**(kwargs or {})) for w in self.workers])

    async def _set_global_steps(self, global_steps):
        self.global_steps = global_steps

    async def abort_all_requests(self, reject_request: bool = False):
        pass

    async def resume_generation(self):
        pass


# ---------------------------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------------------------


def create_trainer(trainer_slice) -> RayWorkerGroup:
    model_config = HFModelConfig(path=MODEL_PATH, use_remove_padding=True)
    engine_config = TorchtitanEngineConfig(
        forward_only=True,
        data_parallel_shard_size=trainer_slice.chips,
        use_simple_fsdp=True,  # as the TPU GRPO recipes
        use_torch_compile=False,
        attn_type="varlen",
        activation_checkpoint="none",
    )
    # forward_only builds no optimizer, but TorchTitanEngine.__init__ still reads the optimizer fields.
    optimizer_config = TorchtitanOptimizerConfig(lr=1e-6, total_training_steps=1)
    config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
    )
    pool = RayResourcePool(
        process_on_nodes=[trainer_slice.chips_per_host] * len(trainer_slice.hosts),
        use_gpu=True,
        name_prefix="raiden_test_trainer",
        accelerator_type=trainer_slice.name,
    )
    wg = RayWorkerGroup(
        resource_pool=pool,
        ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(RaidenTrainerWorker), config=config),
        device_name="tpu",
    )
    wg.reset()  # load the HF checkpoint into the TorchTitan engine
    return wg


def create_replicas(sampler_slice, num_replicas: int, tp_size: int) -> list:
    assert num_replicas * tp_size <= sampler_slice.chips, f"{num_replicas}x TP{tp_size} > {sampler_slice.chips} chips"
    replicas = []
    for i in range(num_replicas):
        host = sampler_slice.hosts[i % len(sampler_slice.hosts)]
        workers = [
            RaidenSamplerWorker.options(resources={"TPU": 1, f"node:{host}": 0.001}).remote(r, tp_size, MODEL_PATH)
            for r in range(tp_size)
        ]
        replicas.append(FakeRaidenReplica(workers))
    return replicas


def run_case(trainer_wg, sampler_slice, num_replicas: int, tp_size: int, release_buffers: bool) -> None:
    engine_kwargs = {"verify_parity": "all", "release_buffers_after_sync": release_buffers}
    trainer_wg.reset_raiden(**engine_kwargs)
    replicas = create_replicas(sampler_slice, num_replicas, tp_size)
    config = CheckpointEngineConfig(backend="raiden", engine_kwargs={"raiden": engine_kwargs})
    try:
        manager = CheckpointEngineManager(config=config, actor_wg=trainer_wg, replicas=replicas)
        for step in range(NUM_SYNCS):
            # Raises RaidenParityError on a mismatch: the engine's exact check at step 0, its norm check after.
            metrics = manager.update_weights(global_steps=step)
            assert metrics["timing_s/tpu-sync/total_sync"] > 0, metrics
            assert all(r.global_steps == step for r in replicas)
            # Every sync, every sampler rank of every replica: received tensors == checkpoint slice, bit for bit.
            results = [ray.get([w.last_exact_result.remote() for w in r.workers]) for r in replicas]
            RaidenParityCheck.report_exact(step, results)
            log(
                f"{num_replicas}xTP{tp_size} release={release_buffers} step {step}: bit-exact, "
                f"total_sync={metrics['timing_s/tpu-sync/total_sync']:.3f}s "
                f"p2p={metrics['timing_s/tpu-sync/p2p_transfer']:.3f}s"
            )
    finally:
        for r in replicas:
            for w in r.workers:
                ray.kill(w)


def main() -> int:
    from verl.plugin.platform import get_platform
    from verl.plugin.platform.platform_tpu_slices import discover_tpu_slices

    # Show the engine's [RAIDEN PARITY ...] / [RAIDEN TELEMETRY ...] lines in the job log.
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s: %(message)s")
    logging.getLogger("verl.checkpoint_engine.raiden_checkpoint_engine").setLevel(logging.INFO)
    # The TPU platform's runtime_env installs patch_ray_worker (Ray's TPU accelerator-id lookup raises IndexError in
    # GKE pods otherwise), as verl's trainers do.
    ray.init(**get_platform().get_ray_init_kwargs())
    slices = discover_tpu_slices()
    log(f"TPU slices: {[(s.name, s.hosts, s.chips_per_host) for s in slices]}")
    if len(slices) < 2:
        log(f"FAILED: needs two TPU slices (trainer + samplers), found {len(slices)}")
        return 1
    trainer_slice, sampler_slice = slices[0], slices[1]
    t0 = time.perf_counter()
    trainer_wg = create_trainer(trainer_slice)
    log(f"trainer ready on {trainer_slice.name} ({trainer_slice.chips} chips) in {time.perf_counter() - t0:.1f}s")

    failed = []
    for num_replicas, tp_size, release_buffers in CASES:
        name = f"{num_replicas}xTP{tp_size}-release={release_buffers}"
        try:
            run_case(trainer_wg, sampler_slice, num_replicas, tp_size, release_buffers)
            log(f"case {name}: PASSED")
        except Exception:
            traceback.print_exc()
            log(f"case {name}: FAILED")
            failed.append(name)
    if failed:
        log(f"FAILED: {len(failed)}/{len(CASES)} cases: {failed}")
        return 1
    log(f"PASSED: {len(CASES)}/{len(CASES)} cases, {NUM_SYNCS} syncs each")
    return 0


if __name__ == "__main__":
    sys.exit(main())

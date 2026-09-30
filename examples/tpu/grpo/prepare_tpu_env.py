#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Cluster-wide pre-flight setup for TPU GRPO runs (model/data caching, eviction, and Ray #66400 patch)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import ray


def _patch_ray_66400_on_host() -> str:
    """Applies ray-project/ray#66400 to the local Ray installation if not already patched."""
    site_packages = Path(ray.__file__).resolve().parent
    worker_path = site_packages / "_private" / "worker.py"
    tpu_path = site_packages / "_private" / "accelerators" / "tpu.py"

    old_worker = """        if self.original_visible_accelerator_ids.get(resource_name, None) is not None:
            original_ids = self.original_visible_accelerator_ids[resource_name]
            assigned_ids = {str(original_ids[i]) for i in assigned_ids}
        return list(assigned_ids)"""
    new_worker = """        if self.original_visible_accelerator_ids.get(resource_name, None) is not None:
            original_ids = self.original_visible_accelerator_ids[resource_name]
            if all(i < len(original_ids) for i in assigned_ids):
                assigned_ids = {str(original_ids[i]) for i in assigned_ids}
            else:
                assigned_ids = {str(x) for x in original_ids[: len(assigned_ids)]}
        return list(assigned_ids)"""

    worker_txt = worker_path.read_text()
    if old_worker in worker_txt:
        worker_path.write_text(worker_txt.replace(old_worker, new_worker, 1))

    old_tpu = """        if env_bool(NOSET_TPU_VISIBLE_CHIPS_ENV_VAR, False):
            return

        num_visible_tpu_chips = len(visible_tpu_chips)
        num_accelerators_on_node = (
            TPUAcceleratorManager.get_current_node_num_accelerators()
        )
        if num_visible_tpu_chips == num_accelerators_on_node:
            # Let the ML framework use the defaults
            os.environ.pop(TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR, None)
            os.environ.pop(TPU_HOST_BOUNDS_ENV_VAR, None)
            return
        os.environ[
            TPUAcceleratorManager.get_visible_accelerator_ids_env_var()
        ] = ",".join([str(i) for i in visible_tpu_chips])
        if num_visible_tpu_chips == 1:
            os.environ[
                TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR
            ] = TPU_CHIPS_PER_HOST_BOUNDS_1_CHIP_CONFIG
            os.environ[TPU_HOST_BOUNDS_ENV_VAR] = TPU_SINGLE_HOST_BOUNDS
        elif num_visible_tpu_chips == 2:
            os.environ[
                TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR
            ] = TPU_CHIPS_PER_HOST_BOUNDS_2_CHIP_CONFIG
            os.environ[TPU_HOST_BOUNDS_ENV_VAR] = TPU_SINGLE_HOST_BOUNDS"""
    new_tpu = """        if env_bool(NOSET_TPU_VISIBLE_CHIPS_ENV_VAR, False):
            return

        physical_chips = sorted({int(device_id) for device_id in visible_tpu_chips})
        stale_subhost_bounds = len(physical_chips) not in (1, 2) and os.environ.get(
            TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR
        ) in (
            TPU_CHIPS_PER_HOST_BOUNDS_1_CHIP_CONFIG,
            TPU_CHIPS_PER_HOST_BOUNDS_2_CHIP_CONFIG,
        )

        if (
            len(visible_tpu_chips)
            == TPUAcceleratorManager.get_current_node_num_accelerators()
        ):
            os.environ.pop(TPU_VISIBLE_CHIPS_ENV_VAR, None)
            if stale_subhost_bounds:
                os.environ.pop(TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR, None)
                os.environ.pop(TPU_HOST_BOUNDS_ENV_VAR, None)
            return

        os.environ[
            TPUAcceleratorManager.get_visible_accelerator_ids_env_var()
        ] = ",".join(str(chip) for chip in physical_chips)
        if len(physical_chips) in (1, 2):
            expected_chip_bounds = (
                TPU_CHIPS_PER_HOST_BOUNDS_1_CHIP_CONFIG
                if len(physical_chips) == 1
                else TPU_CHIPS_PER_HOST_BOUNDS_2_CHIP_CONFIG
            )
            if (
                os.environ.get(TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR)
                != expected_chip_bounds
            ):
                os.environ[TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR] = expected_chip_bounds
                os.environ[TPU_HOST_BOUNDS_ENV_VAR] = TPU_SINGLE_HOST_BOUNDS
            else:
                os.environ.setdefault(TPU_HOST_BOUNDS_ENV_VAR, TPU_SINGLE_HOST_BOUNDS)
        elif stale_subhost_bounds:
            os.environ.pop(TPU_CHIPS_PER_HOST_BOUNDS_ENV_VAR, None)
            os.environ.pop(TPU_HOST_BOUNDS_ENV_VAR, None)"""

    tpu_txt = tpu_path.read_text()
    if old_tpu in tpu_txt:
        tpu_path.write_text(tpu_txt.replace(old_tpu, new_tpu, 1))

    # Fix PyTorch FSDP2 finalize_backward bug with FullAC on PrivateUse1/torch_tpu
    # (line 777 uses torch.accelerator instead of self.device_handle)
    import torch

    fsdp_pg_path = (
        Path(torch.__file__).resolve().parent / "distributed" / "fsdp" / "_fully_shard" / "_fsdp_param_group.py"
    )
    if fsdp_pg_path.is_file():
        pg_txt = fsdp_pg_path.read_text()
        bad_line = "torch.accelerator.current_stream().wait_event(event)"
        good_line = "self.device_handle.current_stream().wait_event(event)"
        if bad_line in pg_txt:
            fsdp_pg_path.write_text(pg_txt.replace(bad_line, good_line))

    # Ensure verl is importable across all spawned Ray / vLLM worker processes.
    # (a) Why needed: Dockerfile.tpu sets PYTHONPATH=/app without /workspace/verl in site-packages.
    #     When vLLM V1 (TP=8) spawns EngineCore subprocesses which schedule cross-node RayWorkerWrapper
    #     actors, they fail with ModuleNotFoundError: No module named 'verl' unless verl.pth is in site-packages.
    # (b) Strictly necessary or Dockerfile-configurable?: Can be fixed directly in Dockerfile.tpu by
    #     setting ENV PYTHONPATH="/workspace/verl:/data/verl:/app" or running pip install -e /workspace/verl --no-deps.
    # (c) Why smaller models (0.6B, 4B) didn't require it: 0.6B/4B run single-host vLLM (TP <= 4, uni/mp
    #     executor) and do not spawn cross-node RayWorkerWrapper actors from inside EngineCore.
    verl_pth = site_packages.parent / "verl.pth"
    verl_pth.write_text("/workspace/verl\n/data/verl\n")

    subprocess.run(["pkill", "-f", "ray::IDLE"], check=False)
    return "patched"


def _is_tpu_worker_node() -> bool:
    """Returns True if this host has TPU devices attached (/dev/vfio/[0-9]*)."""
    vfio = Path("/dev/vfio")
    if not vfio.is_dir():
        return False
    return any(p.name.isdigit() for p in vfio.iterdir())


def _is_model_complete(model_dir: Path, require_weights: bool = True) -> bool:
    """Returns True if model_dir contains config.json, tokenizer.json, and (on TPU nodes) all safetensors shards."""
    if not (model_dir / "config.json").is_file() or not (model_dir / "tokenizer.json").is_file():
        return False
    if not require_weights:
        return True
    safetensors = list(model_dir.glob("*.safetensors"))
    if not safetensors:
        return False
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        import json

        data = json.loads(index_file.read_text())
        expected_files = set(data.get("weight_map", {}).values())
        return all((model_dir / fname).is_file() for fname in expected_files)
    return True


def _prepare_node_storage(repo_id: str, model_name: str) -> dict[str, str]:
    """Ensures /var/data/jialei backs /data/jialei, evicts other models, and downloads target model & GSM8K."""
    import socket

    _patch_ray_66400_on_host()
    is_tpu_node = _is_tpu_worker_node()

    # Store models on root boot disk (/var/data/jialei) instead of 32 GB /tmp tmpfs on head node
    storage_root = Path("/var/data/jialei")
    hf_root = storage_root / "assets" / "hf"
    gsm8k_dir = storage_root / "data" / "gsm8k"
    hf_root.mkdir(parents=True, exist_ok=True)
    gsm8k_dir.mkdir(parents=True, exist_ok=True)

    data_link = Path("/data/jialei")
    data_link.parent.mkdir(parents=True, exist_ok=True)
    if data_link.is_symlink() or not data_link.exists():
        if data_link.is_symlink():
            data_link.unlink()
        data_link.symlink_to(storage_root)
    else:
        # Ensure subdirectories in /data/jialei point to /var/data/jialei
        (data_link / "assets").mkdir(parents=True, exist_ok=True)
        hf_link = data_link / "assets" / "hf"
        if not hf_link.is_symlink():
            shutil.rmtree(hf_link, ignore_errors=True)
            hf_link.symlink_to(hf_root)
        data_sub_link = data_link / "data"
        if not data_sub_link.is_symlink():
            if (data_sub_link / "gsm8k" / "train.parquet").is_file() and not (gsm8k_dir / "train.parquet").is_file():
                shutil.copy2(data_sub_link / "gsm8k" / "train.parquet", gsm8k_dir / "train.parquet")
                shutil.copy2(data_sub_link / "gsm8k" / "test.parquet", gsm8k_dir / "test.parquet")
            shutil.rmtree(data_sub_link, ignore_errors=True)
            data_sub_link.symlink_to(storage_root / "data")

    # 1. Evict any other models in /var/data/jialei/assets/hf and /tmp/models so models NEVER stack up
    for base_dir in (hf_root, Path("/tmp/models")):
        if base_dir.is_dir():
            for child in list(base_dir.iterdir()):
                if child.name != model_name:
                    shutil.rmtree(child, ignore_errors=True)

    # Also clean stale weight cache and HF download cache
    shutil.rmtree("/tmp/verl_weight_cache", ignore_errors=True)
    shutil.rmtree("/var/data/verl_weight_cache", ignore_errors=True)
    shutil.rmtree("/tmp/hf_cache", ignore_errors=True)
    shutil.rmtree(Path.home() / ".cache" / "huggingface" / "hub", ignore_errors=True)

    # 2. If target model exists in /tmp/models/<model_name>, move it into /var/data/jialei/assets/hf/<model_name>
    target_model_dir = hf_root / model_name
    tmp_model_dir = Path("/tmp/models") / model_name
    if not _is_model_complete(target_model_dir, require_weights=is_tpu_node) and _is_model_complete(
        tmp_model_dir, require_weights=is_tpu_node
    ):
        shutil.rmtree(target_model_dir, ignore_errors=True)
        shutil.move(str(tmp_model_dir), str(target_model_dir))
    shutil.rmtree("/tmp/models", ignore_errors=True)

    # 3. Download target model if not already present and complete
    # Use /var/data cache (never /tmp tmpfs, which is only 32 GB and would overflow on 32B's 65.5 GB weights).
    # On the CPU-only head node, only config/tokenizer files are needed (<20 MB).
    if not _is_model_complete(target_model_dir, require_weights=is_tpu_node):
        shutil.rmtree(target_model_dir, ignore_errors=True)
        from huggingface_hub import snapshot_download

        ignore_pats = None if is_tpu_node else ["*.safetensors", "*.bin", "*.pt"]
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_model_dir),
            local_dir_use_symlinks=False,
            cache_dir=str(target_model_dir / ".cache"),
            ignore_patterns=ignore_pats,
            max_workers=4,
        )
        shutil.rmtree(target_model_dir / ".cache", ignore_errors=True)

    # 4. Ensure GSM8K train.parquet and test.parquet exist
    if not (gsm8k_dir / "train.parquet").is_file() or not (gsm8k_dir / "test.parquet").is_file():
        import datasets

        ds = datasets.load_dataset("openai/gsm8k", "main")
        instruction_following = 'Let\'s think step by step and output the final answer after "####".'

        def make_map_fn(split):
            def process_fn(example, idx):
                question_raw = example.pop("question")
                question = question_raw + " " + instruction_following
                answer_raw = example.pop("answer")
                solution = answer_raw.split("#### ")[-1].replace(",", "")
                return {
                    "data_source": "openai/gsm8k",
                    "prompt": [{"role": "user", "content": question}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {"split": split, "index": idx, "answer": answer_raw, "question": question_raw},
                }

            return process_fn

        train_ds = ds["train"].map(function=make_map_fn("train"), with_indices=True)
        test_ds = ds["test"].map(function=make_map_fn("test"), with_indices=True)
        train_ds.to_parquet(str(gsm8k_dir / "train.parquet"))
        test_ds.to_parquet(str(gsm8k_dir / "test.parquet"))
        shutil.rmtree(Path.home() / ".cache" / "huggingface" / "datasets", ignore_errors=True)

    return {
        "hostname": socket.gethostname(),
        "model_dir": str(target_model_dir),
        "gsm8k_train": str(gsm8k_dir / "train.parquet"),
    }


def main() -> None:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TARGET_MODEL_ID", "Qwen/Qwen3-0.6B")
    model_name = repo_id.split("/")[-1]
    print(f"[prepare_tpu_env] Preparing cluster nodes for model {repo_id} ({model_name})...", flush=True)

    ray.init(address="auto", namespace="verl", ignore_reinit_error=True)

    # Kill any stale detached weight registry actors from prior runs
    for actor_name in ("TPUWeightRegistry", "RayWeightRegistry"):
        try:
            actor = ray.get_actor(actor_name, namespace="verl")
            ray.kill(actor, no_restart=True)
        except Exception:
            pass

    alive_nodes = [n for n in ray.nodes() if n.get("Alive")]
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0)
    def prepare_node_task(r_id: str, m_name: str) -> dict[str, str]:
        return _prepare_node_storage(r_id, m_name)

    futures = [
        prepare_node_task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        ).remote(repo_id, model_name)
        for node in alive_nodes
    ]
    results = ray.get(futures)
    for res in results:
        print(f"[prepare_tpu_env] Node {res['hostname']} ready: model={res['model_dir']}", flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()

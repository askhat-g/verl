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
"""Copy a HF checkpoint from a shared (GCS Fuse) path to the same local path on every alive TPU Ray node.

TorchTitan's HF reader and vLLM mmap the safetensors shards. Over GCS Fuse those page faults run at ~20 MB/s
per host (Qwen3-32B: shard 3 of 17 after ~5.5 min), while a parallel sequential copy runs at ~800 MB/s.
Spot TPU slices are recreated with an empty host /tmp, so this runs before every job and skips nodes whose
copy is already complete (same file names and sizes).

Usage: python3 stage_model_local.py <src_dir> <dst_dir> [parallel_files]
"""

import os
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _manifest(path: str) -> dict[str, int]:
    return {
        f: os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if not f.startswith(".") and os.path.isfile(os.path.join(path, f))
    }


@ray.remote(num_cpus=0)
def _stage(src: str, dst: str, parallel: int, expected: dict[str, int]) -> str:
    import socket
    import subprocess

    host = socket.gethostname()
    os.makedirs(dst, exist_ok=True)
    have = _manifest(dst)
    missing = [f for f, size in expected.items() if have.get(f) != size]
    if not missing:
        return f"{host}: already staged ({len(expected)} files)"
    t0 = time.time()
    procs, failed = [], []
    for f in missing:
        while len(procs) >= parallel:
            procs = [p for p in procs if p.poll() is None]
            time.sleep(0.2)
        procs.append(subprocess.Popen(["cp", os.path.join(src, f), os.path.join(dst, f)]))
    for p in procs:
        p.wait()
    have = _manifest(dst)
    failed = [f for f, size in expected.items() if have.get(f) != size]
    if failed:
        raise RuntimeError(f"{host}: staging incomplete, mismatched files: {failed}")
    gb = sum(expected[f] for f in missing) / 1e9
    dt = time.time() - t0
    return f"{host}: copied {len(missing)} files ({gb:.1f} GB) in {dt:.0f}s ({gb / max(dt, 1e-3):.2f} GB/s)"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    parallel = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    if os.path.abspath(src) == os.path.abspath(dst):
        print(f"[stage_model_local] src == dst ({src}); nothing to do")
        return
    expected = _manifest(src)
    if not any(f.endswith(".safetensors") for f in expected):
        raise FileNotFoundError(f"no safetensors found in {src}")
    # CPU-only nodes (e.g. the Ray head, where the TaskRunner builds HFModelConfig / the tokenizer from MODEL_PATH)
    # only need the config, tokenizer and index files, not the weights.
    light = {f: size for f, size in expected.items() if not f.endswith((".safetensors", ".bin", ".pt"))}
    ray.init(address="auto", logging_level="ERROR")
    alive = [n for n in ray.nodes() if n.get("Alive")]
    tpu_nodes = [n for n in alive if n.get("Resources", {}).get("TPU", 0) > 0]
    cpu_nodes = [n for n in alive if n.get("Resources", {}).get("TPU", 0) == 0]
    print(
        f"[stage_model_local] {src} -> {dst}: full copy on {len(tpu_nodes)} TPU nodes "
        f"({sum(expected.values()) / 1e9:.1f} GB), config/tokenizer only on {len(cpu_nodes)} CPU nodes"
    )
    refs = [
        _stage.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)).remote(
            src, dst, parallel, files
        )
        for nodes, files in ((tpu_nodes, expected), (cpu_nodes, light))
        for n in nodes
    ]
    t0 = time.time()
    for line in ray.get(refs):
        print(f"[stage_model_local] {line}")
    print(f"[stage_model_local] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

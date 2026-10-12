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
"""Google TPU platform implementation.

TPU with PyTorch/XLA (torch_tpu) reuses the ``torch.cuda.*`` API surface, so most of
``PlatformCUDA`` works unchanged. This class subclasses ``PlatformCUDA`` and overrides
device-specific environment configuration, resource options, and memory management proxies.
"""

import logging
import os
from typing import Any, Optional

import ray
import torch

from .platform_cuda import PlatformCUDA
from .platform_manager import PlatformRegistry, get_platform
from .platform_tpu_slices import plan_resource_pool_placement, slice_chips_for
from .platform_tpu_workarounds import convert_tensors_to_scalars, patch_ray_worker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Communication port defaults for TPU distributed slice builder meshes
ROLLOUT_BASE_PORT = 8070
TRAINER_BASE_PORT = 8471

# TPU Chip HBM capacities in bytes
HBM_BYTES_TPU_V5P = 95 * 1024 * 1024 * 1024  # 95 GB
HBM_BYTES_TPU_V6E = 32 * 1024 * 1024 * 1024  # 32 GB
HBM_BYTES_TPU_V7X = 96 * 1024 * 1024 * 1024  # 96 GB per logical TensorCore device (192 GB per dual-core chip)

TPU_HBM_BYTES_MAP = {
    "v5p": HBM_BYTES_TPU_V5P,
    "v6e": HBM_BYTES_TPU_V6E,
    "tpu7x": HBM_BYTES_TPU_V7X,
    "v7x": HBM_BYTES_TPU_V7X,
}

# TPU default 3D mesh topology mappings (v6e / single-core TPUs) by pod type or total chips
TPU_TOPOLOGY_MAP = {
    "v6e-32": "4,8,1",
    "v6e-8": "2,4,1",
    "v6e-4": "2,2,1",
    256: "16,16,1",
    128: "8,16,1",
    64: "8,8,1",
    32: "4,8,1",
    16: "4,4,1",
    8: "2,4,1",
    4: "2,2,1",
    2: "1,2,1",
    1: "1,1,1",
}

# TPU v5p 3D mesh topology mappings (X, Y, Z) by pod type or total chips (1 logical device per chip in Megacore mode).
# Each host is 2x2x1 (4 chips = 8 TensorCores); multi-host slices stack along Z then X/Y (2x2x2 = 8 chips,
# 2x2x4 = 16 chips, 2x4x4 = 32 chips, 4x4x4 = 64 chips, 4x4x8 = 128 chips, 4x8x8 = 256 chips).
TPU_V5P_TOPOLOGY_MAP = {
    "v5p-512": "4,8,8",
    "v5p-256": "4,4,8",
    "v5p-128": "4,4,4",
    "v5p-64": "2,4,4",
    "v5p-32": "2,2,4",
    "v5p-16": "2,2,2",
    "v5p-8": "2,2,1",
    256: "4,8,8",
    128: "4,4,8",
    64: "4,4,4",
    32: "2,4,4",
    16: "2,2,4",
    8: "2,2,2",
    4: "2,2,1",
    2: "1,2,1",
    1: "1,1,1",
}

# TPU 7x (Ironwood) 4D mesh topology mappings (X, Y, Z, CoresPerChip=2), keyed by device count.
# A host has 4 chips (2x2x1) and 8 devices; the multi-host entries follow the GKE slice shapes
# (2x2x2 = 2 hosts, 2x2x4 = 4, 2x4x4 = 8, 4x4x4 = 16).
TPU_V7X_TOPOLOGY_MAP = {
    128: "4,4,4,2",
    64: "2,4,4,2",
    32: "2,2,4,2",
    16: "2,2,2,2",
    8: "2,2,1,2",
    4: "1,2,1,2",
    2: "1,1,1,2",
    1: "1,1,1,1",
}


def get_tpu_topology_map() -> dict:
    """Returns the topology map for the configured TPU generation (4D for TPU 7x, 3D for v5p/v6e)."""
    tpu_type = os.environ.get("TPU_ACCELERATOR_TYPE", "v6e").lower()
    if any(k in tpu_type for k in ("tpu7x", "v7x")):
        return TPU_V7X_TOPOLOGY_MAP
    if "v5p" in tpu_type:
        return TPU_V5P_TOPOLOGY_MAP
    return TPU_TOPOLOGY_MAP


def get_tpu_chip_hbm_bytes() -> int:
    """Detects the TPU chip generation from Ray node labels or environment variables and returns its HBM capacity."""
    tpu_type = ""

    # Query Ray cluster node labels for TPU resource type
    try:
        if ray.is_initialized():
            tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
            if tpu_nodes:
                labels = tpu_nodes[0].get("Labels", {})
                tpu_type = (
                    labels.get("ray.io/accelerator-type")
                    or labels.get("ray.io/tpu-pod-type")
                    or labels.get("cloud.google.com/gke-tpu-accelerator")
                    or ""
                ).lower()
    except Exception as e:
        logger.warning(f"Unable to query Ray node labels for TPU chip type: {e}")

    # Fallback to environment variables
    if not tpu_type:
        tpu_type = (
            os.environ.get("TPU_ACCELERATOR_TYPE")
            or os.environ.get("ACCELERATOR_TYPE")
            or os.environ.get("TPU_TYPE")
            or "v6e"
        ).lower()

    for chip_gen, hbm_bytes in TPU_HBM_BYTES_MAP.items():
        if chip_gen in tpu_type:
            return hbm_bytes

    logger.warning(f"Unable to determine TPU chip HBM bytes for tpu_type='{tpu_type}'. Returning -1.")
    return -1


# Enforce static compilation graph for torch.compile on TPU
try:
    _orig_compile = torch.compile

    def patched_compile(*args, **kwargs):
        if get_platform().device_name == "tpu":
            kwargs["dynamic"] = False
        return _orig_compile(*args, **kwargs)

    torch.compile = patched_compile
except Exception as e:
    logger.warning(f"Failed to patch torch.compile for TPU: {e}")


class DummyTpuDeviceModule:
    """Fallback device module for CPU-only nodes and driver processes.

    Provides no-op implementations for torch.tpu APIs on processes where torch_tpu is not imported
    or no TPU devices are attached.
    """

    def is_available(self) -> bool:
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 0

    def synchronize(self) -> None:
        pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def get_device_name(self, device: Any = None) -> str:
        return os.environ.get("TPU_ACCELERATOR_TYPE", "v6e")


class TPUDeviceModuleProxy:
    """Proxy wrapper for torch.tpu to emulate PyTorch CUDA memory management APIs.

    Provides default fallback implementations for CUDA memory tracking methods
    (e.g., memory_reserved, memory_allocated, get_device_properties) that are called throughout verl's codebase
    but not natively provided by torch_tpu.
    """

    def __init__(self, original_module):
        self.__dict__["_original_module"] = original_module

    def __getattr__(self, name):
        if name == "set_device":
            return self.set_device

        if hasattr(self._original_module, name):
            return getattr(self._original_module, name)

        if name == "memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "reset_peak_memory_stats":
            return lambda *args, **kwargs: None
        elif name == "get_device_name":
            return lambda *args, **kwargs: os.environ.get("TPU_ACCELERATOR_TYPE", "v6e")
        elif name == "get_device_properties":

            class DummyDeviceProperties:
                def __init__(self, total_memory=32 * 1024 * 1024 * 1024):
                    self.total_memory = total_memory
                    self.name = "Google TPU"
                    self.major = 1
                    self.minor = 0

            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else 32 * 1024 * 1024 * 1024
            return lambda *args, **kwargs: DummyDeviceProperties(total_memory=total_mem)
        elif name == "mem_get_info":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else 32 * 1024 * 1024 * 1024
            return lambda *args, **kwargs: (total_mem, total_mem)

        raise AttributeError(f"'TPUDeviceModuleProxy' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._original_module, name, value)

    def is_available(self) -> bool:
        if hasattr(self._original_module, "is_available"):
            try:
                return self._original_module.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
                return False
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        if hasattr(self._original_module, "current_device"):
            try:
                return self._original_module.current_device()
            except Exception as e:
                logger.warning(f"torch.tpu.current_device() failed: {e}")
                return 0
        return 0

    def device_count(self) -> int:
        if hasattr(self._original_module, "device_count"):
            try:
                return self._original_module.device_count()
            except Exception as e:
                logger.warning(f"torch.tpu.device_count() failed: {e}")
                return 0
        return 0

    def synchronize(self, device_index: int | None = None) -> None:
        if hasattr(self._original_module, "synchronize"):
            try:
                self._original_module.synchronize()
            except Exception as e:
                logger.warning(f"torch.tpu.synchronize() failed: {e}")

    def empty_cache(self) -> None:
        if hasattr(self._original_module, "_clear_cache"):
            try:
                self._original_module._clear_cache()
            except Exception as e:
                logger.warning(f"Failed to clear TPU cache: {e}")


@PlatformRegistry.register(platform="tpu")
class PlatformTPU(PlatformCUDA):
    """Platform backend for Google TPUs (subclasses PlatformCUDA for API compatibility)."""

    def __init__(self):
        super().__init__()
        original_tpu = getattr(torch, "tpu", DummyTpuDeviceModule())
        self._device_module = TPUDeviceModuleProxy(original_tpu)

    @property
    def vendor_name(self) -> str:
        return "google"

    @property
    def device_name(self) -> str:
        return "tpu"

    @property
    def device_module(self):
        return self._device_module

    def current_device(self) -> int:
        return self.device_module.current_device()

    def device_count(self) -> int:
        return self.device_module.device_count()

    def set_device(self, device_index: int) -> None:
        self.device_module.set_device(device_index)

    def synchronize(self, device_index: int | None = None) -> None:
        self.device_module.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        self.device_module.manual_seed_all(seed)

    def is_available(self) -> bool:
        if hasattr(torch, "tpu"):
            try:
                return torch.tpu.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
        return False

    def is_platform_available(self, use_smi_check=False) -> bool:
        if os.environ.get("VERL_PLATFORM") == "tpu":
            return True
        if "TPU_NAME" in os.environ or "TPU_VISIBLE_DEVICES" in os.environ:
            return True
        return False

    def ray_resource_name(self) -> str:
        return "TPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        tpu_chips = int(num_gpus)
        return {"resources": {"TPU": tpu_chips}} if tpu_chips >= 1 else {}

    def communication_backend_name(self) -> str:
        return "tpu_dist"

    def ray_noset_envvars(self) -> list[str]:
        return super().ray_noset_envvars() + [
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        ]

    def get_tpu_env_vars(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        pgs: list,
        accelerator_type: Optional[str] = None,
    ) -> dict[str, str]:
        """Generates TPU-specific distributed environment variables for PJRT mesh initialization.

        ``accelerator_type`` is the slice resource of the worker group's resource pool (``tpu-group-<k>``);
        it tells a group that uses only part of a slice apart from one spanning the whole slice.
        """
        node_ip_map = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes() if node.get("Alive", False)}
        bundle_ips = []
        local_ip = ray.util.get_node_ip_address()
        clean_prefix = name_prefix.lower().split("_")[0] if name_prefix else ""
        matching_pgs = []

        # 1. Primary filter: Select placement group containing current worker's node IP
        for p in pgs:
            specs = ray._private.state.state.placement_group_table(p.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            pg_ips = [node_ip_map[node_id] for b_idx, node_id in sorted(bundles_map.items()) if node_id in node_ip_map]
            if local_ip in pg_ips:
                matching_pgs.append(p)

        # 2. Secondary fallback: Filter by clean_prefix if placement group names are explicitly set
        if not matching_pgs and clean_prefix:
            for p in pgs:
                p_name = ray._private.state.state.placement_group_table(p.id).get("name", "").lower()
                if clean_prefix in p_name:
                    matching_pgs.append(p)

        matching_bundles = sum(
            len(ray._private.state.state.placement_group_table(p.id).get("bundles_to_node_id", {}))
            for p in matching_pgs
        )
        target_pgs = matching_pgs if (matching_pgs and matching_bundles == world_size) else pgs

        for pg in target_pgs:
            specs = ray._private.state.state.placement_group_table(pg.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            for b_idx in sorted(bundles_map.keys()):
                node_id = bundles_map[b_idx]
                if node_id in node_ip_map:
                    bundle_ips.append(node_ip_map[node_id])

        is_rollout = "rollout" in name_prefix.lower()
        base_port = ROLLOUT_BASE_PORT if is_rollout else TRAINER_BASE_PORT

        sb_addresses = [f"{ip}:{base_port + (b_idx % local_world_size)}" for b_idx, ip in enumerate(bundle_ips)]

        # Extract unique worker hostnames preserving rank order
        unique_hostnames = list(dict.fromkeys(bundle_ips))

        env_vars = {
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "CLOUD_TPU_TASK_ID": str(rank),
            "TPU_WORKER_HOSTNAMES": ",".join(unique_hostnames),
            "TPU_VISIBLE_CHIPS": str(local_rank),
            "TPU_VISIBLE_DEVICES": str(local_rank),
        }

        # Apply TPU topology and host bounds based on TPU pod type or world size. The pod-type label describes
        # the whole slice; a worker group that uses only part of it (e.g. one TP=4 replica per host of a 2-host
        # v6e-8 slice) must take the shape of its own world size instead.
        tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
        tpu_type = tpu_nodes[0].get("Labels", {}).get("ray.io/tpu-pod-type", "") if tpu_nodes else ""

        topo_map = get_tpu_topology_map()
        slice_chips = slice_chips_for(accelerator_type)
        if slice_chips is not None and world_size != slice_chips:
            topo = topo_map.get(world_size) or topo_map.get(tpu_type)
        else:
            topo = topo_map.get(tpu_type) or topo_map.get(world_size)
        if topo is None:
            # Falling back to the single-device topology makes a multi-host trainer fail in ways that do not
            # point here (the mesh forms, then collectives stall or halt), so say what happened.
            logger.warning(
                "No TPU topology entry for pod type %r or %d devices; falling back to %r. "
                "Add the slice shape to the topology map.",
                tpu_type,
                world_size,
                topo_map[1],
            )
            topo = topo_map[1]
        chips_bounds = topo_map[1]

        env_vars.update(
            {
                "TORCH_TPU_TOPOLOGY": topo,
                "TPU_HOST_BOUNDS": topo,
                "TPU_PROCESS_BOUNDS": topo,
                "TPU_CHIPS_PER_HOST_BOUNDS": chips_bounds,
                "TPU_CHIPS_PER_PROCESS_BOUNDS": chips_bounds,
                "CHIPS_PER_HOST": str(max(local_world_size, 4)),
            }
        )

        if is_rollout:
            env_vars.update(
                {
                    "SKIP_JAX_PRECOMPILE": "1",
                    "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
                }
            )
            if world_size > 1:
                env_vars["TPU_MULTIHOST_BACKEND"] = "ray"

        env_vars["LIBTPU_INIT_ARGS"] = _ensure_slicebuilder_insecure_grpc(os.environ.get("LIBTPU_INIT_ARGS", ""))

        return env_vars

    def auto_assign_accelerator_type(self, name_prefix: str, accelerator_type: Optional[str]) -> Optional[str]:
        """Assign a TPU slice (``tpu-group-<k>`` or ``node:<ip>``) to a resource pool on multi-slice clusters."""
        if accelerator_type is not None:
            return accelerator_type
        return self.plan_resource_pool_placement(name_prefix)[0]

    def plan_resource_pool_placement(
        self, name_prefix: str, process_on_nodes: Optional[list[int]] = None
    ) -> tuple[Optional[str], Optional[list[str]]]:
        """Slice resource and per-placement-group host pins for a resource pool.

        Trainer pools get the trainer slice; ``rollout_pool_<idx>`` pools get the slice and the hosts
        of rollout replica ``idx`` from the slice plan (see ``platform_tpu_slices``), so that several
        replicas on one multi-host slice occupy distinct hosts instead of racing for the same one.
        """
        try:
            return plan_resource_pool_placement(name_prefix, process_on_nodes)
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"TPU slice assignment failed for {name_prefix}: {e}")
            return None, None

    def configure_placement_group_bundle(
        self, bundle: dict, use_gpu: bool, device_name: str, name_prefix: str, accelerator_type: Optional[str] = None
    ) -> None:
        """Configure placement group bundle resources to prevent vLLM resource lockups on GKE TPU."""
        is_rollout_pool = any(k in name_prefix.lower() for k in ["rollout", "reward", "teacher"])
        if use_gpu and not is_rollout_pool:
            bundle[device_name] = 1
        if accelerator_type is not None:
            bundle[accelerator_type] = 1e-4

    def pin_placement_group_scheme(self, pg_scheme: list[list[dict]], hosts: Optional[list[str]]) -> None:
        """Pin each placement group of ``pg_scheme`` to the host planned for it.

        Rollout bundles reserve no ``TPU`` (vLLM allocates the chips itself), so without a per-host
        constraint Ray may pack two replicas of one slice onto the same host. ``node:<ip>`` is the
        resource Ray advertises on exactly that node.
        """
        if not hosts:
            return
        if len(hosts) != len(pg_scheme):
            logger.warning(
                f"TPU host pins {hosts} do not match the {len(pg_scheme)} placement group(s); leaving them unpinned"
            )
            return
        for bundles, ip in zip(pg_scheme, hosts, strict=True):
            for bundle in bundles:
                bundle[f"node:{ip}"] = 1e-4

    def get_worker_env_vars(
        self,
        resource_pool,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        device_name: str,
    ) -> dict[str, str]:
        """Return platform-specific TPU environment variables for worker nodes."""
        env_vars = {
            "TPU_ACCELERATOR_TYPE": os.environ.get("TPU_ACCELERATOR_TYPE", "v6e"),
        }
        if "VERL_PLATFORM" in os.environ:
            env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]
        for var in self.ray_noset_envvars():
            env_vars[var] = "1"
        pgs = resource_pool.get_placement_groups(device_name=device_name)
        tpu_env = self.get_tpu_env_vars(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            name_prefix=name_prefix,
            pgs=pgs,
            accelerator_type=getattr(resource_pool, "accelerator_type", None),
        )
        env_vars.update(tpu_env)
        return env_vars

    def sanitize_metrics(self, metrics: Any) -> Any:
        """Convert any TPU tensor in metrics to a standard Python scalar before Ray RPC transfer."""
        return convert_tensors_to_scalars(metrics)

    def rollout_env_vars(self) -> dict[str, str]:
        """Return env vars to inject into the rollout engine.

        Overrides PlatformCUDA, whose NCCL_CUMEM_ENABLE is meaningless here.

        The rollout server actor is created with an explicit runtime_env, which
        replaces inheritance from the job, so compiler flags set on the driver
        have to be named to survive. The caller appends VERL_TPU_EXTRA_<VAR> and
        falls back to the value forwarded off the worker.
        """
        env_vars = {var: os.environ[var] for var in ("XLA_FLAGS", "LIBTPU_INIT_ARGS") if os.environ.get(var)}
        env_vars["LIBTPU_INIT_ARGS"] = _ensure_slicebuilder_insecure_grpc(env_vars.get("LIBTPU_INIT_ARGS", ""))
        return env_vars

    def get_ray_init_kwargs(self) -> dict[str, Any]:
        """Return Ray initialization arguments with runtime_env configured for GKE TPU workers."""
        env_vars = {
            "VERL_PLATFORM": "tpu",
            "TPU_ACCELERATOR_TYPE": os.environ.get("TPU_ACCELERATOR_TYPE", "v6e"),
            "LIBTPU_INIT_ARGS": _ensure_slicebuilder_insecure_grpc(os.environ.get("LIBTPU_INIT_ARGS", "")),
        }
        for key in (
            "WANDB_API_KEY",
            "WANDB_BASE_URL",
            "WANDB_ENTITY",
            "WANDB_PROJECT",
            "WANDB_MODE",
            "VERL_FILE_LOGGER_ROOT",
            "VERL_FILE_LOGGER_PATH",
            "TENSORBOARD_DIR",
        ):
            if os.environ.get(key):
                env_vars[key] = os.environ[key]
        return {
            "runtime_env": {
                "worker_process_setup_hook": patch_ray_worker,
                "env_vars": env_vars,
            }
        }


def _ensure_slicebuilder_insecure_grpc(raw_args: str) -> str:
    """Ensure LIBTPU_INIT_ARGS enables insecure SliceBuilder gRPC to avoid GKE ALTS handshake hangs."""
    tokens = raw_args.split()
    if not any(t.startswith("--slicebuilder_use_insecure_grpc") for t in tokens):
        tokens.append("--slicebuilder_use_insecure_grpc=true")
    if not any(t.startswith("--undefok=") and "slicebuilder_use_insecure_grpc" in t for t in tokens):
        tokens.append("--undefok=slicebuilder_use_insecure_grpc")
    return " ".join(tokens)

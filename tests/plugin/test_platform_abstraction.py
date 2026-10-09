# Copyright (c) 2026 BAAI. All rights reserved.
"""Unit tests for the platform abstraction layer."""

import os
from contextlib import contextmanager
from unittest import mock

import pytest

from verl.plugin.platform import get_platform, set_platform
from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import (
    PlatformRegistry,
    _create_platform,
    _detect_platform_name,
)
from verl.plugin.platform.platform_tpu import PlatformTPU


def _make_mock_platform(name="mock_xpu"):
    """Return a minimal concrete PlatformBase subclass for testing."""

    class _Mock(PlatformBase):
        @property
        def device_name(self):
            return name

        @property
        def vendor_name(self):
            return f"mock_{name}"

        @property
        def device_module(self):
            import torch

            return torch

        def is_available(self):
            return True

        def is_platform_available(self, use_smi_check=False):
            return True

        def current_device(self):
            return 0

        def device_count(self):
            return 1

        def set_device(self, device_index):
            pass

        def synchronize(self, device_index=None):
            pass

        def manual_seed(self, seed):
            pass

        def manual_seed_all(self, seed):
            pass

        def set_allocator_settings(self, settings):
            pass

        def empty_cache(self):
            pass

        def get_device_capability(self, device_index=0):
            return (None, None)

        def communication_backend_name(self):
            return "mock_ccl"

        def visible_devices_envvar(self):
            return "MOCK_VISIBLE_DEVICES"

        def ray_resource_name(self):
            return "MOCK"

        def ray_noset_envvars(self):
            return ["RAY_EXPERIMENTAL_NOSET_MOCK_VISIBLE_DEVICES"]

        def is_ipc_supported(self):
            return False

        @contextmanager
        def nvtx_range(self, msg):
            yield

        def profiler_start(self):
            pass

        def profiler_stop(self):
            pass

        def cudart(self):
            return None

    _Mock.__name__ = f"Mock_{name}"
    return _Mock


class TestPlatformDetection:
    """Test platform auto-detection logic."""

    def setup_method(self):
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_env_override_nvidia(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "nvidia"}):
            assert _detect_platform_name() == "nvidia"

    def test_env_override_huawei(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "huawei"}):
            assert _detect_platform_name() == "huawei"

    def test_env_override_amd(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "amd"}):
            assert _detect_platform_name() == "amd"

    def test_invalid_value_passes_through(self):
        # When an explicit platform name is set, _detect_platform_name returns
        # it as-is (validation happens later in _create_platform).
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "invalid"}):
            assert _detect_platform_name() == "invalid"

    def test_case_insensitive(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "NVIDIA"}):
            assert _detect_platform_name() == "nvidia"

    def test_empty_triggers_auto_detection(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": ""}):
            assert _detect_platform_name() in ("nvidia", "huawei", "amd")


class TestPlatformCreation:
    """Test platform creation."""

    def test_cuda_warns_if_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("verl.plugin.platform.platform_manager.logger") as mock_logger:
                platform = _create_platform("nvidia")
                mock_logger.warning.assert_called_once()
                assert platform is not None

    def test_invalid_platform_raises(self):
        with pytest.raises(ValueError):
            _create_platform("invalid_platform")


class TestPlatformSingleton:
    """Test singleton and external injection."""

    def setup_method(self):
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_get_platform_returns_singleton(self):
        p1 = get_platform()
        p2 = get_platform()
        assert p1 is p2

    def test_set_platform_external_injection(self):
        """Simulates an entry_points plugin calling set_platform()."""
        custom = _make_mock_platform("mock_xpu")()
        set_platform(custom)
        assert get_platform() is custom
        assert get_platform().device_name == "mock_xpu"


class TestPlatformRegistry:
    """Test PlatformRegistry dynamic registration."""

    def setup_method(self):
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_builtin_platforms_registered(self):
        names = PlatformRegistry.registered_names()
        assert "nvidia" in names
        assert "huawei" in names
        assert "amd" in names

    def test_register_custom_platform(self):
        """External plugin registers a new platform via @PlatformRegistry.register()."""
        MockCls = _make_mock_platform("mock_test")
        PlatformRegistry.register(platform="mock_test")(MockCls)

        assert "mock_test" in PlatformRegistry.registered_names()
        assert PlatformRegistry.get("mock_test") is MockCls
        assert _create_platform("mock_test").device_name == "mock_test"

        del PlatformRegistry._platforms["mock_test"]

    def test_register_override(self):
        """Re-registering the same name overrides the previous class (last writer wins)."""
        original = PlatformRegistry.get("nvidia")
        FakeCls = _make_mock_platform("nvidia")
        PlatformRegistry.register(platform="nvidia")(FakeCls)
        assert PlatformRegistry.get("nvidia") is FakeCls
        PlatformRegistry._platforms["nvidia"] = original

    def test_unregistered_platform_raises(self):
        with pytest.raises(ValueError):
            _create_platform("nonexistent_platform")


class TestPlatformTPUSliceAssignment:
    """Test PlatformTPU.auto_assign_accelerator_type across single-host, multi-host, and hybrid TPU clusters."""

    def test_explicit_accelerator_type_preserved(self):
        ptpu = PlatformTPU()
        assert ptpu.auto_assign_accelerator_type("rollout_pool_0", "tpu-group-9", [8]) == "tpu-group-9"

    def test_homogeneous_multihost_slices(self):
        fake_nodes = [
            {"Alive": True, "NodeManagerAddress": "10.0.0.1", "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.0.0.2", "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.0.1.1", "Resources": {"TPU": 4.0, "tpu-group-1": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.0.1.2", "Resources": {"TPU": 4.0, "tpu-group-1": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.0.2.1", "Resources": {"TPU": 4.0, "tpu-group-2": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.0.2.2", "Resources": {"TPU": 4.0, "tpu-group-2": 1.0}},
        ]
        ptpu = PlatformTPU()
        with (
            mock.patch("ray.is_initialized", return_value=True),
            mock.patch("ray.nodes", return_value=fake_nodes),
        ):
            assert ptpu.auto_assign_accelerator_type("global_pool", None, [4, 4]) == "tpu-group-0"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_0", None, [4, 4]) == "tpu-group-1"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_1", None, [4, 4]) == "tpu-group-2"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_2", None, [4, 4]) == "tpu-group-1"

    def test_hybrid_multihost_trainer_and_singlehost_8t_rollout_replicas(self):
        fake_nodes = [
            {"Alive": True, "NodeManagerAddress": "10.11.0.38", "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.39", "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.40", "Resources": {"TPU": 4.0, "tpu-group-1": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.41", "Resources": {"TPU": 4.0, "tpu-group-1": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.50", "Resources": {"TPU": 8.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.55", "Resources": {"TPU": 8.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.61", "Resources": {"TPU": 8.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.62", "Resources": {"TPU": 8.0}},
        ]
        ptpu = PlatformTPU()
        with (
            mock.patch("ray.is_initialized", return_value=True),
            mock.patch("ray.nodes", return_value=fake_nodes),
        ):
            # Trainer requests [4, 4] -> matched to 2-host tpu-group-0
            assert ptpu.auto_assign_accelerator_type("global_pool", None, [4, 4]) == "tpu-group-0"
            # Rollout replicas request [8] -> matched to single-host 8-chip nodes
            assert ptpu.auto_assign_accelerator_type("rollout_pool_0", None, [8]) == "node:10.11.0.50"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_1", None, [8]) == "node:10.11.0.55"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_2", None, [8]) == "node:10.11.0.61"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_3", None, [8]) == "node:10.11.0.62"

    def test_kuberay_tpu_8t_group_resources(self):
        fake_nodes = [
            {"Alive": True, "NodeManagerAddress": "10.11.0.38", "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.39", "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.40", "Resources": {"TPU": 4.0, "tpu-group-1": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.41", "Resources": {"TPU": 4.0, "tpu-group-1": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.50", "Resources": {"TPU": 8.0, "tpu-8t-group-0": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.55", "Resources": {"TPU": 8.0, "tpu-8t-group-1": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.61", "Resources": {"TPU": 8.0, "tpu-8t-group-2": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.11.0.62", "Resources": {"TPU": 8.0, "tpu-8t-group-3": 1.0}},
        ]
        ptpu = PlatformTPU()
        with (
            mock.patch("ray.is_initialized", return_value=True),
            mock.patch("ray.nodes", return_value=fake_nodes),
        ):
            assert ptpu.auto_assign_accelerator_type("global_pool", None, [4, 4]) == "tpu-group-0"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_0", None, [8]) == "tpu-8t-group-0"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_1", None, [8]) == "tpu-8t-group-1"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_2", None, [8]) == "tpu-8t-group-2"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_3", None, [8]) == "tpu-8t-group-3"

    def test_heartbeat_interval_forwarded_in_env_vars(self):
        ptpu = PlatformTPU()
        with mock.patch.dict(
            os.environ,
            {"TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL": "300s", "LIBTPU_INIT_ARGS": "--foo=bar"},
            clear=False,
        ):
            rollout_vars = ptpu.rollout_env_vars()
            assert rollout_vars["TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL"] == "300s"
            assert "--foo=bar" in rollout_vars["LIBTPU_INIT_ARGS"]
            assert "--slicebuilder_use_insecure_grpc=true" in rollout_vars["LIBTPU_INIT_ARGS"]
            ray_kwargs = ptpu.get_ray_init_kwargs()
            assert ray_kwargs["runtime_env"]["env_vars"]["TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL"] == "300s"

    def test_exclude_node_ips_in_auto_assign(self):
        fake_nodes = [
            {"Alive": True, "NodeManagerAddress": "10.204.17.9", "Resources": {"TPU": 8.0, "tpu-8t-group-0": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.204.15.12", "Resources": {"TPU": 8.0, "tpu-8t-group-1": 1.0}},
            {"Alive": True, "NodeManagerAddress": "10.204.8.5", "Resources": {"TPU": 8.0, "tpu-8t-group-2": 1.0}},
        ]
        ptpu = PlatformTPU()
        with (
            mock.patch("ray.is_initialized", return_value=True),
            mock.patch("ray.nodes", return_value=fake_nodes),
            mock.patch.dict(os.environ, {"VERL_TPU_EXCLUDE_NODE_IPS": "10.204.17.9"}, clear=False),
        ):
            assert ptpu.auto_assign_accelerator_type("rollout_pool_0", None, [8]) == "tpu-8t-group-1"
            assert ptpu.auto_assign_accelerator_type("rollout_pool_1", None, [8]) == "tpu-8t-group-2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

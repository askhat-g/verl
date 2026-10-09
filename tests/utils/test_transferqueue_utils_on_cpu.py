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

import asyncio
import gc
import os
import sys
import types

import pytest

from verl.utils import transferqueue_utils as tqu


async def _noop():
    return None


def _open_fd_count() -> int:
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("requires Linux /proc file descriptor accounting")
    return len(os.listdir("/proc/self/fd"))


def test_temp_event_loop_releases_file_descriptors():
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    tqu._shutdown_async_bridge_runtime()
    try:
        tqu._run_async_in_temp_loop(_noop)
        before = _open_fd_count()
        for _ in range(32):
            tqu._run_async_in_temp_loop(_noop)
        leaked = _open_fd_count() - before
    finally:
        tqu._shutdown_async_bridge_runtime()
        if gc_was_enabled:
            gc.enable()
        gc.collect()

    assert leaked == 0


def test_async_bridge_loop_reused_between_calls():
    tqu._shutdown_async_bridge_runtime()
    try:
        tqu._run_async_in_temp_loop(_noop)
        first_thread = tqu._ASYNC_BRIDGE_THREAD
        first_loop = tqu._ASYNC_BRIDGE_LOOP

        assert first_thread is not None
        assert first_loop is not None
        assert first_thread.is_alive()
        assert not first_loop.is_closed()

        tqu._run_async_in_temp_loop(_noop)
        assert tqu._ASYNC_BRIDGE_THREAD is first_thread
        assert tqu._ASYNC_BRIDGE_LOOP is first_loop
    finally:
        tqu._shutdown_async_bridge_runtime()


def test_patch_transfer_queue_clear_order_samples_and_partition(monkeypatch):
    """Verify patched clear methods clear storage units BEFORE releasing controller global_indexes."""
    events: list[str] = []

    class FakeStorageManager:
        async def clear_data(self, metadata):
            events.append(f"storage.clear_data:{metadata.size}")

    class FakeAsyncTransferQueueClient:
        def __init__(self):
            self.client_id = "test_client"
            self.storage_manager = FakeStorageManager()
            self._controller = object()
            self.rebound = 0

        async def _clear_meta_in_controller(self, metadata):
            events.append(f"controller.clear_meta:{metadata.size}")

        async def _get_partition_meta(self, partition_id: str):
            events.append(f"controller.get_partition_meta:{partition_id}")
            if partition_id == "empty_part":
                return None
            return types.SimpleNamespace(size=4)

        async def _clear_partition_in_controller(self, partition_id: str):
            events.append(f"controller.clear_partition:{partition_id}")

        def _bind_sync_methods(self):
            self.rebound += 1

    existing_client = FakeAsyncTransferQueueClient()

    fake_client_mod = types.ModuleType("transfer_queue.client")
    fake_client_mod.AsyncTransferQueueClient = FakeAsyncTransferQueueClient
    fake_interface_mod = types.ModuleType("transfer_queue.interface")
    fake_interface_mod._TQ_CLIENT = existing_client
    fake_tq_pkg = types.ModuleType("transfer_queue")
    fake_tq_pkg.client = fake_client_mod
    fake_tq_pkg.interface = fake_interface_mod

    monkeypatch.setitem(sys.modules, "transfer_queue", fake_tq_pkg)
    monkeypatch.setitem(sys.modules, "transfer_queue.client", fake_client_mod)
    monkeypatch.setitem(sys.modules, "transfer_queue.interface", fake_interface_mod)

    assert tqu.patch_transfer_queue_clear_order() is True
    assert getattr(FakeAsyncTransferQueueClient, "_verl_clear_order_patched", False) is True
    assert existing_client.rebound == 1

    # 1. Non-empty clear_samples must clear storage unit BEFORE controller metadata
    events.clear()
    asyncio.run(existing_client.async_clear_samples(types.SimpleNamespace(size=16)))
    assert events == ["storage.clear_data:16", "controller.clear_meta:16"]

    # 2. Empty metadata (size == 0) is a safe no-op
    events.clear()
    asyncio.run(existing_client.async_clear_samples(types.SimpleNamespace(size=0)))
    assert events == []

    # 3. clear_partition must clear storage unit BEFORE controller partition metadata
    events.clear()
    asyncio.run(existing_client.async_clear_partition("train"))
    assert events == [
        "controller.get_partition_meta:train",
        "storage.clear_data:4",
        "controller.clear_partition:train",
    ]

    # 4. Non-existent partition is a safe no-op after _get_partition_meta
    events.clear()
    asyncio.run(existing_client.async_clear_partition("empty_part"))
    assert events == ["controller.get_partition_meta:empty_part"]


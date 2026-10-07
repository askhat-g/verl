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
"""release_sync_buffers() must unbind the send tensors without tearing down the WeightSynchronizer."""

from verl.checkpoint_engine.raiden_checkpoint_engine import RaidenCheckpointEngine


class _FakeWeightSynchronizer:
    def __init__(self):
        self.unbind_calls = 0
        self.closed = False

    def unbind_weights(self):
        self.unbind_calls += 1

    def close(self):
        self.closed = True


def _engine(release: bool) -> tuple[RaidenCheckpointEngine, _FakeWeightSynchronizer]:
    # __init__ looks up the Ray-backed TPUWeightRegistry; only the fields release_sync_buffers reads are needed.
    engine = RaidenCheckpointEngine.__new__(RaidenCheckpointEngine)
    ws = _FakeWeightSynchronizer()
    engine.release_buffers_after_sync = release
    engine._trainer_raiden_ws = ws
    engine._registered_signature = [("w", (2, 2), "bf16", None)]
    engine._bound_tensors = [object()]
    return engine, ws


def test_release_unbinds_but_keeps_synchronizer_and_signature():
    engine, ws = _engine(release=True)

    assert engine.release_sync_buffers() == {}

    assert ws.unbind_calls == 1
    assert not ws.closed
    assert engine._bound_tensors is None
    # Kept so the next send_weights takes the bind_weights reuse path instead of re-creating the synchronizer.
    assert engine._trainer_raiden_ws is ws
    assert engine._registered_signature == [("w", (2, 2), "bf16", None)]


def test_release_is_noop_when_disabled():
    engine, ws = _engine(release=False)

    assert engine.release_sync_buffers() == {}

    assert ws.unbind_calls == 0
    assert engine._bound_tensors is not None
    assert engine._trainer_raiden_ws is ws


def test_release_is_noop_before_first_sync():
    engine, _ = _engine(release=True)
    engine._trainer_raiden_ws = None

    assert engine.release_sync_buffers() == {}

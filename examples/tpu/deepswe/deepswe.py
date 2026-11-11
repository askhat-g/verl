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
"""DeepSWE recipe hooks for 2-slot lookahead Kubernetes sandbox pre-warming and pinned evaluation pools.

This module lives inside the DeepSWE recipe folder (``examples/tpu/deepswe/deepswe.py``, to be migrated to
``verl-recipe/deepswe/``) so core ``verl`` trainer files remain untouched.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from tensordict import TensorDict

from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY, PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu

try:
    from uni_agent.sandbox.k8s_agent_sandbox import (
        PrewarmBatchIterator,
        close_lookahead_warmer,
        ensure_lookahead_batch_warm,
        extract_sandbox_images,
        finish_lookahead_batch_async,
        pin_eval_sandbox_images,
        prefetch_lookahead_batch_async,
        wait_eval_sandbox_images_ready,
    )
except ImportError:
    PrewarmBatchIterator = None  # type: ignore[assignment]
    close_lookahead_warmer = None  # type: ignore[assignment]
    ensure_lookahead_batch_warm = None  # type: ignore[assignment]
    extract_sandbox_images = None  # type: ignore[assignment]
    finish_lookahead_batch_async = None  # type: ignore[assignment]
    pin_eval_sandbox_images = None  # type: ignore[assignment]
    prefetch_lookahead_batch_async = None  # type: ignore[assignment]
    wait_eval_sandbox_images_ready = None  # type: ignore[assignment]

try:
    from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
except ImportError:
    AgentFrameworkRolloutAdapter = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class _DeepSWESandboxLookaheadMixin:
    """Mixin adding 2-slot ``PrewarmBatchIterator`` lookahead and pinned eval pools to ``PPOTrainer``."""

    _prewarm_batch_iterator: PrewarmBatchIterator | None = None
    _sandbox_lookahead_initialized: bool = False

    def _init_sandbox_lookahead(self: PPOTrainer) -> None:
        if getattr(self, "_sandbox_lookahead_initialized", False):
            return
        self._sandbox_lookahead_initialized = True
        self._prewarm_batch_iterator = None

        if (
            PrewarmBatchIterator is None
            or extract_sandbox_images is None
            or ensure_lookahead_batch_warm is None
            or prefetch_lookahead_batch_async is None
            or finish_lookahead_batch_async is None
        ):
            return
        if os.getenv("VERL_SANDBOX_LOOKAHEAD", "1").strip().lower() in ("0", "false", "no"):
            return

        eval_images = extract_sandbox_images(self.val_dataset)
        if eval_images and pin_eval_sandbox_images is not None:
            val_n = int(
                os.getenv(
                    "SANDBOX_EVAL_REPLICAS",
                    str(self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1) or 1),
                )
            )
            val_before_train = bool(self.config.trainer.get("val_before_train", True))
            logger.info(
                "Pinning %d unique evaluation sandbox images (replicas=%d, wait=%s)",
                len(set(eval_images)),
                val_n,
                val_before_train,
            )
            pin_eval_sandbox_images(
                eval_images,
                replicas_override=val_n,
                wait=val_before_train,
            )

        rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 8) or 8)
        self._prewarm_batch_iterator = PrewarmBatchIterator(
            fetch_batch_fn=self._raw_next_train_batch,
            extract_images_fn=extract_sandbox_images,
            ensure_warm_fn=ensure_lookahead_batch_warm,
            prefetch_async_fn=prefetch_lookahead_batch_async,
            finish_batch_fn=finish_lookahead_batch_async,
            replicas_override=rollout_n,
        )

    def _prime_sandbox_lookahead(self: PPOTrainer) -> None:
        self._init_sandbox_lookahead()
        if self._prewarm_batch_iterator is not None:
            self._prewarm_batch_iterator.prime()

    def _wait_eval_sandboxes_ready(self: PPOTrainer) -> None:
        self._init_sandbox_lookahead()
        if self._prewarm_batch_iterator is not None and wait_eval_sandbox_images_ready is not None:
            wait_eval_sandbox_images_ready()

    def _finish_sampled_sandbox_batch(self: PPOTrainer) -> None:
        if (
            self._prewarm_batch_iterator is not None
            and self.local_trigger_step == self.parameter_sync_step - 1
        ):
            self._prewarm_batch_iterator.finish_oldest_inflight_batch(wait=False)

    def _close_sandbox_lookahead(self: PPOTrainer) -> None:
        if close_lookahead_warmer is not None:
            close_lookahead_warmer()

    def _raw_next_train_batch(self: PPOTrainer, num_prompts: int | None = None) -> TensorDict:
        return super()._next_train_batch(num_prompts)  # type: ignore[misc]

    def _next_train_batch(self: PPOTrainer, num_prompts: int | None = None) -> TensorDict:
        train_batch_size = self.config.data.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        if getattr(self, "_prewarm_batch_iterator", None) is not None and num_prompts == train_batch_size:
            batch = self._prewarm_batch_iterator.next_batch()
        else:
            batch = self._raw_next_train_batch(num_prompts)
            if (
                getattr(self, "_prewarm_batch_iterator", None) is not None
                and extract_sandbox_images is not None
                and ensure_lookahead_batch_warm is not None
            ):
                imgs = extract_sandbox_images(batch)
                if imgs:
                    rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 8) or 8)
                    ensure_lookahead_batch_warm(
                        f"partial_{self.global_steps}_{uuid.uuid4().hex[:6]}",
                        imgs,
                        replicas_override=rollout_n,
                        wait=True,
                    )
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch


class DeepSWESeparateAsyncPPOTrainer(_DeepSWESandboxLookaheadMixin, PPOTrainerSeparateAsync):
    """Separate-async PPO trainer with 2-slot K8s sandbox lookahead pre-warming for DeepSWE."""

    def _should_refill_after_sample(self) -> bool:
        return not self.hybrid_rollout_config.enable_switch and os.getenv(
            "VERL_REFILL_AFTER_SAMPLE", "0"
        ).strip().lower() in ("1", "true", "yes")

    def on_train_begin(self) -> None:
        self._prime_sandbox_lookahead()
        super().on_train_begin()

    def on_train_end(self) -> None:
        try:
            super().on_train_end()
        finally:
            self._close_sandbox_lookahead()

    def on_validate_begin(self) -> None:
        self._wait_eval_sandboxes_ready()
        super().on_validate_begin()

    def on_sample_end(self) -> None:
        self._finish_sampled_sandbox_batch()
        super().on_sample_end()
        if (
            self._should_refill_after_sample()
            and self.local_trigger_step == self.parameter_sync_step - 1
            and self.global_steps < self.total_training_steps
        ):
            self._add_batch_to_generate()

    def prepare_step(self) -> dict:
        if self._should_refill_after_sample():
            return self._wait_for_sampleable_and_switch()
        return super().prepare_step()


class DeepSWESyncPPOTrainer(_DeepSWESandboxLookaheadMixin, PPOTrainerSync):
    """Synchronous PPO trainer with 2-slot K8s sandbox lookahead pre-warming for DeepSWE."""

    def on_train_begin(self) -> None:
        self._prime_sandbox_lookahead()
        super().on_train_begin()

    def on_train_end(self) -> None:
        try:
            super().on_train_end()
        finally:
            self._close_sandbox_lookahead()

    def on_validate_begin(self) -> None:
        self._wait_eval_sandboxes_ready()
        super().on_validate_begin()

    def on_sample_end(self) -> None:
        self._finish_sampled_sandbox_batch()
        super().on_sample_end()


class DeepSWERolloutAdapter(AgentFrameworkRolloutAdapter):
    """Rollout adapter entry point that also ensures DeepSWE trainer hooks are registered."""

    @classmethod
    def create(cls, **kwargs: Any) -> Any:
        register_deepswe_trainers()
        return super().create(**kwargs)


def register_deepswe_trainers() -> None:
    """Register DeepSWE lookahead trainers in ``TRAINER_REGISTRY``."""
    TRAINER_REGISTRY["separate_async"] = DeepSWESeparateAsyncPPOTrainer
    TRAINER_REGISTRY["sync"] = DeepSWESyncPPOTrainer


register_deepswe_trainers()

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RelativeActionsProcessorStep
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine, PolicyQuery

logger = logging.getLogger(__name__)


# Relative-action support: a predicted chunk of offsets is anchored to the robot
# state at prediction time, but the sync engine reruns the pre/post pipeline every
# tick, so ``RelativeActionsProcessorStep`` would re-anchor cached actions to the
# current (moved) state and drift through the chunk. We pin the anchor per chunk:
# ``PreTrainedPolicy.queued_action_count()`` reports whether this tick will serve an
# already-computed action (hold the anchor) or force a fresh prediction (let it
# advance). ``select_action`` stays on the hot path, so per-tick side effects (e.g.
# LingBot-VA keyframe feedback) are preserved.


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a CPU action tensor
    in the policy's own dimension order.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        super().__init__(task=task)
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type

        # Find an enabled RelativeActionsProcessorStep to pin its anchor per chunk
        # (see module comment), mirroring the RTC engine.
        self._relative_step = next(
            (
                s
                for s in getattr(preprocessor, "steps", ())
                if isinstance(s, RelativeActionsProcessorStep) and s.enabled
            ),
            None,
        )
        if self._relative_step is not None:
            # ``action_names`` is optional on the step; fill it lazily from the
            # policy/dataset so the relative<->absolute mask is built correctly. This is
            # a deliberate engine->step side effect (the step is configured by its consumer).
            if self._relative_step.action_names is None:
                cfg_names = getattr(policy.config, "action_feature_names", None)
                self._relative_step.action_names = list(cfg_names) if cfg_names else list(ordered_action_keys)
            logger.info("Relative actions enabled: chunk anchor pinned per predicted chunk")

        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        # The policy was just reset, so a pending task change has nothing stale to flush.
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if obs_frame is None:
            return None
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        task, task_changed = self._take_task()
        with torch.inference_mode(), autocast_ctx:
            if task_changed:
                # Chunking policies queue actions computed under the previous instruction,
                # so drop them and let the new one take effect on this tick.  Narrower
                # than ``policy.reset``: observation history and other episode state stay.
                logger.info("Task changed to '%s' — dropping precomputed actions", task)
                self._policy.drop_queued_actions()
            # A non-empty queue means this tick will serve an already-computed action, so
            # the anchor must hold; snapshot it before the preprocessor overwrites it below.
            # ``clone`` so the snapshot survives even if the cached tensor is ever mutated
            # in place (today it is only rebound, but the copy is cheap for a state vector).
            will_drain_cached = self._relative_step is not None and self._policy.queued_action_count() > 0
            anchor_before = None
            if will_drain_cached:
                cached = self._relative_step.get_cached_state()
                anchor_before = cached.clone() if cached is not None else None
            observation = prepare_observation_for_inference(observation, self._device, task, self._robot_type)
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            if will_drain_cached:
                self._relative_step.set_cached_state(anchor_before)
            action = self._postprocessor(action)
        action_tensor = action.squeeze(0).cpu()

        # ``task`` is the pre-inference snapshot: a /subtask landing mid-inference must
        # not relabel this action.
        self._set_dispatched_task(task)
        # Policy's own dimension order — the order ``send_next_action`` labels it with.
        return action_tensor

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when the policy has a text head."""
        return self._policy.supports_text_generation()

    @property
    def control_thread_owns_policy(self) -> bool:
        """Inference runs inline on the control thread, so queries are served there too."""
        return True

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head on the current observation."""
        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        # Live task, read without consuming the task-changed edge (the action path needs it).
        task = self.task
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(obs_frame, self._device, task, self._robot_type)
            observation = self._mark_query(observation, query)
            # Reusing the action path's preprocessor is safe only while its steps are
            # stateless per call; the one that is not (an enabled RelativeActionsProcessorStep)
            # is rejected for this backend at context-build time.
            observation = self._preprocessor(observation)
            # No str() coercion: _service_query validates the return value.
            return self._policy.generate_text(observation)

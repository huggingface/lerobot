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
from collections import deque
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.

    Relative-action policies freeze the preprocessor reference state for the
    lifetime of a ``select_action`` queue so queued relative steps stay
    anchored to the observation that produced the chunk.
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
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._relative_step = next(
            (
                step
                for step in getattr(preprocessor, "steps", ())
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d, relative_actions=%s)",
            self._device,
            len(ordered_action_keys),
            self._relative_step is not None,
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
        if self._relative_step is not None:
            self._relative_step.unfreeze_reference_state()

    def _policy_action_queue_empty(self) -> bool:
        queue = getattr(self._policy, "_action_queue", None)
        if queue is None:
            return True
        if isinstance(queue, deque):
            return len(queue) == 0
        try:
            return len(queue) == 0
        except TypeError:
            return True

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
        with torch.inference_mode(), autocast_ctx:
            if self._relative_step is not None and self._policy_action_queue_empty():
                # New chunk: allow preprocessor to refresh the absolute reference.
                self._relative_step.unfreeze_reference_state()

            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)

            if self._relative_step is not None:
                # Hold the chunk reference while select_action drains its queue.
                self._relative_step.freeze_reference_state()

            action = self._policy.select_action(observation)
            action = self._postprocessor(action)

            if self._relative_step is not None and self._policy_action_queue_empty():
                self._relative_step.unfreeze_reference_state()

        action_tensor = action.squeeze(0).cpu()

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

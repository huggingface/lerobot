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
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RelativeActionsProcessorStep

from .base import InferenceEngine

logger = logging.getLogger(__name__)


# Relative-action support (drift-free anchoring)
# ----------------------------------------------
# Relative-action policies predict a *chunk* of offsets anchored to the robot
# state at chunk-prediction time.  ``select_action`` serves that chunk one action
# per tick from an internal ``_action_queue``, recomputing only when the queue is
# empty.  The per-tick flow here runs the full pre/post pipeline every call, and
# ``RelativeActionsProcessorStep`` would otherwise refresh its cached anchor state
# on every tick — so actions popped from the queue on later ticks would be
# re-anchored to the *current* (already-moved) state and absolute targets would
# drift through the chunk.
#
# Fix: detect chunk boundaries by inspecting the policy's ``_action_queue`` length
# *before* running the pipeline, and freeze the relative step's cached anchor
# (``set_hold``) on ticks that pop a cached action.  The whole chunk is then
# anchored to a single state, exactly like RTC.  ``select_action`` stays on the
# hot path, so policy-specific side effects (e.g. LingBot-VA's per-tick keyframe
# feedback) are preserved.  Policies without an ``_action_queue`` (e.g. ACT
# temporal ensembling, which recomputes every tick) fall back to refreshing the
# anchor every tick, which is the correct behaviour there.


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.
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

        # Relative-action policies need the chunk anchor held while cached actions
        # are popped (see module docstring).  Introspect the preprocessor for an
        # enabled RelativeActionsProcessorStep, mirroring the RTC engine.
        self._relative_step = next(
            (
                s
                for s in getattr(preprocessor, "steps", ())
                if isinstance(s, RelativeActionsProcessorStep) and s.enabled
            ),
            None,
        )
        if self._relative_step is not None:
            if self._relative_step.action_names is None:
                cfg_names = getattr(policy.config, "action_feature_names", None)
                self._relative_step.action_names = list(cfg_names) if cfg_names else list(ordered_action_keys)
            logger.info("Relative actions enabled: chunk anchor will be held per chunk")

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
        # ``policy.reset()`` empties ``_action_queue`` so the next ``get_action``
        # recomputes and refreshes the anchor; clear any leftover hold defensively.
        if self._relative_step is not None:
            self._relative_step.set_hold(False)

    def _policy_will_recompute(self) -> bool:
        """True if the next ``select_action`` will predict a fresh chunk (queue empty/absent).

        Relative-action policies expose an ``_action_queue`` deque that is refilled
        only when empty.  When it is non-empty the upcoming ``select_action`` will
        pop a cached action, so the anchor state must be held.  Policies without the
        attribute recompute every tick, so we always refresh the anchor.
        """
        queue = getattr(self._policy, "_action_queue", None)
        return queue is None or len(queue) == 0

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
        # For relative-action policies, hold the cached anchor on ticks that pop a
        # cached action so the whole chunk stays anchored to the state captured when
        # it was predicted.  Decided before the pipeline runs (the queue is drained
        # inside ``select_action``); always released in ``finally`` so a hold never
        # leaks across ticks or on exception.
        hold_anchor = self._relative_step is not None and not self._policy_will_recompute()
        if self._relative_step is not None:
            self._relative_step.set_hold(hold_anchor)
        try:
            with torch.inference_mode(), autocast_ctx:
                observation = prepare_observation_for_inference(
                    observation, self._device, self._task, self._robot_type
                )
                observation = self._preprocessor(observation)
                action = self._policy.select_action(observation)
                action = self._postprocessor(action)
        finally:
            if self._relative_step is not None:
                self._relative_step.set_hold(False)
        action_tensor = action.squeeze(0).cpu()

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

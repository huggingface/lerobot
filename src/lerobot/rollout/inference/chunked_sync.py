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
"""Chunked-sync inference engine: predict a full chunk, decode it once, serve it tick by tick."""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.rollout.inference.sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


class ChunkedSyncInferenceEngine(SyncInferenceEngine):
    """Predict the whole action chunk at once, postprocess it as a chunk
    (so relative actions are anchored to the observation that produced them),
    then pop one action per control tick.  When the queue empties, infer again.
    """

    def __init__(self, *args, execution_steps: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue: deque[torch.Tensor] = deque()
        self._execution_steps = execution_steps
        logger.info("ChunkedSyncInferenceEngine initialized (execution_steps=%s)", execution_steps)

    def reset(self) -> None:
        self._queue.clear()
        super().reset()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if self._queue:
            return self._queue.popleft()
        if obs_frame is None:
            return None

        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        task, task_changed = self._take_task()
        with torch.inference_mode(), autocast_ctx:
            if task_changed:
                self._policy.drop_queued_actions()
            observation = prepare_observation_for_inference(observation, self._device, task, self._robot_type)
            observation = self._preprocessor(observation)
            chunk = self._policy.predict_action_chunk(observation)  # (B, H, D)
            chunk = self._postprocessor(chunk)  # decoded as a chunk

        chunk = chunk.squeeze(0).cpu()  # (H, D)
        if self._execution_steps is not None:
            chunk = chunk[: self._execution_steps]

        for step in chunk:
            action_dict = make_robot_action(step, self._dataset_features)
            self._queue.append(torch.tensor([action_dict[k] for k in self._ordered_action_keys]))
        self._set_dispatched_task(task)
        logger.info("ChunkedSync: queued %d actions", len(self._queue))
        return self._queue.popleft()

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Deterministic clock, observations and chunks for runtime tests."""

import numpy as np
import torch

from lerobot.inference import (
    ActionChunk,
    ActionProvenance,
    ChunkRuntime,
    ExecutionMode,
    ObservationSnapshot,
)


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


def runtime(mode=ExecutionMode.CHUNK, max_age=10, training_max_delay=2):
    clock = Clock()
    runtime = ChunkRuntime(
        mode=mode,
        action_interval=0.1,
        refill_seconds=0.3,
        max_observation_age_s=max_age,
        action_timeout_s=3,
        startup_timeout_s=5,
        training_max_delay=training_max_delay,
        clock=clock,
    )
    runtime.active = True
    return runtime, clock


def observation(clock, task="a", version=0):
    return ObservationSnapshot({"state": np.zeros(1, dtype=np.float32)}, clock(), task, version, task)


def chunk(request, offset=0, length=4):
    actions = torch.arange(offset, offset + length, dtype=torch.float32).reshape(-1, 1)
    return ActionChunk(
        actions, actions, ActionProvenance(request.observation.capture_time, request.observation.task), length
    )

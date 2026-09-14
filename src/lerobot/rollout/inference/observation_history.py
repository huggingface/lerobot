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

import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.observation_history import resolve_observation_delta_indices
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_snapshot(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True)
class ObservationSnapshot:
    observation_id: int
    timestamp: float
    observation: dict[str, Any]


class ObservationHistory:
    """Bounded CPU history recorded at observation ticks, independent of inference cadence.

    The caller synchronizes append, clear and snapshot. Snapshots own their buffers;
    preprocessing receives copies so in-place processor steps cannot corrupt history.
    """

    def __init__(self, config: Any) -> None:
        self.offsets: dict[str, tuple[int, ...]] = {}
        for key in getattr(config, "input_features", None) or {}:
            if key != OBS_STATE and not key.startswith(OBS_IMAGE):
                continue
            indices = resolve_observation_delta_indices(config, key)
            if indices is None:
                continue
            if not indices or any(not isinstance(i, int) or i > 0 for i in indices):
                raise ValueError(f"RTC observation offsets for {key} must be nonempty past/current integers")
            self.offsets[key] = tuple(indices)
        self.enabled = any(indices != (0,) for indices in self.offsets.values())
        capacity = 1 + max((-min(indices) for indices in self.offsets.values()), default=0)
        self._frames: deque[ObservationSnapshot] = deque(maxlen=capacity)
        self._next_id = 0

    def append(self, observation: dict[str, Any]) -> ObservationSnapshot:
        frame = ObservationSnapshot(self._next_id, time.monotonic(), _cpu_snapshot(observation))
        self._next_id += 1
        self._frames.append(frame)
        return frame

    def clear(self) -> None:
        self._frames.clear()

    def snapshot(self) -> tuple[ObservationSnapshot, ...]:
        return tuple(self._frames)

    def preprocess(
        self,
        frames: tuple[ObservationSnapshot, ...],
        prepare: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Preprocess unique selected ticks oldest first, then stack requested features."""
        latest = len(frames) - 1
        selected = {
            key: [max(0, latest + offset) for offset in offsets] for key, offsets in self.offsets.items()
        }
        # Always finish on the latest observation, even if all requested offsets are
        # negative: stateful relative-action processors and language refer to now.
        needed = sorted({latest} | {index for indices in selected.values() for index in indices})
        processed = {index: prepare(_cpu_snapshot(frames[index].observation)) for index in needed}
        batch = dict(processed[latest])
        for key, indices in selected.items():
            if key not in batch:
                continue
            values = [processed[index][key] for index in indices]
            batch[key] = torch.stack(values, dim=1)
            padding = [latest + offset < 0 for offset in self.offsets[key]]
            batch[f"{key}_is_pad"] = torch.tensor(padding, dtype=torch.bool, device=batch[key].device)[
                None
            ].expand(batch[key].shape[0], -1)
        return batch

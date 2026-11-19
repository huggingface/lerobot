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

from collections import deque
from collections.abc import Iterator

import numpy as np


class FrameHistory:
    """The last `length` frames of one quantity, oldest first, as a controller's network reads them.

    Args:
        length: Number of frames kept.
        initial: The frame the history is filled with on construction and on `reset()`.
    """

    def __init__(self, length: int, initial: np.ndarray):
        self.initial = np.array(initial)
        self._frames: deque[np.ndarray] = deque(maxlen=length)
        self.reset()

    def reset(self) -> None:
        self._frames.clear()
        self._frames.extend(self.initial.copy() for _ in range(self._frames.maxlen or 0))

    def push(self, frame: np.ndarray) -> None:
        """Add the newest frame, dropping the oldest."""
        self._frames.append(np.asarray(frame).copy())

    def stacked(self) -> np.ndarray:
        """All frames concatenated, oldest first."""
        return np.concatenate(self._frames)

    @property
    def newest(self) -> np.ndarray:
        return self._frames[-1]

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._frames)

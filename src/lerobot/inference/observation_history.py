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

"""Control-frame camera sampling shared by local and remote chunk execution."""

from collections.abc import Sequence

import numpy as np


def sample_camera_history(
    frames: Sequence[np.ndarray], memory_frames: int, memory_stride: int
) -> tuple[tuple[np.ndarray, ...], tuple[bool, ...]]:
    """Select past frames, excluding current; mark episode-start repeats as padding."""
    indices = range(len(frames) - 1 - memory_frames * memory_stride, len(frames) - 1, memory_stride)
    return tuple(frames[max(0, index)] for index in indices), tuple(index < 0 for index in indices)

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

"""Frame-difference motion detector."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MotionDetector:
    """Frame-difference motion detector usable as a ``detect_fn``.

    Compares consecutive frames in grayscale and fires when the fraction of
    pixels whose intensity changed by more than ``PIXEL_DIFF_THRESHOLD`` exceeds
    ``motion_area_threshold``. Dependency-free (numpy only).
    """

    PIXEL_DIFF_THRESHOLD = 25.0

    def __init__(self, motion_area_threshold: float):
        self.motion_area_threshold = motion_area_threshold
        self._prev_gray: NDArray | None = None

    def __call__(self, frame: NDArray) -> bool:
        gray = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(float)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        diff = np.abs(gray - self._prev_gray)
        self._prev_gray = gray
        motion_ratio = float((diff > self.PIXEL_DIFF_THRESHOLD).mean())
        return motion_ratio > self.motion_area_threshold

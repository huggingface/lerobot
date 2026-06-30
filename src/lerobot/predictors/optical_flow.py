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

"""Dense optical-flow estimation for per-patch latent time-advance.

Unlike the analytic ``CubePredictor`` (which tracks one colour blob and reports a
single centroid velocity), this estimates a *dense* flow field between consecutive
overhead frames -- a per-pixel motion estimate with no colour heuristic. The
``latent_flow`` predictor mode pools this field onto the policy's patch grid and
advances each patch token by *its own* flow (see :mod:`latent_warp`), which is the
training-free, classical-CV realisation of the optical-flow-conditioned latent
forecasting in AHEAD / LaDi-WM (see ``docs/overhead-predictor.md``).

Backed by OpenCV's classical estimators (DIS by default, Farneback as a fallback),
so no learned model or extra submodule is needed. A learned flow backend
(SEA-RAFT, NeuFlow, ...) can later implement the same ``estimate`` contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FlowOutput:
    """Dense flow between the previous and current frame.

    ``flow`` is an ``(H, W, 2)`` per-pixel displacement field ``(dx, dy)`` in pixels
    *per ``dt`` seconds* (i.e. over the inter-frame interval). Dividing by ``dt``
    gives a velocity field; multiplying that by a lead horizon gives the
    time-advance displacement.
    """

    flow: NDArray
    dt: float


class DenseFlowEstimator:
    """Stateful dense optical-flow estimator over consecutive frames.

    Holds the previous (grayscale) frame and its timestamp; each call returns the
    flow from the previous frame to the current one, or ``None`` on the first frame
    (and whenever the frame size changes). Reset per episode via :meth:`reset`.
    """

    def __init__(self, algorithm: str = "dis", preset: str = "fast"):
        if algorithm not in ("dis", "farneback"):
            raise ValueError(f"algorithm must be 'dis' or 'farneback', got {algorithm!r}")
        self.algorithm = algorithm
        self.preset = preset
        self._impl = None  # lazily created cv2 DIS handle
        self._prev_gray: NDArray | None = None
        self._prev_time_s: float | None = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_time_s = None

    def __call__(self, frame: NDArray) -> FlowOutput | None:
        return self.estimate(frame, now_s=time.perf_counter())

    def estimate(self, frame: NDArray, now_s: float) -> FlowOutput | None:
        gray = self._to_gray(frame)
        prev = self._prev_gray
        prev_t = self._prev_time_s
        self._prev_gray = gray
        self._prev_time_s = now_s
        if prev is None or prev.shape != gray.shape:
            return None

        dt = now_s - prev_t if (prev_t is not None and now_s > prev_t) else 1e-3
        flow = self._calc(prev, gray)
        return FlowOutput(flow=flow, dt=float(dt))

    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: NDArray) -> NDArray[np.uint8]:
        import cv2

        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[..., :3]
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.float32)
                if rgb.max(initial=0.0) <= 1.0:
                    rgb = rgb * 255.0
                rgb = rgb.clip(0, 255).astype(np.uint8)
            return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        gray = arr
        if gray.dtype != np.uint8:
            gray = gray.astype(np.float32)
            if gray.max(initial=0.0) <= 1.0:
                gray = gray * 255.0
            gray = gray.clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(gray)

    def _calc(self, prev_gray: NDArray, gray: NDArray) -> NDArray:
        import cv2

        if self.algorithm == "farneback":
            return cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        if self._impl is None:
            presets = {
                "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
                "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
                "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
            }
            self._impl = cv2.DISOpticalFlow_create(presets.get(self.preset, presets["fast"]))
        return self._impl.calc(prev_gray, gray, None)

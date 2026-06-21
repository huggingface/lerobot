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

import numpy as np
import pytest

from lerobot.async_inference.supervisor import DetectorOutput, RedCubeSpeedDetector, SupervisorMonitor


def _frame_with_red_square(x0: int, y0: int = 20, size: int = 10) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[y0 : y0 + size, x0 : x0 + size, 0] = 255
    return frame


def test_red_cube_speed_detector_estimates_speed_and_threshold():
    detector = RedCubeSpeedDetector(
        slow_speed_px_s=40.0,
        fast_speed_px_s=200.0,
        min_chunk_size_threshold=0.25,
        max_chunk_size_threshold=0.75,
        urgent_speed_px_s=250.0,
    )

    first = detector.detect(_frame_with_red_square(10), now_s=1.0)
    second = detector.detect(_frame_with_red_square(30), now_s=1.1)

    assert first.center_px == (14.5, 24.5)
    assert first.speed_px_s is None
    assert second.center_px == (34.5, 24.5)
    assert second.speed_px_s == pytest.approx(200.0)
    assert second.effective_chunk_size_threshold == pytest.approx(0.75)
    assert second.replan_now is False


def test_red_cube_speed_detector_triggers_urgent_replan_for_fast_cube():
    detector = RedCubeSpeedDetector(
        slow_speed_px_s=40.0,
        fast_speed_px_s=200.0,
        min_chunk_size_threshold=0.25,
        max_chunk_size_threshold=0.75,
        urgent_speed_px_s=150.0,
    )

    detector.detect(_frame_with_red_square(10), now_s=1.0)
    output = detector.detect(_frame_with_red_square(30), now_s=1.1)

    assert output.replan_now is True
    assert output.reason == "red_cube_urgent_speed"


def test_red_cube_speed_detector_ignores_frames_without_red_cube():
    detector = RedCubeSpeedDetector(
        slow_speed_px_s=40.0,
        fast_speed_px_s=200.0,
        min_chunk_size_threshold=0.25,
        max_chunk_size_threshold=0.75,
        urgent_speed_px_s=150.0,
    )

    output = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8), now_s=1.0)

    assert output == DetectorOutput(reason="red_cube_not_visible")


def test_supervisor_monitor_normalizes_bool_detector_output():
    assert SupervisorMonitor._normalize_output(True) == DetectorOutput(replan_now=True, reason="motion")
    assert SupervisorMonitor._normalize_output(False) == DetectorOutput()

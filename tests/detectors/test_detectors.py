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

from dataclasses import dataclass, field

import draccus
import numpy as np
import pytest

from lerobot.detectors import (
    DetectorConfig,
    DetectorOutput,
    MotionDetector,
    MotionDetectorConfig,
    RedCubeSpeedDetector,
    RedCubeSpeedDetectorConfig,
    SupervisorConfig,
    make_detector,
    normalize_detector_output,
)


def _frame_with_red_square(x0: int, y0: int = 20, size: int = 10) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[y0 : y0 + size, x0 : x0 + size, 0] = 255
    return frame


# --- Detector algorithms -----------------------------------------------------


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

    assert first.target_visible is True
    assert first.center_px == (14.5, 24.5)
    assert first.speed_px_s is None
    assert second.target_visible is True
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

    assert output == DetectorOutput(target_visible=False, reason="red_cube_not_visible")


def test_motion_detector_fires_on_change():
    detector = MotionDetector(motion_area_threshold=0.02)
    blank = np.zeros((16, 16, 3), dtype=np.uint8)
    moved = _frame_with_red_square(2, y0=2, size=8)

    assert detector(blank) is False  # first frame primes the buffer
    assert detector(moved) is True
    assert detector(moved) is False  # no further change


def test_normalize_detector_output_handles_bool_and_struct():
    assert normalize_detector_output(True) == DetectorOutput(replan_now=True, reason="motion")
    assert normalize_detector_output(False) == DetectorOutput()
    passthrough = DetectorOutput(replan_now=True, reason="x")
    assert normalize_detector_output(passthrough) is passthrough


# --- Config / factory --------------------------------------------------------


def test_make_detector_builds_selected_backend():
    assert isinstance(make_detector(MotionDetectorConfig()), MotionDetector)
    assert isinstance(make_detector(RedCubeSpeedDetectorConfig()), RedCubeSpeedDetector)


def test_red_cube_config_validates_speed_ordering():
    with pytest.raises(ValueError):
        RedCubeSpeedDetectorConfig(slow_speed_px_s=200.0, fast_speed_px_s=100.0)
    with pytest.raises(ValueError):
        RedCubeSpeedDetectorConfig(min_chunk_size_threshold=0.8, max_chunk_size_threshold=0.5)


def test_motion_config_validates_threshold_range():
    with pytest.raises(ValueError):
        MotionDetectorConfig(motion_threshold=0.0)
    with pytest.raises(ValueError):
        MotionDetectorConfig(motion_threshold=1.5)


def test_supervisor_config_validates_only_when_enabled():
    # Disabled: bad poll_fps is tolerated (not validated).
    SupervisorConfig(enabled=False, poll_fps=0)
    with pytest.raises(ValueError):
        SupervisorConfig(enabled=True, poll_fps=0)


def test_detector_choice_parses_from_cli():
    @dataclass
    class W:
        detector: DetectorConfig = field(default_factory=MotionDetectorConfig)

    cfg = draccus.parse(
        W,
        args=["--detector.type=red_cube_speed", "--detector.urgent_speed_px_s=300"],
    )
    assert cfg.detector.type == "red_cube_speed"
    assert cfg.detector.urgent_speed_px_s == pytest.approx(300.0)

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

import dataclasses

import numpy as np
import pytest

from lerobot.predictors import (
    CubePredictor,
    CubePredictorConfig,
    PredictorConfig,
    PredictorOutput,
    shift_cube_in_frame,
)


def _frame_with_red_square(x0: int, y0: int = 20, size: int = 10) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[y0 : y0 + size, x0 : x0 + size, 0] = 255
    return frame


# --- CubePredictor -----------------------------------------------------------


def test_cube_predictor_estimates_center_and_velocity_vector():
    predictor = CubePredictor()

    first = predictor.detect(_frame_with_red_square(10), now_s=1.0)
    second = predictor.detect(_frame_with_red_square(30), now_s=1.1)

    assert first.target_visible is True
    assert first.center_px == (14.5, 24.5)
    assert first.velocity_px_s is None  # no estimate from a single frame
    assert second.center_px == (34.5, 24.5)
    # Cube moved +20px in x over 0.1s -> vx=200, vy=0.
    assert second.velocity_px_s == pytest.approx((200.0, 0.0))
    assert second.reason == "red_cube_tracked"


def test_cube_predictor_reports_not_visible():
    predictor = CubePredictor()
    output = predictor.detect(np.zeros((64, 64, 3), dtype=np.uint8), now_s=1.0)
    assert output == PredictorOutput(target_visible=False, reason="red_cube_not_visible")


def test_cube_predictor_resets_track_when_cube_disappears():
    predictor = CubePredictor()
    predictor.detect(_frame_with_red_square(10), now_s=1.0)
    predictor.detect(np.zeros((64, 64, 3), dtype=np.uint8), now_s=1.1)  # cube gone -> reset
    after = predictor.detect(_frame_with_red_square(30), now_s=1.2)
    assert after.velocity_px_s is None  # track restarted, no velocity yet


def test_predict_center_is_constant_velocity_extrapolation():
    assert CubePredictor.predict_center((10.0, 5.0), (200.0, -50.0), lead_s=0.1) == pytest.approx(
        (30.0, 0.0)
    )


# --- shift_cube_in_frame -----------------------------------------------------


def test_shift_cube_translates_mask_and_fills_hole():
    frame = _frame_with_red_square(10)
    mask = frame[..., 0] > 0
    shifted = shift_cube_in_frame(frame, mask, offset_px=(20.0, 0.0))

    # Cube erased from the original location (filled with black background)...
    assert shifted[20:30, 10:20, 0].max() == 0
    # ...and re-pasted 20px to the right.
    assert shifted[20:30, 30:40, 0].min() == 255


def test_shift_cube_noop_for_zero_offset():
    frame = _frame_with_red_square(10)
    mask = frame[..., 0] > 0
    shifted = shift_cube_in_frame(frame, mask, offset_px=(0.4, -0.4))  # rounds to (0, 0)
    assert np.array_equal(shifted, frame)


def test_shift_cube_drops_out_of_bounds_pixels():
    frame = _frame_with_red_square(58)  # near the right edge (size 10 -> cols 58..67 clipped)
    mask = frame[..., 0] > 0
    shifted = shift_cube_in_frame(frame, mask, offset_px=(100.0, 0.0))  # entirely off-frame
    assert shifted[..., 0].max() == 0  # cube shifted out, nothing pasted


# --- Config ------------------------------------------------------------------


def test_cube_predictor_config_makes_predictor():
    assert isinstance(CubePredictorConfig().make(), CubePredictor)
    assert isinstance(PredictorConfig().make(), CubePredictor)


def test_cube_predictor_config_validates_ranges():
    with pytest.raises(ValueError):
        CubePredictorConfig(saturation_min=1.5)
    with pytest.raises(ValueError):
        CubePredictorConfig(min_area_ratio=0.0)


def test_predictor_config_validates_camera_only_when_enabled():
    PredictorConfig(enabled=False, camera="")  # tolerated when disabled
    with pytest.raises(ValueError):
        PredictorConfig(enabled=True, camera="")


def test_predictor_output_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PredictorOutput().center_px = (1.0, 2.0)

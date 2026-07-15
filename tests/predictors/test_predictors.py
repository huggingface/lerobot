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
import torch

from lerobot.predictors import (
    CubePredictor,
    CubePredictorConfig,
    DenseFlowEstimator,
    PredictorConfig,
    PredictorOutput,
    make_flow_token_warp_fn,
    make_token_warp_fn,
    mask_to_token_grid,
    shift_cube_in_frame,
    warp_token_grid,
    warp_token_grid_by_flow,
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
    assert CubePredictor.predict_center((10.0, 5.0), (200.0, -50.0), lead_s=0.1) == pytest.approx((30.0, 0.0))


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


# --- latent_warp -------------------------------------------------------------


def _grid_tokens(grid_h, grid_w, dim=2):
    """(grid_h*grid_w, dim) row-major tokens, each token equal to its flat index."""
    n = grid_h * grid_w
    return torch.arange(n, dtype=torch.float32).unsqueeze(-1).repeat(1, dim)


def test_warp_token_grid_translates_masked_tokens_and_fills_hole():
    tokens = torch.zeros(16, 2)
    tokens[1 * 4 + 1] = torch.tensor([7.0, 7.0])  # cube token at grid (row=1, col=1)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[1, 1] = True

    out = warp_token_grid(tokens, (4, 4), mask, offset_tokens=(1.0, 0.0))  # +1 col

    # Source cell erased to the background token (median of non-cube tokens = 0)...
    assert torch.equal(out[1 * 4 + 1], torch.tensor([0.0, 0.0]))
    # ...and the cube token re-written one column to the right.
    assert torch.equal(out[1 * 4 + 2], torch.tensor([7.0, 7.0]))


def test_warp_token_grid_noop_for_zero_offset():
    tokens = _grid_tokens(4, 4)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[2, 2] = True
    out = warp_token_grid(tokens, (4, 4), mask, offset_tokens=(0.4, -0.4))  # rounds to (0, 0)
    assert torch.equal(out, tokens)


def test_warp_token_grid_drops_out_of_bounds_tokens():
    tokens = torch.zeros(16, 2)
    tokens[1 * 4 + 3] = torch.tensor([5.0, 5.0])  # cube at right edge (col 3)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[1, 3] = True
    out = warp_token_grid(tokens, (4, 4), mask, offset_tokens=(2.0, 0.0))  # shifted off-grid
    assert out.abs().max() == 0  # cube dropped, hole filled with background (0)


def test_warp_token_grid_supports_batched_tokens():
    tokens = torch.zeros(2, 16, 2)
    tokens[:, 1 * 4 + 1] = torch.tensor([3.0, 3.0])
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[1, 1] = True
    out = warp_token_grid(tokens, (4, 4), mask, offset_tokens=(0.0, 1.0))  # +1 row
    assert out.shape == (2, 16, 2)
    assert torch.equal(out[:, 2 * 4 + 1], torch.full((2, 2), 3.0))


def test_mask_to_token_grid_pools_to_patches():
    pixel_mask = np.zeros((8, 8), dtype=bool)
    pixel_mask[4:6, 4:6] = True  # one 2x2 patch worth of cube -> grid cell (2, 2)
    grid_mask = mask_to_token_grid(pixel_mask, (4, 4))
    assert grid_mask.sum() == 1
    assert bool(grid_mask[2, 2])


def test_make_token_warp_fn_advances_cube_on_token_grid():
    pixel_mask = np.zeros((8, 8), dtype=bool)
    pixel_mask[4:6, 4:6] = True  # maps to grid (2, 2) for a 4x4 token grid
    # +2px in x at 8px / 4-grid -> +1 token column.
    warp = make_token_warp_fn(pixel_mask, offset_px=(2.0, 0.0), image_hw=(8, 8))

    tokens = torch.zeros(1, 16, 2)
    tokens[:, 2 * 4 + 2] = torch.tensor([9.0, 9.0])  # cube token at grid (2, 2)
    out = warp(tokens)

    assert torch.equal(out[:, 2 * 4 + 2].squeeze(0), torch.tensor([0.0, 0.0]))  # erased
    assert torch.equal(out[:, 2 * 4 + 3].squeeze(0), torch.tensor([9.0, 9.0]))  # moved +1 col


def test_make_token_warp_fn_skips_non_square_token_grid():
    warp = make_token_warp_fn(np.ones((8, 8), dtype=bool), offset_px=(4.0, 0.0), image_hw=(8, 8))
    tokens = torch.randn(1, 12, 2)  # 12 is not a perfect square -> unknown layout
    assert torch.equal(warp(tokens), tokens)


# --- per-patch flow warp -----------------------------------------------------


def test_warp_token_grid_by_flow_uniform_flow_translates_grid():
    # Distinct token per cell; a uniform +1 column flow shifts every cell right by 1.
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    flow = torch.zeros(4, 4, 2)
    flow[..., 0] = 1.0  # +1 patch in x for every cell
    out = warp_token_grid_by_flow(tokens, (4, 4), flow).reshape(4, 4)
    # Column c now holds what column c-1 held (border-replicated at the left edge).
    expected = torch.tensor(
        [[r * 4 + max(c - 1, 0) for c in range(4)] for r in range(4)], dtype=torch.float32
    )
    assert torch.allclose(out, expected)


def test_warp_token_grid_by_flow_motion_mask_keeps_static_cells():
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    flow = torch.zeros(4, 4, 2)
    flow[1, 1, 0] = 1.0  # only one cell "moves"
    motion_mask = torch.zeros(4, 4, dtype=torch.bool)
    motion_mask[1, 1] = True
    out = warp_token_grid_by_flow(tokens, (4, 4), flow, motion_mask=motion_mask).reshape(16)
    untouched = torch.arange(16, dtype=torch.float32).clone()
    untouched[1 * 4 + 1] = float(1 * 4 + 0)  # cell (1,1) sampled from (1,0)
    assert torch.allclose(out, untouched)


def test_make_flow_token_warp_fn_pools_field_and_warps():
    # Uniform +2px/x flow at 8px over a 4-grid -> +1 token column everywhere.
    flow_px = np.zeros((8, 8, 2), dtype=np.float32)
    flow_px[..., 0] = 2.0
    warp = make_flow_token_warp_fn(flow_px, image_hw=(8, 8))
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    out = warp(tokens).reshape(4, 4)
    expected = torch.tensor(
        [[r * 4 + max(c - 1, 0) for c in range(4)] for r in range(4)], dtype=torch.float32
    )
    assert torch.allclose(out, expected)


def test_make_flow_token_warp_fn_skips_non_square_token_grid():
    warp = make_flow_token_warp_fn(np.ones((8, 8, 2), dtype=np.float32), image_hw=(8, 8))
    tokens = torch.randn(1, 12, 2)
    assert torch.equal(warp(tokens), tokens)


# --- DenseFlowEstimator ------------------------------------------------------


def _shifted_texture(dx: int) -> np.ndarray:
    """A textured frame translated by dx px in x (so flow should report ~+dx)."""
    rng = np.arange(48)
    base = ((rng[None, :] * 7 + rng[:, None] * 3) % 255).astype(np.uint8)
    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    shifted = np.roll(base, dx, axis=1)
    frame[..., 0] = frame[..., 1] = frame[..., 2] = shifted
    return frame


def test_dense_flow_estimator_returns_none_on_first_frame():
    est = DenseFlowEstimator()
    assert est.estimate(_shifted_texture(0), now_s=0.0) is None


def test_dense_flow_estimator_tracks_horizontal_shift():
    est = DenseFlowEstimator(algorithm="farneback")
    est.estimate(_shifted_texture(0), now_s=0.0)
    out = est.estimate(_shifted_texture(3), now_s=0.1)  # content moved +3px in x
    assert out is not None
    assert out.flow.shape == (48, 48, 2)
    assert out.dt == pytest.approx(0.1)
    assert out.flow[..., 0].mean() > 0.5  # positive x flow detected


def test_dense_flow_estimator_reset_clears_history():
    est = DenseFlowEstimator()
    est.estimate(_shifted_texture(0), now_s=0.0)
    est.reset()
    assert est.estimate(_shifted_texture(3), now_s=0.1) is None  # history cleared


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


def test_predictor_config_defaults_to_image_shift():
    assert PredictorConfig().mode == "image_shift"


def test_predictor_config_validates_mode_and_threshold():
    PredictorConfig(mode="latent_warp", latent_mask_threshold=0.3)
    PredictorConfig(mode="latent_flow", flow_algorithm="farneback", flow_motion_threshold=0.5)
    PredictorConfig(
        mode="engage_gate",
        engage_axis="y",
        engage_threshold=0.4,
        engage_direction="negative",
        engage_lead_s=0.2,
    )
    with pytest.raises(ValueError):
        PredictorConfig(mode="latent_shift")  # typo / unknown mode
    with pytest.raises(ValueError):
        PredictorConfig(latent_mask_threshold=1.0)
    with pytest.raises(ValueError):
        PredictorConfig(flow_algorithm="raft")  # unsupported backend
    with pytest.raises(ValueError):
        PredictorConfig(flow_motion_threshold=-0.1)
    with pytest.raises(ValueError):
        PredictorConfig(engage_threshold=1.1)
    with pytest.raises(ValueError):
        PredictorConfig(engage_lead_s=-0.1)


def test_predictor_config_make_flow_builds_estimator():
    est = PredictorConfig(mode="latent_flow").make_flow()
    assert isinstance(est, DenseFlowEstimator)


def test_predictor_output_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PredictorOutput().center_px = (1.0, 2.0)

#!/usr/bin/env python

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

import math

import numpy as np
import pytest

from lerobot.scripts.lerobot_find_cameras import (
    _check_cv2_gui_available,
    build_camera_grid,
)


def _frame(h: int, w: int, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _expected_grid(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 9])
def test_grid_shape_matches_layout_formula(n):
    tile_h, tile_w = 480, 640
    frames = [_frame(120, 160) for _ in range(n)]
    labels = [f"OpenCV {i}" for i in range(n)]

    grid = build_camera_grid(frames, labels, tile_size=(tile_h, tile_w))

    rows, cols = _expected_grid(n)
    assert grid.shape == (rows * tile_h, cols * tile_w, 3)
    assert grid.dtype == np.uint8


def test_non_uniform_input_sizes_are_resized():
    tile_h, tile_w = 240, 320
    frames = [_frame(120, 160), _frame(1080, 1920)]
    labels = ["OpenCV 0", "RealSense 1"]

    grid = build_camera_grid(frames, labels, tile_size=(tile_h, tile_w))

    rows, cols = _expected_grid(2)
    assert grid.shape == (rows * tile_h, cols * tile_w, 3)


def test_none_frame_becomes_black_tile():
    tile_h, tile_w = 200, 200
    frames = [None, _frame(100, 100, value=255)]
    labels = ["OpenCV 0", "OpenCV 1"]

    grid = build_camera_grid(frames, labels, tile_size=(tile_h, tile_w))

    # First tile (top-left) corresponds to the None frame: its corner stays black
    # (the label is drawn near (10, 30), so sample an untouched corner pixel).
    assert grid[tile_h - 1, 0].tolist() == [0, 0, 0]


def test_padding_cell_is_black_when_not_a_perfect_rectangle():
    tile_h, tile_w = 100, 100
    frames = [_frame(50, 50, value=200) for _ in range(3)]
    labels = [f"OpenCV {i}" for i in range(3)]

    grid = build_camera_grid(frames, labels, tile_size=(tile_h, tile_w))

    rows, cols = _expected_grid(3)  # 2x2
    assert (rows, cols) == (2, 2)
    # The 4th cell (bottom-right) is padding -> fully black.
    bottom_right = grid[tile_h:, tile_w:]
    assert not bottom_right.any()


def test_empty_list_returns_single_black_tile():
    tile_h, tile_w = 480, 640
    grid = build_camera_grid([], [], tile_size=(tile_h, tile_w))
    assert grid.shape == (tile_h, tile_w, 3)
    assert grid.dtype == np.uint8
    assert not grid.any()


def test_check_cv2_gui_available_raises_on_headless(monkeypatch):
    import cv2

    def _raise(*args, **kwargs):
        raise cv2.error("The function is not implemented. Rebuild the library with GUI support.")

    monkeypatch.setattr(cv2, "namedWindow", _raise)

    with pytest.raises(RuntimeError, match="opencv-python"):
        _check_cv2_gui_available()

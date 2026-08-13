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

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("transformers")

from lerobot.rewards.rynnvalue.compute_rabc_weights import (  # noqa: E402
    _build_episode_table,
    _interpolate_anchor_values,
    _remaining_time_to_progress,
    _select_anchor_indices,
    _select_prefix_indices,
)


def test_anchor_selection_uses_inference_fps_and_keeps_boundaries():
    assert _select_anchor_indices(61, dataset_fps=30, inference_fps=1).tolist() == [0, 30, 60]
    assert _select_anchor_indices(1, dataset_fps=30, inference_fps=1).tolist() == [0]

    with pytest.raises(ValueError, match="inference_fps"):
        _select_anchor_indices(10, dataset_fps=30, inference_fps=31)


def test_prefix_selection_is_causal_and_uniform():
    assert _select_prefix_indices(10, max_frames=4).tolist() == [0, 3, 7, 10]
    assert _select_prefix_indices(10, max_frames=1).tolist() == [10]
    assert _select_prefix_indices(2, max_frames=8).tolist() == [0, 1, 2]
    assert _select_prefix_indices(2, max_frames=None).tolist() == [0, 1, 2]


def test_anchor_predictions_are_interpolated_without_changing_anchors():
    anchors = np.asarray([0, 2, 4], dtype=np.int64)
    values = np.asarray([4.0, 2.0, 0.0], dtype=np.float32)
    dense = _interpolate_anchor_values(5, anchors, values)

    np.testing.assert_allclose(dense, [4.0, 3.0, 2.0, 1.0, 0.0])
    np.testing.assert_allclose(dense[anchors], values)


def test_horizon_normalization_is_bounded_and_not_episode_relative():
    remaining_time = np.asarray([12.0, 10.0, 5.0, 0.0, -1.0], dtype=np.float32)
    progress = _remaining_time_to_progress(remaining_time, max_remaining_time_s=10.0)

    np.testing.assert_allclose(progress, [0.0, 0.0, 0.5, 1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        _remaining_time_to_progress(remaining_time, max_remaining_time_s=0)


def test_episode_table_preserves_raw_values_and_adds_optional_rabc_column():
    remaining_time = np.asarray([4.0, 2.0, 0.0], dtype=np.float32)
    anchors = np.asarray([0, 2], dtype=np.int64)
    table = _build_episode_table(
        global_start=10,
        episode_index=3,
        remaining_time_s=remaining_time,
        anchor_indices=anchors,
        max_remaining_time_s=4.0,
    )

    assert table.column_names == [
        "index",
        "episode_index",
        "frame_index",
        "remaining_time_s",
        "potential",
        "is_inference_frame",
        "progress_sparse",
    ]
    assert table["index"].to_pylist() == [10, 11, 12]
    assert table["episode_index"].to_pylist() == [3, 3, 3]
    assert table["potential"].to_pylist() == [-4.0, -2.0, -0.0]
    assert table["is_inference_frame"].to_pylist() == [True, False, True]
    assert table["progress_sparse"].to_pylist() == [0.0, 0.5, 1.0]

    raw_table = _build_episode_table(
        global_start=0,
        episode_index=0,
        remaining_time_s=remaining_time,
        anchor_indices=anchors,
        max_remaining_time_s=None,
    )
    assert "progress_sparse" not in raw_table.column_names

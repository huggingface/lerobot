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

pytest.importorskip("cv2")
pd = pytest.importorskip("pandas")

from examples.dataset.create_progress_videos import (  # noqa: E402
    GRAPH_Y_BOT_FRAC,
    GRAPH_Y_TOP_FRAC,
    _precompute_pixel_coords,
    load_progress_data,
)


def test_load_progress_data_accepts_local_path_and_explicit_value_column(tmp_path):
    parquet_path = tmp_path / "rynnvalue.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 1, 0],
            "frame_index": [1, 0, 0],
            "remaining_time_s": [2.0, 9.0, 3.0],
        }
    ).to_parquet(parquet_path)

    values = load_progress_data(
        tmp_path,
        episode=0,
        progress_path=parquet_path,
        value_column="remaining_time_s",
    )

    np.testing.assert_allclose(values, [[0, 3.0], [1, 2.0]])


def test_load_progress_data_reports_missing_explicit_column(tmp_path):
    parquet_path = tmp_path / "rynnvalue.parquet"
    pd.DataFrame(
        {
            "episode_index": [0],
            "frame_index": [0],
            "potential": [-3.0],
        }
    ).to_parquet(parquet_path)

    with pytest.raises(ValueError, match="remaining_time_s"):
        load_progress_data(
            tmp_path,
            episode=0,
            progress_path=parquet_path,
            value_column="remaining_time_s",
        )


def test_pixel_coordinates_support_physical_value_range():
    frame_height = 100
    coordinates = _precompute_pixel_coords(
        np.asarray([[0, 20.0], [1, 10.0], [2, 0.0]]),
        num_frames=3,
        frame_width=101,
        frame_height=frame_height,
        value_min=0.0,
        value_max=20.0,
    )

    assert coordinates[:, 0].tolist() == [0, 50, 100]
    assert coordinates[0, 1] == int(frame_height * GRAPH_Y_TOP_FRAC)
    assert coordinates[-1, 1] == int(frame_height * GRAPH_Y_BOT_FRAC)

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
av = pytest.importorskip("av")
pd = pytest.importorskip("pandas")

from examples.dataset.create_progress_videos import (  # noqa: E402
    GRAPH_Y_BOT_FRAC,
    GRAPH_Y_TOP_FRAC,
    _iter_episode_frames_pyav,
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


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_load_progress_data_auto_detects_local_progress(tmp_path, suffix):
    path = tmp_path / f"progress{suffix}"
    frame = pd.DataFrame(
        {
            "episode_index": [1, 0, 0],
            "frame_index": [0, 1, 0],
            "progress": [0.9, 0.6, 0.2],
        }
    )
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)

    values = load_progress_data(tmp_path, episode=0, progress_path=path)
    np.testing.assert_allclose(values, [[0, 0.2], [1, 0.6]])


def test_load_progress_data_returns_none_for_missing_episode(tmp_path):
    path = tmp_path / "progress.csv"
    pd.DataFrame(
        {
            "episode_index": [0],
            "frame_index": [0],
            "progress": [0.2],
        }
    ).to_csv(path, index=False)

    assert load_progress_data(tmp_path, episode=1, progress_path=path) is None


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


def test_pyav_decoder_reads_requested_episode_segment(tmp_path):
    video_path = tmp_path / "source.mp4"
    with av.open(video_path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 8
        stream.height = 8
        stream.pix_fmt = "yuv420p"
        stream.gop_size = 1
        for index in range(6):
            image = np.full((8, 8, 3), index * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    frames = list(
        _iter_episode_frames_pyav(
            video_path,
            from_timestamp=0.3,
            num_frames=3,
            fps=10,
        )
    )

    assert len(frames) == 3
    assert all(frame.shape == (8, 8, 3) for frame in frames)
    assert [float(frame.mean()) for frame in frames] == pytest.approx([90, 120, 150], abs=3)

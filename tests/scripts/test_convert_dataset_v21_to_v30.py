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

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from lerobot.scripts.convert_dataset_v21_to_v30 import convert_data, validate_episode_labels


@pytest.fixture
def episode_frames() -> list[pd.DataFrame]:
    return [
        pd.DataFrame({"episode_index": [0, 0, 0], "index": [0, 1, 2], "value": [10, 11, 12]}),
        pd.DataFrame({"episode_index": [1, 1], "index": [3, 4], "value": [13, 14]}),
        pd.DataFrame({"episode_index": [2, 2, 2, 2], "index": [5, 6, 7, 8], "value": [15, 16, 17, 18]}),
    ]


@pytest.fixture
def write_v21_dataset(tmp_path: Path) -> Callable[[list[pd.DataFrame]], tuple[Path, list[Path]]]:
    def write(frames: list[pd.DataFrame]) -> tuple[Path, list[Path]]:
        root = tmp_path / "v21"
        meta_dir = root / "meta"
        meta_dir.mkdir(parents=True)
        info = {
            "codebase_version": "v2.1",
            "fps": 30,
            "features": {
                key: {"dtype": "int64", "shape": [1], "names": None}
                for key in ("episode_index", "index", "value")
            },
        }
        (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
        paths = [
            root / "data/chunk-000/episode_000010.parquet",
            root / "data/chunk-000/episode_000020.parquet",
            root / "data/chunk-001/episode_000000.parquet",
        ]
        # Create files in reverse order to distinguish creation order from sorted path order.
        for path, frame in reversed(list(zip(paths, frames, strict=True))):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, engine="pyarrow", index=False)
        return root, paths

    return write


@pytest.mark.parametrize(
    "ep_idx,first_index,num_rows",
    [(0, 0, 3), (7, 23, 4), (2, 11, 1), (4, 12, 0)],
    ids=["first-episode", "later-episode", "single-row", "empty-episode"],
)
def test_validate_episode_labels_accepts_consistent_rows(
    tmp_path: Path, ep_idx: int, first_index: int, num_rows: int
) -> None:
    ep_path = tmp_path / "episode.parquet"
    frame = pd.DataFrame(
        {
            "episode_index": pd.Series([ep_idx] * num_rows, dtype="int64"),
            "index": pd.Series(range(first_index, first_index + num_rows), dtype="int64"),
            "value": pd.Series(range(num_rows), dtype="int64"),
        }
    )
    frame.to_parquet(ep_path, engine="pyarrow", index=False)

    assert validate_episode_labels(ep_path, ep_idx, first_index) is None


@pytest.mark.parametrize(
    "column,labels,num_differing",
    [
        ("episode_index", [3, 7, 3, 7], 2),
        ("episode_index", [7, 7, 7, 7], 4),
        ("index", [10, 11, 14, 13], 1),
        ("index", [10, 12, 11, 13], 2),
        ("index", [0, 1, 2, 3], 4),
        ("index", [11, 12, 13, 14], 4),
    ],
    ids=["partial-episode-label", "wrong-episode", "index-gap", "row-order", "index-restart", "index-offset"],
)
def test_validate_episode_labels_reports_path_column_and_differing_rows(
    tmp_path: Path, column: str, labels: list[int], num_differing: int
) -> None:
    ep_path = tmp_path / "episode.parquet"
    frame = pd.DataFrame({"episode_index": [3, 3, 3, 3], "index": [10, 11, 12, 13]})
    frame[column] = labels
    frame.to_parquet(ep_path, engine="pyarrow", index=False)

    with pytest.raises(ValueError) as exc_info:
        validate_episode_labels(ep_path, ep_idx=3, first_index=10)

    message = str(exc_info.value)
    assert str(ep_path) in message
    assert column in message
    assert f"{num_differing} of 4 rows differ" in message


def test_convert_data_preserves_sorted_episodes_and_cumulative_ranges(
    tmp_path: Path,
    episode_frames: list[pd.DataFrame],
    write_v21_dataset: Callable[[list[pd.DataFrame]], tuple[Path, list[Path]]],
) -> None:
    root, _ = write_v21_dataset(episode_frames)
    new_root = tmp_path / "v30"

    metadata = convert_data(root, new_root, data_file_size_in_mb=1)

    assert metadata == [
        {
            "episode_index": 0,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": 3,
        },
        {
            "episode_index": 1,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 3,
            "dataset_to_index": 5,
        },
        {
            "episode_index": 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 5,
            "dataset_to_index": 9,
        },
    ]
    output_paths = sorted((new_root / "data").glob("*/*.parquet"))
    assert output_paths == [new_root / "data/chunk-000/file-000.parquet"]
    actual = pq.read_table(output_paths[0]).to_pandas()
    expected = pd.concat(episode_frames, ignore_index=True)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "episode,column,labels,num_differing",
    [
        (1, "episode_index", [0, 0], 2),
        (1, "episode_index", [1, 0], 1),
        (1, "index", [0, 1], 2),
        (1, "index", [3, 5], 1),
        (1, "index", [4, 3], 2),
        (0, "index", [1, 2, 3], 3),
        (2, "index", [5, 6, 7, 9], 1),
    ],
    ids=[
        "later-wrong-episode",
        "later-partial-episode-label",
        "later-index-restart",
        "later-index-gap",
        "later-row-order",
        "first-index-offset",
        "third-index-gap",
    ],
)
def test_convert_data_rejects_inconsistent_episode_labels(
    tmp_path: Path,
    episode_frames: list[pd.DataFrame],
    write_v21_dataset: Callable[[list[pd.DataFrame]], tuple[Path, list[Path]]],
    episode: int,
    column: str,
    labels: list[int],
    num_differing: int,
) -> None:
    episode_frames[episode][column] = labels
    root, paths = write_v21_dataset(episode_frames)

    with pytest.raises(ValueError) as exc_info:
        convert_data(root, tmp_path / "v30", data_file_size_in_mb=1)

    message = str(exc_info.value)
    assert str(paths[episode]) in message
    assert column in message
    assert f"{num_differing} of {len(labels)} rows differ" in message

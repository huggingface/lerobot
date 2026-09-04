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

"""Tests for frame-signal output reading and legacy compatibility."""

import numpy as np
import pytest

from lerobot.rewards.scoring import (
    get_scoring_provenance,
    get_signal_descriptors,
    read_frame_signals,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_read_frame_signals_recognizes_legacy_progress_artifact(tmp_path):
    path = tmp_path / "robometer_progress.parquet"
    table = pa.table(
        {
            "index": np.asarray([0, 1], dtype=np.int64),
            "episode_index": np.asarray([0, 0], dtype=np.int64),
            "frame_index": np.asarray([0, 1], dtype=np.int64),
            "progress_sparse": np.asarray([0.1, 0.9], dtype=np.float32),
        }
    ).replace_schema_metadata({b"reward_model_path": b"lerobot/Robometer-4B"})
    pq.write_table(table, path)

    loaded = read_frame_signals(path)

    assert loaded.column_names == table.column_names
    descriptor = get_signal_descriptors(loaded)["progress_sparse"]
    assert descriptor.direction == "higher"
    assert descriptor.bounds is None
    assert descriptor.missing_values == "forbidden"
    assert get_scoring_provenance(loaded) == {
        "legacy_output": True,
        "reward_model_path": "lerobot/Robometer-4B",
        "source_path": str(path),
    }


def test_read_frame_signals_rejects_unrecognized_parquet(tmp_path):
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"value": [1.0]}), path)

    with pytest.raises(ValueError, match="Not a recognized"):
        read_frame_signals(path)


def test_read_frame_signals_preserves_missing_values_in_legacy_progress(tmp_path):
    path = tmp_path / "sarm_progress.parquet"
    table = pa.table(
        {
            "index": np.asarray([0, 1], dtype=np.int64),
            "episode_index": np.asarray([0, 0], dtype=np.int64),
            "frame_index": np.asarray([0, 1], dtype=np.int64),
            "progress_sparse": np.asarray([2.0, np.nan], dtype=np.float32),
        }
    )
    pq.write_table(table, path)

    loaded = read_frame_signals(path)

    assert get_signal_descriptors(loaded)["progress_sparse"].missing_values == "nan"
    assert loaded["progress_sparse"].to_numpy(zero_copy_only=False)[0] == 2.0
    assert np.isnan(loaded["progress_sparse"].to_numpy(zero_copy_only=False)[1])

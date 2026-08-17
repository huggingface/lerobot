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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.rewards.compute_chunk_td_residuals import (
    CHUNK_TD_METADATA_KEY,
    _discounted_chunk_return,
    compute_chunk_td_residuals,
)


def _write_scores(path, scores, *, column="model_score"):
    frame = pd.DataFrame(
        {
            "episode_index": np.zeros(len(scores), dtype=np.int64),
            "frame_index": np.arange(len(scores), dtype=np.int64),
            column: np.asarray(scores, dtype=np.float32),
        }
    )
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def test_cost_to_go_scores_produce_expected_residual(tmp_path):
    source = tmp_path / "scores.parquet"
    output = tmp_path / "chunks.parquet"
    _write_scores(source, np.linspace(10.0, 8.0, 61), column="remaining_time_s")

    compute_chunk_td_residuals(
        source,
        output,
        score_column="remaining_time_s",
        score_semantics="cost_to_go",
        reward_per_second=-1.0,
        fps=30.0,
        chunk_size=30,
    )
    table = pq.read_table(output)

    assert table["chunk_start_frame"].to_pylist() == [0, 30, 60]
    assert table["chunk_end_frame"].to_pylist() == [30, 60, 60]
    assert table["valid_chunk"].to_pylist() == [True, True, False]
    np.testing.assert_allclose(table["chunk_td_residual"].to_numpy()[:2], [0.0, 0.0])
    assert np.isnan(table["chunk_td_residual"].to_numpy()[2])


def test_value_scores_use_standard_n_step_td_equation(tmp_path):
    source = tmp_path / "scores.parquet"
    output = tmp_path / "chunks.parquet"
    _write_scores(source, [0.0, 0.5, 1.0], column="progress")

    compute_chunk_td_residuals(
        source,
        output,
        score_column="progress",
        score_semantics="value",
        reward_per_second=0.0,
        fps=2.0,
        chunk_size=2,
        include_incomplete=False,
    )
    table = pq.read_table(output)

    assert table["value_start"].to_pylist() == [0.0]
    assert table["value_end"].to_pylist() == [1.0]
    assert table["chunk_td_residual"].to_pylist() == [1.0]


def test_discounted_return_and_output_metadata(tmp_path):
    assert _discounted_chunk_return(
        2,
        fps=2.0,
        gamma=0.5,
        reward_per_second=-1.0,
    ) == pytest.approx(-0.75)

    source = tmp_path / "scores.parquet"
    output = tmp_path / "chunks.parquet"
    _write_scores(source, [4.0, 3.0, 2.0])
    compute_chunk_td_residuals(
        source,
        output,
        score_column="model_score",
        score_semantics="cost_to_go",
        reward_per_second=-1.0,
        fps=2.0,
        chunk_size=2,
        gamma=0.5,
        include_incomplete=False,
    )
    table = pq.read_table(output)

    # return + gamma^N * (-score_end) - (-score_start)
    assert table["chunk_td_residual"].to_pylist() == pytest.approx([-0.75 - 0.25 * 2.0 + 4.0])
    metadata = json.loads(table.schema.metadata[CHUNK_TD_METADATA_KEY])
    assert metadata["score_column"] == "model_score"
    assert metadata["score_semantics"] == "cost_to_go"
    assert metadata["reward_per_second"] == -1.0

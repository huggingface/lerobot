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

"""Tests for lerobot-compute-returns script."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.scripts.lerobot_compute_returns import (
    IS_TERMINAL_COL,
    MC_RETURN_COL,
    ComputeReturnsConfig,
    _get_episode_success,
    compute_episode_returns,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parquet_dataset(tmp_path):
    """Build a minimal parquet shard + info.json for testing I/O logic.

    Mirrors the lerobot-rollout DAgger convention: ``next.success`` is False
    on all frames except the terminal frame of successful episodes.
    Even episodes are successful, odd episodes are failures.
    """
    num_episodes = 3
    frames_per_ep = 10

    root = tmp_path / "test_dataset"
    data_dir = root / "data" / "chunk-000"
    meta_dir = root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    all_rows = []
    episodes_meta = []
    global_idx = 0
    for ep in range(num_episodes):
        ep_from = global_idx
        is_successful = ep % 2 == 0
        for frame in range(frames_per_ep):
            is_last_frame = frame == frames_per_ep - 1
            all_rows.append(
                {
                    "episode_index": ep,
                    "frame_index": frame,
                    "index": global_idx,
                    "next.success": is_successful and is_last_frame,
                }
            )
            global_idx += 1
        ep_to = global_idx
        episodes_meta.append(
            {
                "episode_index": ep,
                "length": frames_per_ep,
                "dataset_from_index": ep_from,
                "dataset_to_index": ep_to,
            }
        )

    table = pa.table(
        {
            "episode_index": [r["episode_index"] for r in all_rows],
            "frame_index": [r["frame_index"] for r in all_rows],
            "index": [r["index"] for r in all_rows],
            "next.success": [r["next.success"] for r in all_rows],
        }
    )

    parquet_path = data_dir / "episode_000000.parquet"
    pq.write_table(table, parquet_path)

    info = {
        "codebase_version": "v3.0",
        "total_episodes": num_episodes,
        "total_frames": global_idx,
        "fps": 30,
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "next.success": {"dtype": "bool", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    return root, parquet_path, episodes_meta


def _rewrite_shard(parquet_path: Path, episodes_meta: list[dict], config: ComputeReturnsConfig):
    """Rewrite a single parquet shard using the core logic from compute_returns."""
    table = pq.read_table(parquet_path)

    if not config.force and IS_TERMINAL_COL in table.column_names:
        return

    all_is_terminal = np.zeros(len(table), dtype=bool)
    all_mc_return = np.zeros(len(table), dtype=np.float32)
    episode_col = table.column("episode_index").to_pylist()

    for ep_info in episodes_meta:
        ep_idx = ep_info["episode_index"]
        ep_len = ep_info["length"]

        mask = np.array([v == ep_idx for v in episode_col], dtype=bool)
        local_indices = np.where(mask)[0]

        ep_subtable = table.filter(mask)
        success = _get_episode_success(ep_subtable, config.success_key, config.default_success)

        is_terminal, mc_return = compute_episode_returns(
            num_frames=ep_len,
            success=success,
            c_fail=config.c_fail,
            gamma=config.gamma,
            max_episode_length=config.max_episode_length or ep_len,
        )

        all_is_terminal[local_indices] = is_terminal
        all_mc_return[local_indices] = mc_return

    if IS_TERMINAL_COL in table.column_names:
        table = table.drop(IS_TERMINAL_COL)
    if MC_RETURN_COL in table.column_names:
        table = table.drop(MC_RETURN_COL)

    table = table.append_column(IS_TERMINAL_COL, pa.array(all_is_terminal))
    table = table.append_column(MC_RETURN_COL, pa.array(all_mc_return))
    pq.write_table(table, parquet_path)


# ---------------------------------------------------------------------------
# Tests: compute_episode_returns (pure math, no I/O)
# ---------------------------------------------------------------------------


def test_successful_episode_terminal_reward_is_zero():
    """Terminal MC return for a successful episode should be 0."""
    _, mc_return = compute_episode_returns(
        num_frames=10, success=True, c_fail=50.0, gamma=1.0, max_episode_length=10
    )
    assert mc_return[-1] == pytest.approx(0.0, abs=1e-6)


def test_failed_episode_terminal_reward_reflects_cfail():
    """Terminal MC return for a failed episode should be -C_fail / H."""
    horizon = 100
    c_fail = 50.0
    _, mc_return = compute_episode_returns(
        num_frames=10, success=False, c_fail=c_fail, gamma=1.0, max_episode_length=horizon
    )
    assert mc_return[-1] == pytest.approx(-c_fail / horizon, abs=1e-5)


def test_is_terminal_only_last_frame():
    """Only the last frame of an episode should be marked terminal."""
    is_terminal, _ = compute_episode_returns(
        num_frames=20, success=True, c_fail=50.0, gamma=1.0, max_episode_length=20
    )
    assert is_terminal[-1] == True  # noqa: E712
    assert not any(is_terminal[:-1])


def test_mc_return_monotonically_increases_for_success():
    """For a successful undiscounted episode, returns should increase toward 0."""
    _, mc_return = compute_episode_returns(
        num_frames=50, success=True, c_fail=50.0, gamma=1.0, max_episode_length=50
    )
    for i in range(len(mc_return) - 1):
        assert mc_return[i] <= mc_return[i + 1]


def test_mc_return_bounded_negative_to_zero():
    """MC returns for successful episodes should be in (-1, 0]."""
    _, mc_return = compute_episode_returns(
        num_frames=100, success=True, c_fail=50.0, gamma=1.0, max_episode_length=100
    )
    assert mc_return[-1] == pytest.approx(0.0, abs=1e-6)
    assert all(v <= 0.0 for v in mc_return)
    assert all(v >= -1.0 - 1e-6 for v in mc_return)


def test_first_frame_return_success():
    """First frame return for successful episode equals -(N-1)/H."""
    num_frames = 10
    horizon = 10
    _, mc_return = compute_episode_returns(
        num_frames=num_frames, success=True, c_fail=50.0, gamma=1.0, max_episode_length=horizon
    )
    expected = -(num_frames - 1) / horizon
    assert mc_return[0] == pytest.approx(expected, abs=1e-5)


def test_first_frame_return_failure():
    """First frame return for failed episode includes the failure penalty."""
    num_frames = 10
    horizon = 100
    c_fail = 50.0
    _, mc_return = compute_episode_returns(
        num_frames=num_frames, success=False, c_fail=c_fail, gamma=1.0, max_episode_length=horizon
    )
    expected = (-(num_frames - 1) / horizon) + (-c_fail / horizon)
    assert mc_return[0] == pytest.approx(expected, abs=1e-5)


def test_discount_factor_less_than_one():
    """Discount factor < 1 should make earlier frames have smaller magnitude."""
    _, mc_undiscounted = compute_episode_returns(
        num_frames=20, success=True, c_fail=50.0, gamma=1.0, max_episode_length=20
    )
    _, mc_discounted = compute_episode_returns(
        num_frames=20, success=True, c_fail=50.0, gamma=0.99, max_episode_length=20
    )
    assert abs(mc_discounted[0]) < abs(mc_undiscounted[0])


def test_single_frame_episode_success():
    """Single-frame successful episode: return should be 0."""
    is_terminal, mc_return = compute_episode_returns(
        num_frames=1, success=True, c_fail=50.0, gamma=1.0, max_episode_length=1
    )
    assert mc_return[0] == pytest.approx(0.0, abs=1e-6)
    assert is_terminal[0] == True  # noqa: E712


def test_single_frame_episode_failure():
    """Single-frame failed episode: return should be -C_fail/H."""
    horizon = 100
    c_fail = 50.0
    is_terminal, mc_return = compute_episode_returns(
        num_frames=1, success=False, c_fail=c_fail, gamma=1.0, max_episode_length=horizon
    )
    assert mc_return[0] == pytest.approx(-c_fail / horizon, abs=1e-5)
    assert is_terminal[0] == True  # noqa: E712


def test_horizon_normalization_scales_returns():
    """Larger horizon should scale down the per-step penalty."""
    _, mc_small_h = compute_episode_returns(
        num_frames=10, success=True, c_fail=50.0, gamma=1.0, max_episode_length=10
    )
    _, mc_large_h = compute_episode_returns(
        num_frames=10, success=True, c_fail=50.0, gamma=1.0, max_episode_length=100
    )
    assert abs(mc_large_h[0]) < abs(mc_small_h[0])


# ---------------------------------------------------------------------------
# Tests: _get_episode_success (in-memory PyArrow tables)
# ---------------------------------------------------------------------------


def test_default_success_overrides_column():
    """default_success should override any column value."""
    table = pa.table({"next.success": [True, True, True]})
    assert _get_episode_success(table, "next.success", default_success=False) is False


def test_reads_bool_column():
    """Should detect success via any() reduction over the column."""
    table_success = pa.table({"next.success": [False, False, True]})
    table_fail = pa.table({"next.success": [False, False, False]})
    assert _get_episode_success(table_success, "next.success", None) is True
    assert _get_episode_success(table_fail, "next.success", None) is False


def test_reads_int_column():
    """Should interpret integer success column (0/1) as bool via any()."""
    table = pa.table({"task_success": [0, 0, 1]})
    assert _get_episode_success(table, "task_success", None) is True


def test_all_zeros_means_failure():
    """An episode with all-zero success values is a failure."""
    table = pa.table({"next.success": [0, 0, 0]})
    assert _get_episode_success(table, "next.success", None) is False


def test_missing_column_defaults_to_true():
    """When success column is missing, assume success (demo data)."""
    table = pa.table({"frame_index": [0, 1, 2]})
    assert _get_episode_success(table, "next.success", None) is True


# ---------------------------------------------------------------------------
# Tests: parquet rewriting (integration, writes to disk)
# ---------------------------------------------------------------------------


def test_writes_columns_to_parquet(parquet_dataset):
    """The rewrite logic should add is_terminal and mc_return columns."""
    root, parquet_path, episodes_meta = parquet_dataset

    table_before = pq.read_table(parquet_path)
    assert IS_TERMINAL_COL not in table_before.column_names
    assert MC_RETURN_COL not in table_before.column_names

    config = ComputeReturnsConfig(success_key="next.success", max_episode_length=10, force=True)
    _rewrite_shard(parquet_path, episodes_meta, config)

    table_after = pq.read_table(parquet_path)
    assert IS_TERMINAL_COL in table_after.column_names
    assert MC_RETURN_COL in table_after.column_names


def test_terminal_frames_correct(parquet_dataset):
    """Only the last frame of each episode should be terminal."""
    root, parquet_path, episodes_meta = parquet_dataset

    config = ComputeReturnsConfig(success_key="next.success", max_episode_length=10, force=True)
    _rewrite_shard(parquet_path, episodes_meta, config)

    table = pq.read_table(parquet_path)
    is_terminal = table.column(IS_TERMINAL_COL).to_pylist()
    terminal_indices = [i for i, v in enumerate(is_terminal) if v]
    assert terminal_indices == [9, 19, 29]


def test_success_episodes_return_zero_at_terminal(tmp_path):
    """Successful episodes (ep 0) should have mc_return=0 at terminal."""
    num_episodes = 2
    frames_per_ep = 5

    root = tmp_path / "test_dataset"
    data_dir = root / "data" / "chunk-000"
    meta_dir = root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    all_rows = []
    episodes_meta = []
    global_idx = 0
    for ep in range(num_episodes):
        ep_from = global_idx
        is_successful = ep % 2 == 0
        for frame in range(frames_per_ep):
            is_last_frame = frame == frames_per_ep - 1
            all_rows.append(
                {
                    "episode_index": ep,
                    "frame_index": frame,
                    "index": global_idx,
                    "next.success": is_successful and is_last_frame,
                }
            )
            global_idx += 1
        episodes_meta.append(
            {
                "episode_index": ep,
                "length": frames_per_ep,
                "dataset_from_index": ep_from,
                "dataset_to_index": global_idx,
            }
        )

    table = pa.table(
        {
            "episode_index": [r["episode_index"] for r in all_rows],
            "frame_index": [r["frame_index"] for r in all_rows],
            "index": [r["index"] for r in all_rows],
            "next.success": [r["next.success"] for r in all_rows],
        }
    )
    parquet_path = data_dir / "episode_000000.parquet"
    pq.write_table(table, parquet_path)

    info = {
        "codebase_version": "v3.0",
        "total_episodes": num_episodes,
        "total_frames": global_idx,
        "fps": 30,
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "next.success": {"dtype": "bool", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    config = ComputeReturnsConfig(success_key="next.success", max_episode_length=5, force=True)
    _rewrite_shard(parquet_path, episodes_meta, config)

    table = pq.read_table(parquet_path)
    mc_return = table.column(MC_RETURN_COL).to_pylist()
    assert mc_return[4] == pytest.approx(0.0, abs=1e-5)


def test_failed_episodes_have_negative_terminal(tmp_path):
    """Failed episodes (ep 1) should have mc_return < 0 at terminal."""
    num_episodes = 2
    frames_per_ep = 5

    root = tmp_path / "test_dataset"
    data_dir = root / "data" / "chunk-000"
    meta_dir = root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    all_rows = []
    episodes_meta = []
    global_idx = 0
    for ep in range(num_episodes):
        ep_from = global_idx
        is_successful = ep % 2 == 0
        for frame in range(frames_per_ep):
            is_last_frame = frame == frames_per_ep - 1
            all_rows.append(
                {
                    "episode_index": ep,
                    "frame_index": frame,
                    "index": global_idx,
                    "next.success": is_successful and is_last_frame,
                }
            )
            global_idx += 1
        episodes_meta.append(
            {
                "episode_index": ep,
                "length": frames_per_ep,
                "dataset_from_index": ep_from,
                "dataset_to_index": global_idx,
            }
        )

    table = pa.table(
        {
            "episode_index": [r["episode_index"] for r in all_rows],
            "frame_index": [r["frame_index"] for r in all_rows],
            "index": [r["index"] for r in all_rows],
            "next.success": [r["next.success"] for r in all_rows],
        }
    )
    parquet_path = data_dir / "episode_000000.parquet"
    pq.write_table(table, parquet_path)

    config = ComputeReturnsConfig(success_key="next.success", max_episode_length=5, c_fail=50.0, force=True)
    _rewrite_shard(parquet_path, episodes_meta, config)

    table = pq.read_table(parquet_path)
    mc_return = table.column(MC_RETURN_COL).to_pylist()
    assert mc_return[9] < 0.0


def test_idempotent_with_force_flag(parquet_dataset):
    """Running twice with force should produce identical results."""
    root, parquet_path, episodes_meta = parquet_dataset

    config = ComputeReturnsConfig(success_key="next.success", max_episode_length=10, force=True)
    _rewrite_shard(parquet_path, episodes_meta, config)
    table1 = pq.read_table(parquet_path)
    mc1 = table1.column(MC_RETURN_COL).to_pylist()

    _rewrite_shard(parquet_path, episodes_meta, config)
    table2 = pq.read_table(parquet_path)
    mc2 = table2.column(MC_RETURN_COL).to_pylist()

    assert mc1 == mc2


def test_skips_if_columns_exist_without_force(parquet_dataset):
    """Without force, existing columns should not be overwritten."""
    root, parquet_path, episodes_meta = parquet_dataset

    config = ComputeReturnsConfig(success_key="next.success", max_episode_length=10, force=True)
    _rewrite_shard(parquet_path, episodes_meta, config)

    table = pq.read_table(parquet_path)
    original_mc = table.column(MC_RETURN_COL).to_pylist()

    config_no_force = ComputeReturnsConfig(success_key="next.success", max_episode_length=20, force=False)
    _rewrite_shard(parquet_path, episodes_meta, config_no_force)

    table2 = pq.read_table(parquet_path)
    assert table2.column(MC_RETURN_COL).to_pylist() == original_mc


def test_updates_info_json(parquet_dataset):
    """info.json should be updated with is_terminal and mc_return features."""
    from lerobot.scripts.lerobot_compute_returns import _update_info_json

    root, parquet_path, episodes_meta = parquet_dataset

    _update_info_json(root, None)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert IS_TERMINAL_COL in info["features"]
    assert MC_RETURN_COL in info["features"]
    assert info["features"][IS_TERMINAL_COL]["dtype"] == "bool"
    assert info["features"][MC_RETURN_COL]["dtype"] == "float32"

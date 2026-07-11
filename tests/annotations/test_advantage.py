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

"""Tests for the advantage scoring annotation module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.annotations.steerable_pipeline.config import AdvantageConfig
from lerobot.annotations.steerable_pipeline.modules.advantage import AdvantageModule
from lerobot.annotations.steerable_pipeline.reader import EpisodeRecord
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging


def _make_record(
    episode_index: int = 0,
    num_frames: int = 20,
    task: str = "pick up the cup",
    mc_returns: np.ndarray | None = None,
    intervention_mask: np.ndarray | None = None,
    fps: float = 10.0,
) -> EpisodeRecord:
    """Build a minimal EpisodeRecord with a mocked frames_df."""
    import pandas as pd

    timestamps = tuple(round(i / fps, 6) for i in range(num_frames))
    frame_indices = tuple(range(num_frames))

    if mc_returns is None:
        mc_returns = np.linspace(-0.9, -0.1, num_frames).astype(np.float32)

    data = {
        "episode_index": [episode_index] * num_frames,
        "frame_index": list(range(num_frames)),
        "timestamp": list(timestamps),
        "mc_return": mc_returns,
    }

    if intervention_mask is not None:
        data["intervention"] = intervention_mask.astype(bool)

    df = pd.DataFrame(data)

    record = EpisodeRecord(
        episode_index=episode_index,
        episode_task=task,
        frame_timestamps=timestamps,
        frame_indices=frame_indices,
        data_path=Path("/fake/data.parquet"),
        row_offset=0,
        row_count=num_frames,
    )
    record._frames_df_cache = df
    return record


@pytest.fixture
def staging(tmp_path: Path) -> EpisodeStaging:
    return EpisodeStaging(tmp_path, episode_index=0)


def test_advantage_module_disabled():
    """Disabled module has enabled=False."""
    cfg = AdvantageConfig(enabled=False)
    module = AdvantageModule(config=cfg)
    assert not module.enabled


def test_advantage_module_enabled_by_default():
    """Module is enabled by default."""
    cfg = AdvantageConfig()
    module = AdvantageModule(config=cfg)
    assert module.enabled


def test_run_episode_skips_without_value_function_path(staging: EpisodeStaging):
    """Module gracefully returns when no value_function_path is configured."""
    cfg = AdvantageConfig(value_function_path="")
    module = AdvantageModule(config=cfg)
    record = _make_record()

    module.run_episode(record, staging)

    rows = staging.read("advantage")
    assert rows == []


def test_binarization_with_mock_values(staging: EpisodeStaging):
    """Advantage binarization produces positive/negative labels based on threshold."""
    num_frames = 10
    mc_returns = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, -0.5, -0.6, -0.7, -0.8, -0.9], dtype=np.float32)
    mock_values = np.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4], dtype=np.float32)

    cfg = AdvantageConfig(
        value_function_path="/fake/vf",
        threshold_percentile=0.5,
    )
    module = AdvantageModule(config=cfg)
    record = _make_record(num_frames=num_frames, mc_returns=mc_returns)

    with (
        patch.object(module, "_ensure_model_loaded"),
        patch.object(module, "_compute_values", return_value=mock_values),
    ):
        module.run_episode(record, staging)

    rows = staging.read("advantage")
    assert len(rows) == num_frames

    # A_t = mc_returns - values
    # advantages = [-0.1, 0.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3, -0.4, -0.5]
    # Median (50th pctile) = -0.1
    # positive: advantage > -0.1 → indices 1,2,3,4
    # negative: advantage <= -0.1 → indices 0,5,6,7,8,9
    positives = [r for r in rows if r["content"] == "positive"]
    negatives = [r for r in rows if r["content"] == "negative"]
    assert len(positives) == 4
    assert len(negatives) == 6


def test_intervention_frames_forced_positive(staging: EpisodeStaging):
    """Intervention frames are always scored as positive regardless of advantage value."""
    num_frames = 5
    mc_returns = np.array([-0.9, -0.9, -0.9, -0.9, -0.9], dtype=np.float32)
    mock_values = np.array([-0.1, -0.1, -0.1, -0.1, -0.1], dtype=np.float32)
    intervention = np.array([False, False, True, False, False])

    cfg = AdvantageConfig(
        value_function_path="/fake/vf",
        force_positive_on_intervention=True,
    )
    module = AdvantageModule(config=cfg)
    record = _make_record(num_frames=num_frames, mc_returns=mc_returns, intervention_mask=intervention)

    with (
        patch.object(module, "_ensure_model_loaded"),
        patch.object(module, "_compute_values", return_value=mock_values),
    ):
        module.run_episode(record, staging)

    rows = staging.read("advantage")
    # Frame 2 (intervention) should be positive despite negative advantage
    assert rows[2]["content"] == "positive"


def test_all_frames_labeled(staging: EpisodeStaging):
    """Every frame gets an advantage label (no annotation-level dropout)."""
    num_frames = 100
    mc_returns = np.linspace(-0.9, -0.1, num_frames).astype(np.float32)
    mock_values = np.full(num_frames, -0.5, dtype=np.float32)

    cfg = AdvantageConfig(value_function_path="/fake/vf")
    module = AdvantageModule(config=cfg)
    record = _make_record(num_frames=num_frames, mc_returns=mc_returns)

    with (
        patch.object(module, "_ensure_model_loaded"),
        patch.object(module, "_compute_values", return_value=mock_values),
    ):
        module.run_episode(record, staging)

    rows = staging.read("advantage")
    assert len(rows) == num_frames


def test_staged_row_format(staging: EpisodeStaging):
    """Staged rows have the correct schema for language_persistent."""
    num_frames = 5
    mc_returns = np.array([-0.5, -0.4, -0.3, -0.2, -0.1], dtype=np.float32)
    mock_values = np.full(5, -0.3, dtype=np.float32)

    cfg = AdvantageConfig(value_function_path="/fake/vf")
    module = AdvantageModule(config=cfg)
    record = _make_record(num_frames=num_frames, mc_returns=mc_returns)

    with (
        patch.object(module, "_ensure_model_loaded"),
        patch.object(module, "_compute_values", return_value=mock_values),
    ):
        module.run_episode(record, staging)

    rows = staging.read("advantage")
    for row in rows:
        assert row["role"] == "user"
        assert row["content"] in ("positive", "negative")
        assert row["style"] == "advantage"
        assert isinstance(row["timestamp"], float)
        assert row["camera"] is None
        assert row["tool_calls"] is None


def test_n_step_advantage():
    """N-step advantage uses partial returns + bootstrapped value."""
    num_frames = 10
    mc_returns = np.linspace(-0.9, 0.0, num_frames).astype(np.float32)
    mock_values = np.full(num_frames, -0.45, dtype=np.float32)

    cfg = AdvantageConfig(
        value_function_path="/fake/vf",
        n_step=3,
    )
    module = AdvantageModule(config=cfg)
    record = _make_record(num_frames=num_frames, mc_returns=mc_returns)

    with patch.object(module, "_ensure_model_loaded"):
        advantages, _ = (
            module.compute_advantages_for_episode.__wrapped__(module, record)
            if hasattr(module.compute_advantages_for_episode, "__wrapped__")
            else (None, None)
        )

    # Just verify computation works - use the internal method directly
    module._model = MagicMock()
    module._preprocessor = MagicMock()
    with patch.object(module, "_compute_values", return_value=mock_values):
        advantages, _ = module.compute_advantages_for_episode(record)

    # For t where t+n < num_frames: A = mc_return[t] - mc_return[t+n] + values[t+n] - values[t]
    # Since values are constant: A = mc_return[t] - mc_return[t+n]
    # For t where t+n >= num_frames: A = mc_return[t] - values[t]
    for t in range(num_frames):
        if t + 3 < num_frames:
            expected = mc_returns[t] - mc_returns[t + 3] + mock_values[t + 3] - mock_values[t]
        else:
            expected = mc_returns[t] - mock_values[t]
        np.testing.assert_almost_equal(advantages[t], expected, decimal=5)


def test_compute_threshold():
    """Threshold is computed as configured percentile of non-intervention advantages."""
    cfg = AdvantageConfig(threshold_percentile=0.3)
    module = AdvantageModule(config=cfg)

    advantages = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    intervention_mask = np.array([False, False, False, False, False])

    threshold = module._compute_threshold(advantages, intervention_mask)
    expected = float(np.percentile(advantages, 30))
    assert abs(threshold - expected) < 1e-6


def test_compute_threshold_excludes_intervention():
    """Threshold computation excludes intervention frames."""
    cfg = AdvantageConfig(threshold_percentile=0.5)
    module = AdvantageModule(config=cfg)

    advantages = np.array([100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
    intervention_mask = np.array([True, False, False, False, True])

    threshold = module._compute_threshold(advantages, intervention_mask)
    # Only non-intervention: [-1.0, 0.0, 1.0], median = 0.0
    expected = float(np.percentile([-1.0, 0.0, 1.0], 50))
    assert abs(threshold - expected) < 1e-6


def test_missing_mc_return_raises():
    """Module raises if mc_return column is missing from dataset."""
    import pandas as pd

    cfg = AdvantageConfig(value_function_path="/fake/vf")
    module = AdvantageModule(config=cfg)
    module._model = MagicMock()
    module._preprocessor = MagicMock()

    record = EpisodeRecord(
        episode_index=0,
        episode_task="test",
        frame_timestamps=(0.0, 0.1),
        frame_indices=(0, 1),
        data_path=Path("/fake/data.parquet"),
        row_offset=0,
        row_count=2,
    )
    record._frames_df_cache = pd.DataFrame({"episode_index": [0, 0], "frame_index": [0, 1]})

    with pytest.raises(KeyError, match="mc_return"):
        module.compute_advantages_for_episode(record)

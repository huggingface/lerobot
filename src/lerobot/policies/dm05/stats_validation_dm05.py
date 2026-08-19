#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

"""Validation helpers for DM05 dataset and processor statistics."""

import shlex
from typing import Any

import numpy as np

from lerobot.configs import NormalizationMode
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_dm05 import DM05Config
from .utils import relative_action_mask

_REQUIRED_STATS = {
    NormalizationMode.MEAN_STD: ("mean", "std"),
    NormalizationMode.MIN_MAX: ("min", "max"),
    NormalizationMode.QUANTILES: ("q01", "q99"),
    NormalizationMode.QUANTILE10: ("q10", "q90"),
}

_RANGE_STATS = {
    NormalizationMode.MIN_MAX: ("min", "max"),
    NormalizationMode.QUANTILES: ("q01", "q99"),
    NormalizationMode.QUANTILE10: ("q10", "q90"),
}


def dm05_feature_stats_complete(
    config: DM05Config,
    stats: dict[str, Any],
    key: str,
    feature_type: str,
) -> bool:
    """Check whether one DM05 feature has the stats required by its norm mode."""
    required = _REQUIRED_STATS.get(config.normalization_mapping.get(feature_type), ())
    features = config.input_features if key == OBS_STATE else config.output_features
    if key not in features:
        return False
    expected_shape = tuple(features[key].shape)
    feature_stats = stats.get(key, {})
    return all(
        name in feature_stats
        and (value := np.asarray(feature_stats[name])).shape == expected_shape
        and np.isfinite(value).all()
        for name in required
    )


def dm05_stats_complete(config: DM05Config, stats: dict[str, Any] | None) -> bool:
    """Return whether state/action stats satisfy the configured normalization modes."""
    stats = stats or {}
    return all(
        dm05_feature_stats_complete(config, stats, key, feature_type)
        for key, feature_type in ((OBS_STATE, "STATE"), (ACTION, "ACTION"))
    )


def validate_dm05_relative_action_stats(config: DM05Config, stats: dict[str, Any] | None) -> None:
    """Require an invertible action scale on dimensions converted to deltas."""
    if not config.use_relative_actions:
        return
    stats = stats or {}
    stat_names = _RANGE_STATS.get(config.normalization_mapping.get("ACTION"))
    if stat_names is None:
        return
    if not dm05_feature_stats_complete(config, stats, ACTION, "ACTION"):
        return

    low = np.asarray(stats[ACTION][stat_names[0]])
    high = np.asarray(stats[ACTION][stat_names[1]])
    mask = np.asarray(
        relative_action_mask(
            int(config.output_features[ACTION].shape[-1]),
            config.action_feature_names,
            config.relative_exclude_joints,
        ),
        dtype=bool,
    )
    invalid = np.flatnonzero(mask & (high <= low))
    if invalid.size:
        raise ValueError(
            "DM05 relative actions require non-degenerate action normalization stats on delta dimensions; "
            f"invalid indices: {invalid.tolist()}. Exclude constant dimensions or regenerate the dataset stats."
        )


def dm05_prepare_stats_command(config: DM05Config, dataset_meta: Any | None = None) -> str:
    """Build the preparation command for a failed DM05 dataset contract."""
    repo_id = getattr(dataset_meta, "repo_id", None) or "DATASET_REPO_ID"
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "lerobot.policies.dm05.prepare_stats_dm05",
        f"--repo-id={repo_id}",
    ]
    if (root := getattr(dataset_meta, "root", None)) is not None:
        command.append(f"--root={root}")
    command.extend(
        [
            f"--chunk-size={config.chunk_size}",
            f"--drop-n-last-frames={config.drop_n_last_frames}",
        ]
    )
    if config.use_relative_actions:
        command.append("--use-relative-actions")
        command.append("--relative-exclude-joints")
        command.extend(config.relative_exclude_joints)
    return shlex.join(command)

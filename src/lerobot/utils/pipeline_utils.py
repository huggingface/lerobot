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

"""
Utilities for building dataset features from robot/teleoperator pipelines and for
checking action/observation space compatibility between teleops and robots.
"""

import logging
import re
from collections.abc import Sequence

from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR

# Prefixes stripped from feature keys to produce clean dataset names.
# Handles both fully-qualified (e.g. "observation.state.ee.x") and short (e.g. "state.ee.x") forms.
_PREFIXES_TO_STRIP = tuple(
    f"{token}."
    for const in (ACTION, OBS_STATE, OBS_IMAGES)
    for token in (const, const.split(".")[-1])
)

_IMAGES_TOKEN = OBS_IMAGES.split(".")[-1]


def _should_keep(key: str, patterns: Sequence[str] | None) -> bool:
    if patterns is None:
        return True
    return any(re.search(pat, key) for pat in patterns)


def _strip_prefix(key: str) -> str:
    for prefix in _PREFIXES_TO_STRIP:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _features_to_dataset_spec(
    features: dict,
    *,
    is_action: bool,
    use_videos: bool,
    patterns: Sequence[str] | None = None,
) -> dict:
    """
    Convert a flat feature dict (as returned by ``robot.observation_features`` or
    ``teleop.action_features``) into a LeRobot dataset feature specification.

    Args:
        features: Flat dict mapping feature key → type or shape.
        is_action: True when ``features`` describes actions; False for observations.
        use_videos: When False, image observation features are excluded entirely.
        patterns: Optional regex patterns to filter state/action features.
                  Image features are not affected by this filter.

    Returns:
        A dict suitable for passing to ``LeRobotDataset.create(..., features=...)``.
    """
    categorized: dict = {}
    for key, value in features.items():
        is_image = not is_action and (
            (isinstance(value, tuple) and len(value) == 3)
            or key.startswith(f"{OBS_IMAGES}.")
            or key.startswith(f"{_IMAGES_TOKEN}.")
            or f".{_IMAGES_TOKEN}." in key
        )

        if is_image and not use_videos:
            continue
        if not is_image and not _should_keep(key, patterns):
            continue

        categorized[_strip_prefix(key)] = value

    if not categorized:
        return {}

    prefix = ACTION if is_action else OBS_STR
    return hw_to_dataset_features(categorized, prefix, use_videos)


def build_dataset_features(
    robot,
    teleop=None,
    *,
    use_videos: bool = True,
    action_features: dict | None = None,
) -> dict:
    """
    Derive dataset feature specifications from robot and teleoperator pipelines.

    Reads ``robot.observation_features`` (which already reflects the robot's output
    pipeline transformation) and, when provided, ``teleop.action_features`` or an
    explicit ``action_features`` dict to determine what the dataset will store.

    This replaces the old pattern of manually calling ``aggregate_pipeline_dataset_features``
    with explicit processor objects.

    Args:
        robot: The robot instance (must have ``observation_features``).
        teleop: The teleoperator instance. When ``None`` and ``action_features`` is also
            ``None`` (policy-only recording), only observation features are returned.
        use_videos: If True, image observations are included as video features.
        action_features: Explicit action feature dict, used when no teleop is available
            (e.g. evaluate/inference mode) but the dataset must match a specific action
            space (e.g. EE coordinates from a previously recorded dataset).

    Returns:
        A combined feature dict suitable for passing to ``LeRobotDataset.create(..., features=...)``.

    Example::

        # Teleop recording
        features = build_dataset_features(follower, leader, use_videos=True)

        # Policy-only recording (no teleop)
        features = build_dataset_features(robot, use_videos=True)

        # Evaluate with explicit EE action space
        features = build_dataset_features(
            robot,
            use_videos=True,
            action_features={
                f"ee.{k}": PolicyFeature(type=FeatureType.ACTION, shape=(1,))
                for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]
            },
        )
    """
    obs_ds = _features_to_dataset_spec(robot.observation_features, is_action=False, use_videos=use_videos)

    if action_features is not None:
        act_ds = _features_to_dataset_spec(action_features, is_action=True, use_videos=False)
    elif teleop is not None:
        act_ds = _features_to_dataset_spec(teleop.action_features, is_action=True, use_videos=False)
    else:
        return obs_ds

    return combine_feature_dicts(act_ds, obs_ds)


def check_action_space_compatibility(teleop, robot) -> None:
    """
    Warn if the teleoperator's pipeline-transformed action features don't match the robot's
    declared ``action_features``.

    This is a soft check — a mismatch produces a warning but does not raise. It is intended
    to catch obvious misconfigurations (e.g., sending EE actions to a robot expecting joints)
    before the control loop starts.

    Args:
        teleop: The teleoperator whose ``action_features`` describe what it sends.
        robot: The robot whose ``action_features`` describe what it expects.
    """
    teleop_out = set(teleop.action_features.keys())
    robot_in = set(robot.action_features.keys())
    if teleop_out != robot_in:
        import warnings

        warnings.warn(
            f"Action space mismatch between teleop and robot.\n"
            f"  Teleop sends: {sorted(teleop_out)}\n"
            f"  Robot expects: {sorted(robot_in)}\n"
            "Ensure pipelines map between these spaces correctly.",
            UserWarning,
            stacklevel=2,
        )
    else:
        logging.debug("Action space compatibility check passed.")


def check_observation_space_compatibility(robot, teleop) -> None:
    """
    Warn if the robot's observation features don't cover what the teleoperator's
    ``feedback_features`` expects.

    A non-empty ``feedback_features`` that is not a subset of the robot's observation keys
    will produce a warning.

    Args:
        robot: The robot whose ``observation_features`` describe what it produces.
        teleop: The teleoperator whose ``feedback_features`` describe what it expects as feedback.
    """
    robot_obs = set(robot.observation_features.keys())
    teleop_feedback = set(teleop.feedback_features.keys())
    if teleop_feedback and not teleop_feedback.issubset(robot_obs):
        import warnings

        warnings.warn(
            f"Observation/feedback space mismatch.\n"
            f"  Robot obs: {sorted(robot_obs)}\n"
            f"  Teleop feedback expects: {sorted(teleop_feedback)}\n"
            "Ensure the robot observation pipeline covers all feedback keys.",
            UserWarning,
            stacklevel=2,
        )
    else:
        logging.debug("Observation/feedback space compatibility check passed.")

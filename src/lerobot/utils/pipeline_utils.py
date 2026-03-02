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

from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts


def build_dataset_features(
    robot,
    teleop=None,
    *,
    use_videos: bool = True,
) -> dict:
    """
    Derive dataset feature specifications from robot and teleoperator pipelines.

    Uses the robot's ``output_pipeline`` and ``raw_observation_features`` to determine
    what the dataset will store as observations, and (when provided) the teleoperator's
    ``output_pipeline`` and ``raw_action_features`` to determine what will be stored as actions.

    This replaces the old pattern of manually calling ``aggregate_pipeline_dataset_features``
    with explicit processor objects.

    Args:
        robot: The robot instance (must have ``output_pipeline()`` and ``raw_observation_features``).
        teleop: The teleoperator instance. When ``None`` (policy-only recording), only observation
            features are returned.
        use_videos: If True, image observations are included as video features.

    Returns:
        A combined feature dict suitable for passing to ``LeRobotDataset.create(..., features=...)``.

    Example::

        # Teleop recording
        features = build_dataset_features(follower, leader, use_videos=True)

        # Policy-only recording (no teleop)
        features = build_dataset_features(robot, use_videos=True)
    """
    obs_features = aggregate_pipeline_dataset_features(
        pipeline=robot.output_pipeline(),
        initial_features=create_initial_features(observation=robot.raw_observation_features),
        use_videos=use_videos,
    )
    if teleop is None:
        return obs_features
    action_features = aggregate_pipeline_dataset_features(
        pipeline=teleop.output_pipeline(),
        initial_features=create_initial_features(action=teleop.raw_action_features),
        use_videos=False,
    )
    return combine_feature_dicts(action_features, obs_features)


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

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
Unit tests for the robot/teleoperator pipeline interface.

Tests cover:
- Default (identity) pipeline behaviour
- Custom pipeline attachment via set_output_pipeline / set_input_pipeline
- Auto-derived observation_features / action_features via pipelines
- Compatibility checks
- build_dataset_features utility
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lerobot.configs.types import PipelineFeatureType
from lerobot.processor import RobotAction, RobotObservation
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.processor.factory import (
    _make_identity_feedback_pipeline,
    _make_identity_observation_pipeline,
    _make_identity_robot_action_pipeline,
    _make_identity_teleop_action_pipeline,
)
from lerobot.processor.pipeline import (
    IdentityProcessorStep,
    ObservationProcessorStep,
    RobotActionProcessorStep,
    RobotProcessorPipeline,
)
from lerobot.robots.robot import Robot
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.pipeline_utils import (
    build_dataset_features,
    check_action_space_compatibility,
    check_observation_space_compatibility,
)


# ─── Mock hardware classes ────────────────────────────────────────────────────


@dataclass
class MockRobotConfig:
    id: str = "mock_robot"
    calibration_dir: Path | None = None


@dataclass
class MockTeleopConfig:
    id: str = "mock_teleop"
    calibration_dir: Path | None = None


_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_JOINT_FEATURES = {f"{j}.pos": float for j in _JOINT_NAMES}
_EE_FEATURES = {"ee.x": float, "ee.y": float, "ee.z": float, "ee.wx": float, "ee.wy": float, "ee.wz": float, "ee.gripper_vel": float}


class MockRobot(Robot):
    """Minimal Robot that stores last action for assertion."""

    config_class = MockRobotConfig
    name = "mock_robot"

    def __init__(self):
        # bypass filesystem calibration setup; initialize with identity pipelines directly
        self._output_pipeline = _make_identity_observation_pipeline()
        self._input_pipeline = _make_identity_robot_action_pipeline()
        self._last_raw_obs: RobotObservation = {}
        self._last_sent: RobotAction = {}

    @property
    def raw_observation_features(self) -> dict:
        return {**_JOINT_FEATURES, "camera": (480, 640, 3)}

    @property
    def action_features(self) -> dict:
        return _JOINT_FEATURES

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate=True):
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def _get_observation(self) -> RobotObservation:
        return {f"{j}.pos": float(i) for i, j in enumerate(_JOINT_NAMES)} | {"camera": None}

    def _send_action(self, action: RobotAction) -> RobotAction:
        self._last_sent = action
        return action

    def disconnect(self):
        pass


class MockTeleop(Teleoperator):
    """Minimal Teleoperator."""

    config_class = MockTeleopConfig
    name = "mock_teleop"

    def __init__(self):
        # bypass filesystem calibration setup; initialize with identity pipelines directly
        self._output_pipeline = _make_identity_teleop_action_pipeline()
        self._input_pipeline = _make_identity_feedback_pipeline()

    @property
    def raw_action_features(self) -> dict:
        return _JOINT_FEATURES

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate=True):
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def _get_action(self) -> RobotAction:
        return {f"{j}.pos": float(i) for i, j in enumerate(_JOINT_NAMES)}

    def _send_feedback(self, feedback):
        pass

    def disconnect(self):
        pass


# ─── Simple transform step (doubles all float values) ────────────────────────


class DoubleActionStep(RobotActionProcessorStep):
    """Doubles all float action values."""

    def action(self, action: RobotAction) -> RobotAction:
        return {k: v * 2 for k, v in action.items()}

    def transform_features(self, features):
        return features


class RenameToEEObsStep(ObservationProcessorStep):
    """Renames joint obs keys to EE-like keys for testing transform_features."""

    def observation(self, obs: RobotObservation) -> RobotObservation:
        return {f"ee.{i}": v for i, v in enumerate(obs.values()) if isinstance(v, float)}

    def transform_features(self, features):
        obs = features.get(PipelineFeatureType.OBSERVATION, {})
        new_obs = {f"ee.{i}": float for i in range(len([v for v in obs.values() if v == float]))}
        return {**features, PipelineFeatureType.OBSERVATION: new_obs}


# ─── Tests: Robot pipeline interface ─────────────────────────────────────────


def test_robot_default_pipeline_is_identity():
    """With no custom pipeline, get_observation returns the same as _get_observation."""
    robot = MockRobot()
    raw = robot._get_observation()
    obs = robot.get_observation()
    assert obs == raw


def test_robot_observation_caches_last_raw():
    """get_observation caches raw result for IK use in send_action."""
    robot = MockRobot()
    robot.get_observation()
    assert robot._last_raw_obs is not None
    assert "shoulder_pan.pos" in robot._last_raw_obs


def test_robot_default_send_action_is_identity():
    """With no custom pipeline, send_action passes action unchanged to _send_action."""
    robot = MockRobot()
    robot.get_observation()  # populate _last_raw_obs
    action = {f"{j}.pos": 1.0 for j in _JOINT_NAMES}
    sent = robot.send_action(action)
    assert sent == action
    assert robot._last_sent == action


def test_robot_custom_output_pipeline_applied():
    """A custom action pipeline is applied to the action before _send_action."""
    robot = MockRobot()
    double_pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[DoubleActionStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot.set_input_pipeline(double_pipeline)
    robot.get_observation()  # populate _last_raw_obs
    action = {f"{j}.pos": 1.0 for j in _JOINT_NAMES}
    robot.send_action(action)
    assert all(v == 2.0 for v in robot._last_sent.values())


def test_robot_observation_features_identity_matches_raw():
    """observation_features equals raw_observation_features with identity pipeline."""
    robot = MockRobot()
    assert robot.observation_features == robot.raw_observation_features


def test_robot_raw_observation_features_unchanged_after_pipeline():
    """raw_observation_features is unaffected by the output pipeline."""
    robot = MockRobot()
    # Even with an FK-like renaming pipeline, raw_observation_features stays the same
    transform_pipeline = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[RenameToEEObsStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    robot.set_output_pipeline(transform_pipeline)
    # raw should still be joints + camera
    raw = robot.raw_observation_features
    assert "shoulder_pan.pos" in raw
    assert "camera" in raw


def test_robot_set_output_pipeline_replaces_identity():
    """set_output_pipeline replaces the default identity."""
    robot = MockRobot()
    p = _make_identity_observation_pipeline()
    robot.set_output_pipeline(p)
    assert robot._output_pipeline is p


def test_robot_set_input_pipeline_replaces_identity():
    robot = MockRobot()
    p = _make_identity_robot_action_pipeline()
    robot.set_input_pipeline(p)
    assert robot._input_pipeline is p


# ─── Tests: Teleoperator pipeline interface ───────────────────────────────────


def test_teleop_default_get_action_is_identity():
    """With no custom pipeline, get_action returns the same as _get_action."""
    teleop = MockTeleop()
    raw = teleop._get_action()
    action = teleop.get_action()
    assert action == raw


def test_teleop_action_features_identity_matches_raw():
    """action_features equals raw_action_features with identity pipeline."""
    teleop = MockTeleop()
    assert teleop.action_features == teleop.raw_action_features


def test_teleop_set_output_pipeline():
    teleop = MockTeleop()
    p = _make_identity_teleop_action_pipeline()
    teleop.set_output_pipeline(p)
    assert teleop._output_pipeline is p


def test_teleop_send_feedback_calls_send_feedback_impl():
    """send_feedback applies identity pipeline and delegates to _send_feedback."""
    teleop = MockTeleop()
    received = {}

    def capture(fb):
        received.update(fb)

    teleop._send_feedback = capture
    teleop.send_feedback({"key": 1.0})
    assert received == {"key": 1.0}


# ─── Tests: Compatibility checks ─────────────────────────────────────────────


def test_check_action_space_compatibility_matching():
    """No warning when teleop output and robot action features match."""
    teleop = MockTeleop()
    robot = MockRobot()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_action_space_compatibility(teleop, robot)  # should not warn


def test_check_action_space_compatibility_mismatch_warns():
    """Warning issued when teleop and robot action features differ."""

    class EETeleop(MockTeleop):
        @property
        def raw_action_features(self):
            return _EE_FEATURES

    teleop = EETeleop()
    robot = MockRobot()  # still returns joint features
    with pytest.warns(UserWarning, match="Action space mismatch"):
        check_action_space_compatibility(teleop, robot)


def test_check_observation_space_compatibility_no_feedback():
    """No warning when teleop has empty feedback_features."""
    robot = MockRobot()
    teleop = MockTeleop()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_observation_space_compatibility(robot, teleop)  # empty feedback → no warning


# ─── Tests: build_dataset_features ───────────────────────────────────────────


def test_build_dataset_features_identity():
    """With identity pipelines, dataset features contain joint keys."""
    robot = MockRobot()
    teleop = MockTeleop()
    features = build_dataset_features(robot, teleop, use_videos=False)
    # Should contain action features (joint names)
    action_keys = {k for k in features if "action" in k or any(j in k for j in _JOINT_NAMES)}
    assert len(action_keys) > 0


def test_build_dataset_features_includes_images_when_use_videos_true():
    """Image features are included when use_videos=True."""
    robot = MockRobot()
    teleop = MockTeleop()
    feats_with = build_dataset_features(robot, teleop, use_videos=True)
    feats_without = build_dataset_features(robot, teleop, use_videos=False)
    # With videos should have more features (camera)
    assert len(feats_with) >= len(feats_without)


# ─── Tests: Factory identity pipeline helpers ─────────────────────────────────


def test_make_identity_observation_pipeline_is_noop():
    pipeline = _make_identity_observation_pipeline()
    obs = {"shoulder_pan.pos": 1.0, "camera": None}
    result = pipeline(obs)
    assert result == obs


def test_make_identity_robot_action_pipeline_is_noop():
    pipeline = _make_identity_robot_action_pipeline()
    action = {"shoulder_pan.pos": 1.0}
    obs = {"shoulder_pan.pos": 0.0}
    result = pipeline((action, obs))
    assert result == action


def test_make_identity_teleop_action_pipeline_is_noop():
    pipeline = _make_identity_teleop_action_pipeline()
    action = {"shoulder_pan.pos": 1.0}
    result = pipeline(action)
    assert result == action

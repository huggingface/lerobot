#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.types import FeatureType
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.teleoperators.keyboard import KeyboardTeleopConfig
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig

_SO_OBSERVATION = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 10.0,
    "elbow_flex.pos": -20.0,
    "wrist_flex.pos": 30.0,
    "wrist_roll.pos": -40.0,
    "gripper.pos": 50.0,
}


class _FakeSOFollower:
    name = "so_follower"
    action_features = {key: {"type": FeatureType.ACTION, "shape": (1,)} for key in _SO_OBSERVATION}

    def __init__(self):
        self.sent_actions = []
        self._is_connected = False

    @property
    def is_connected(self):
        return self._is_connected

    def connect(self):
        self._is_connected = True

    def get_observation(self):
        return _SO_OBSERVATION.copy()

    def send_action(self, action):
        self.sent_actions.append(action)
        return action

    def disconnect(self):
        self._is_connected = False


class _FakeKeyboardTeleop:
    def __init__(self):
        self._is_connected = False

    @property
    def is_connected(self):
        return self._is_connected

    def connect(self):
        self._is_connected = True

    def get_action(self):
        return {}

    def send_feedback(self, feedback):
        pass

    def disconnect(self):
        self._is_connected = False


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)


def test_teleoperate_keyboard_so_follower_uses_joint_processor():
    robot = _FakeSOFollower()
    teleop_device = _FakeKeyboardTeleop()
    cfg = TeleoperateConfig(
        robot=SO101FollowerConfig(port="/dev/null"),
        teleop=KeyboardTeleopConfig(),
        fps=1000,
        teleop_time_s=0.001,
    )

    with (
        patch("lerobot.scripts.lerobot_teleoperate.make_robot_from_config", return_value=robot),
        patch(
            "lerobot.scripts.lerobot_teleoperate.make_teleoperator_from_config", return_value=teleop_device
        ),
    ):
        teleoperate(cfg)

    assert robot.sent_actions
    assert robot.sent_actions[0] == _SO_OBSERVATION
    assert set(robot.sent_actions[0]) == set(_SO_OBSERVATION)


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_and_replay(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)

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

"""Tests for shared smooth-handover helpers used by teleoperate / record / replay."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.common.control_utils import (
    _joint_pos_from_observation,
    follower_smooth_move_to,
    smooth_follower_to_action,
    smooth_teleop_session_start,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)


def test_joint_pos_from_observation_prefers_action_keys():
    obs = {"shoulder.pos": 1.0, "camera": "img", "other": 3}
    keys = {"shoulder.pos": float, "elbow.pos": float}
    assert _joint_pos_from_observation(obs, keys) == {"shoulder.pos": 1.0}


def test_joint_pos_from_observation_fallback_dot_pos():
    obs = {"a.pos": 0.1, "b.pos": 0.2, "cam": object()}
    assert _joint_pos_from_observation(obs, None) == {"a.pos": 0.1, "b.pos": 0.2}


def test_teleop_supports_feedback_requires_torque_apis():
    good = SimpleNamespace(
        feedback_features={"m.pos": float},
        enable_torque=lambda: None,
        disable_torque=lambda: None,
    )
    assert teleop_supports_feedback(good) is True

    no_torque = SimpleNamespace(feedback_features={"m.pos": float})
    assert teleop_supports_feedback(no_torque) is False

    empty = SimpleNamespace(feedback_features={}, enable_torque=lambda: None, disable_torque=lambda: None)
    assert teleop_supports_feedback(empty) is False


def test_follower_smooth_move_to_interpolates_and_sends(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("lerobot.common.control_utils.time.sleep", lambda s: sleeps.append(s))

    robot = MagicMock()
    current = {"j.pos": 0.0}
    target = {"j.pos": 10.0}
    follower_smooth_move_to(robot, current, target, duration_s=0.1, fps=10)

    assert robot.send_action.call_count == 2  # steps = max(int(0.1*10),1)=1 → range(2)
    first = robot.send_action.call_args_list[0].args[0]
    last = robot.send_action.call_args_list[-1].args[0]
    assert first["j.pos"] == pytest.approx(0.0)
    assert last["j.pos"] == pytest.approx(10.0)
    assert len(sleeps) == 2


def test_teleop_smooth_move_to_enables_torque(monkeypatch):
    monkeypatch.setattr("lerobot.common.control_utils.time.sleep", lambda s: None)
    teleop = MagicMock()
    teleop.get_action.return_value = {"j.pos": 0.0}
    teleop_smooth_move_to(teleop, {"j.pos": 5.0}, duration_s=0.05, fps=20)
    teleop.enable_torque.assert_called_once()
    assert teleop.send_feedback.call_count >= 2
    assert teleop.send_feedback.call_args_list[-1].args[0]["j.pos"] == pytest.approx(5.0)


def test_smooth_teleop_session_start_actuated_moves_leader(monkeypatch):
    monkeypatch.setattr("lerobot.common.control_utils.time.sleep", lambda s: None)
    robot = MagicMock()
    robot.get_observation.return_value = {"m1.pos": 3.0, "m2.pos": 4.0}
    robot.action_features = {"m1.pos": float, "m2.pos": float}

    teleop = MagicMock()
    teleop.feedback_features = {"m1.pos": float, "m2.pos": float}
    teleop.enable_torque = MagicMock()
    teleop.disable_torque = MagicMock()
    teleop.get_action.return_value = {"m1.pos": 0.0, "m2.pos": 0.0}

    smooth_teleop_session_start(robot, teleop, duration_s=0.1, fps=10)
    teleop.enable_torque.assert_called()
    assert teleop.send_feedback.called
    final = teleop.send_feedback.call_args_list[-1].args[0]
    assert final["m1.pos"] == pytest.approx(3.0)
    assert final["m2.pos"] == pytest.approx(4.0)
    robot.send_action.assert_not_called()


def test_smooth_teleop_session_start_non_actuated_moves_follower(monkeypatch):
    monkeypatch.setattr("lerobot.common.control_utils.time.sleep", lambda s: None)
    robot = MagicMock()
    robot.get_observation.return_value = {"m1.pos": 0.0}
    robot.action_features = {"m1.pos": float}

    teleop = MagicMock()
    teleop.feedback_features = {}  # non-actuated
    teleop.get_action.return_value = {"m1.pos": 9.0}

    def identity_processor(pair):
        action, _obs = pair
        return action

    smooth_teleop_session_start(
        robot, teleop, robot_action_processor=identity_processor, duration_s=0.1, fps=10
    )
    assert robot.send_action.called
    assert robot.send_action.call_args_list[-1].args[0]["m1.pos"] == pytest.approx(9.0)
    teleop.enable_torque.assert_not_called()


def test_smooth_teleop_session_start_duration_zero_is_noop(monkeypatch):
    robot = MagicMock()
    teleop = MagicMock()
    smooth_teleop_session_start(robot, teleop, duration_s=0.0)
    robot.get_observation.assert_not_called()


def test_smooth_follower_to_action(monkeypatch):
    monkeypatch.setattr("lerobot.common.control_utils.time.sleep", lambda s: None)
    robot = MagicMock()
    robot.get_observation.return_value = {"j.pos": 1.0, "cam": "x"}
    robot.action_features = {"j.pos": float}
    smooth_follower_to_action(robot, {"j.pos": 5.0}, duration_s=0.1, fps=10)
    assert robot.send_action.call_args_list[-1].args[0]["j.pos"] == pytest.approx(5.0)

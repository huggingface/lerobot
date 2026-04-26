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

from unittest.mock import MagicMock

from lerobot.robots.nero_follower.config_nero_follower import NEOKeyboardEEConfig
from lerobot.robots.nero_follower.config_nero_follower import NEOFollowerRobotConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, _build_processors
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig


def test_build_processors_uses_nero_keyboard_processor(monkeypatch):
    sentinel = MagicMock(name="nero_robot_action_processor")

    monkeypatch.setattr(
        "lerobot.robots.nero_follower.make_nero_keyboard_ee_robot_action_processor",
        lambda _cfg: sentinel,
    )

    cfg = TeleoperateConfig(
        teleop=KeyboardEndEffectorTeleopConfig(),
        robot=NEOFollowerRobotConfig(cameras={}),
    )

    _, robot_action_processor, _ = _build_processors(cfg)
    assert robot_action_processor is sentinel


def test_build_processors_keeps_default_for_non_nero_keyboard_ee():
    cfg = TeleoperateConfig(
        teleop=KeyboardEndEffectorTeleopConfig(),
        robot=NEOFollowerRobotConfig(cameras={}, keyboard_ee=NEOKeyboardEEConfig(enabled=False)),
    )

    _, robot_action_processor, _ = _build_processors(cfg)
    assert robot_action_processor is not None

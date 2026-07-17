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

from unittest.mock import MagicMock, call, patch

import pytest

from lerobot.robots import RobotConfig
from lerobot.scripts.lerobot_setup_motors import SetupConfig, main, setup_motors
from lerobot.teleoperators import TeleoperatorConfig


@pytest.mark.parametrize(
    ("field", "config_class", "factory_name"),
    [
        ("robot", RobotConfig, "make_robot_from_config"),
        ("teleop", TeleoperatorConfig, "make_teleoperator_from_config"),
    ],
)
def test_setup_motors_accepts_registered_device_types(field, config_class, factory_name):
    config = MagicMock(spec=config_class)
    config.type = "third_party_device"
    device = MagicMock()

    with patch(f"lerobot.scripts.lerobot_setup_motors.{factory_name}", return_value=device):
        setup_motors(SetupConfig(**{field: config}))

    device.setup_motors.assert_called_once_with()


def test_setup_motors_reports_devices_without_setup_support():
    config = MagicMock(spec=RobotConfig)
    config.type = "camera_only_robot"
    device = MagicMock(spec=[])

    with patch("lerobot.scripts.lerobot_setup_motors.make_robot_from_config", return_value=device):
        with pytest.raises(NotImplementedError, match="camera_only_robot"):
            setup_motors(SetupConfig(robot=config))


def test_main_registers_third_party_plugins_before_parsing():
    manager = MagicMock()
    with (
        patch(
            "lerobot.scripts.lerobot_setup_motors.register_third_party_plugins",
            side_effect=lambda: manager.mock_calls.append(call.register()),
        ),
        patch(
            "lerobot.scripts.lerobot_setup_motors.setup_motors",
            side_effect=lambda: manager.mock_calls.append(call.parse()),
        ),
    ):
        main()

    assert manager.mock_calls == [call.register(), call.parse()]

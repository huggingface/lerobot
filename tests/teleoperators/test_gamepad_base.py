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

import math
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import draccus
import pytest

from lerobot.common.teleop_setup import make_mobile_teleop
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import BaseActionProcessorStep, RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.processor.gamepad_base_processor import GamepadToBaseActionProcessorStep
from lerobot.processor.keyboard_base_processor import KeyboardToBaseActionProcessorStep
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig
from lerobot.utils.errors import DeviceNotConnectedError
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


@pytest.mark.parametrize(
    "axes,expected",
    [
        ((0.5, -0.25, -0.5), (0.1, -0.05, -30.0)),
        ((1.0, 1.0, 2.0), (0.2 / math.sqrt(2), 0.2 / math.sqrt(2), 60.0)),
        ((float("nan"), 1.0, 1.0), (0.0, 0.0, 0.0)),
    ],
)
def test_mapping_bounds_motion_and_preserves_arm_commands(axes, expected):
    mapping = GamepadToBaseActionProcessorStep(linear_speed=0.2, angular_speed=60.0)
    action = dict(zip(("delta_x", "delta_y", "delta_z"), axes, strict=True))
    action["arm_joint.pos"] = 12.0
    assert mapping.action(action) == pytest.approx(
        dict(zip(("x.vel", "y.vel", "theta.vel", "arm_joint.pos"), (*expected, 12.0), strict=True))
    )


@pytest.mark.parametrize("kwargs", [{"linear_speed": -0.1}, {"angular_speed": float("inf")}])
def test_invalid_speed_is_rejected(kwargs):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        GamepadToBaseActionProcessorStep(**kwargs)


def test_gamepad_and_leader_commands_are_combined_and_sticks_return_to_neutral(tmp_path):
    base = GamepadTeleop(GamepadTeleopConfig(use_gripper=False, calibration_dir=tmp_path))
    backend = MagicMock(running=True)
    backend.get_deltas.side_effect = [(1.0, 0.0, -0.5), (0.0, 0.0, 0.0)]
    arm = MockTeleop(
        MockTeleopConfig(n_motors=1, random_values=False, static_values=[12.0], calibration_dir=tmp_path)
    )
    arm.motors = ["joint"]
    robot = SimpleNamespace(arm_action_prefix="arm_", action_features={"arm_joint.pos": float})
    teleop, pipeline = make_mobile_teleop(
        robot, base, arm, base_mapping=GamepadToBaseActionProcessorStep(linear_speed=0.2)
    )
    with patch(
        "lerobot.teleoperators.gamepad.gamepad_utils.GamepadController", return_value=backend
    ) as controller:
        try:
            teleop.connect()
            controller.assert_called_once_with(right_stick_axis="x")
            assert set(teleop.action_features) == {"delta_x", "delta_y", "delta_z", "arm_joint.pos"}
            assert pipeline((teleop.get_action(), {"arm_joint.pos": 8.0})) == {
                "x.vel": 0.2,
                "y.vel": 0.0,
                "theta.vel": -15.0,
                "arm_joint.pos": 12.0,
            }
            assert pipeline((teleop.get_action(), {"arm_joint.pos": 5.0})) == {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
                "arm_joint.pos": 12.0,
            }
        finally:
            teleop.disconnect()
    backend.stop.assert_called_once()


def test_missing_controller_fails_connection_and_releases_backend(tmp_path):
    base = GamepadTeleop(GamepadTeleopConfig(calibration_dir=tmp_path))
    backend = MagicMock(running=False)
    with (
        patch("lerobot.teleoperators.gamepad.gamepad_utils.GamepadController", return_value=backend),
        pytest.raises(DeviceNotConnectedError, match="No gamepad"),
    ):
        base.connect()
    assert not base.is_connected
    backend.stop.assert_called_once()


def test_gamepad_rejects_incompatible_base_configuration(tmp_path):
    base = GamepadTeleop(GamepadTeleopConfig(calibration_dir=tmp_path))
    assert base.action_features["gripper"] is int
    assert base.right_stick_axis == "y"
    with pytest.raises(ValueError, match="use_gripper=false"):
        make_mobile_teleop(SimpleNamespace(action_features={}), base)
    base.config.use_gripper = False
    with pytest.raises(ValueError, match="must match"):
        make_mobile_teleop(
            SimpleNamespace(action_features={}), base, base_mapping=KeyboardToBaseActionProcessorStep()
        )


def test_saved_pipeline_replaces_delta_features_and_restores_speed_settings(tmp_path):
    base = GamepadTeleop(GamepadTeleopConfig(use_gripper=False, calibration_dir=tmp_path))
    _, pipeline = make_mobile_teleop(
        SimpleNamespace(action_features={"joint.pos": float}),
        base,
        base_mapping=GamepadToBaseActionProcessorStep(linear_speed=0.2, angular_speed=60.0),
    )
    feature = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
    features = pipeline.transform_features(
        {PipelineFeatureType.ACTION: dict.fromkeys(("joint.pos", "delta_x", "delta_y", "delta_z"), feature)}
    )
    assert list(features[PipelineFeatureType.ACTION]) == ["joint.pos", "x.vel", "y.vel", "theta.vel"]
    pipeline.save_pretrained(tmp_path, config_filename="processor.json")
    restored = RobotProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="processor.json",
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    sample = ({"delta_x": 0.5, "delta_y": 0.0, "delta_z": -0.5}, {"joint.pos": 8.0})
    expected = {"x.vel": 0.1, "y.vel": 0.0, "theta.vel": -30.0, "joint.pos": 8.0}
    assert restored(sample) == pipeline(sample) == expected


@pytest.mark.parametrize(
    "kind,options,action,speed",
    [
        ("keyboard", [], {"w": None}, 0.1),
        ("gamepad", [], {"delta_x": 1.0, "delta_y": 0.0, "delta_z": 0.0}, 0.1),
        (
            "gamepad",
            ["--base_mapping.type=gamepad", "--base_mapping.linear_speed=0.2"],
            {"delta_x": 1.0, "delta_y": 0.0, "delta_z": 0.0},
            0.2,
        ),
    ],
)
def test_shared_mapping_cli_uses_device_defaults_or_explicit_settings(kind, options, action, speed):
    @dataclass
    class Config:
        base_teleop: TeleoperatorConfig
        base_mapping: BaseActionProcessorStep | None = None

    args = [f"--base_teleop.type={kind}", *options]
    if kind == "gamepad":
        args.append("--base_teleop.use_gripper=false")
    cfg = draccus.parse(Config, args=args)
    with patch("lerobot.teleoperators.keyboard.teleop_keyboard.require_package"):
        base = make_teleoperator_from_config(cfg.base_teleop)
    _, pipeline = make_mobile_teleop(SimpleNamespace(action_features={}), base, base_mapping=cfg.base_mapping)
    assert pipeline((action, {})) == {"x.vel": speed, "y.vel": 0.0, "theta.vel": 0.0}


@pytest.mark.parametrize("axis,expected", [("x", -0.5), ("y", 0.25)])
def test_gamepad_backend_selects_the_requested_right_stick_axis(axis, expected):
    from lerobot.teleoperators.gamepad import gamepad_utils

    with patch.object(gamepad_utils, "require_package"):
        backend = gamepad_utils.GamepadController(right_stick_axis=axis)
    backend.joystick = MagicMock()
    backend.joystick.get_axis.side_effect = [0.0, 0.0, 0.5 if axis == "x" else -0.25]
    assert backend.get_deltas() == (0.0, 0.0, expected)
    assert backend.joystick.get_axis.call_args.args == (2 if axis == "x" else 3,)

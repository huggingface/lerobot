"""
Tests meant to be used locally and launched manually.

Example usage:
```bash
pytest -sx tests/test_robots.py::test_robot
```
"""

from pathlib import Path

import pytest
import torch

from lerobot import available_robots
from lerobot.common.robot_devices.robots.factory import make_robot as make_robot_from_cfg
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import init_hydra_config
from tests.utils import ROBOT_CONFIG_PATH_TEMPLATE, require_robot


def make_robot(robot_type: str, overrides: list[str] | None = None) -> Robot:
    config_path = ROBOT_CONFIG_PATH_TEMPLATE.format(robot=robot_type)
    robot_cfg = init_hydra_config(config_path, overrides)
    robot = make_robot_from_cfg(robot_cfg)
    return robot


@pytest.mark.parametrize("robot_type", available_robots)
@require_robot
def test_robot(tmpdir, request, robot_type):
    # TODO(rcadene): measure fps in nightly?
    # TODO(rcadene): test logs
    # TODO(rcadene): add compatibility with other robots
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

    # Save calibration preset
    tmpdir = Path(tmpdir)
    calibration_dir = tmpdir / robot_type

    # Test connecting without devices raises an error
    robot = ManipulatorRobot()
    with pytest.raises(ValueError):
        robot.connect()
    del robot

    # Test using robot before connecting raises an error
    robot = ManipulatorRobot()
    with pytest.raises(RobotDeviceNotConnectedError):
        robot.teleop_step()
    with pytest.raises(RobotDeviceNotConnectedError):
        robot.teleop_step(record_data=True)
    with pytest.raises(RobotDeviceNotConnectedError):
        robot.capture_observation()
    with pytest.raises(RobotDeviceNotConnectedError):
        robot.send_action(None)
    with pytest.raises(RobotDeviceNotConnectedError):
        robot.disconnect()

    # Test deleting the object without connecting first
    del robot

    # Test connecting
    robot = make_robot(robot_type, overrides=[f"calibration_dir={calibration_dir}"])
    robot.connect()  # run the manual calibration precedure
    assert robot.is_connected

    # Test connecting twice raises an error
    with pytest.raises(RobotDeviceAlreadyConnectedError):
        robot.connect()

    # Test disconnecting with `__del__`
    del robot

    # Test teleop can run
    robot = make_robot(robot_type, overrides=[f"calibration_dir={calibration_dir}"])
    robot.calibration_dir = calibration_dir
    robot.connect()
    robot.teleop_step()

    # Test data recorded during teleop are well formated
    observation, action = robot.teleop_step(record_data=True)
    # State
    assert "observation.state" in observation
    assert isinstance(observation["observation.state"], torch.Tensor)
    assert observation["observation.state"].ndim == 1
    dim_state = sum(len(robot.follower_arms[name].motors) for name in robot.follower_arms)
    assert observation["observation.state"].shape[0] == dim_state
    # Cameras
    for name in robot.cameras:
        assert f"observation.images.{name}" in observation
        assert isinstance(observation[f"observation.images.{name}"], torch.Tensor)
        assert observation[f"observation.images.{name}"].ndim == 3
    # Action
    assert "action" in action
    assert isinstance(action["action"], torch.Tensor)
    assert action["action"].ndim == 1
    dim_action = sum(len(robot.follower_arms[name].motors) for name in robot.follower_arms)
    assert action["action"].shape[0] == dim_action
    # TODO(rcadene): test if observation and action data are returned as expected

    # Test capture_observation can run and observation returned are the same (since the arm didnt move)
    captured_observation = robot.capture_observation()
    assert set(captured_observation.keys()) == set(observation.keys())
    for name in captured_observation:
        if "image" in name:
            # TODO(rcadene): skipping image for now as it's challenging to assess equality between two consecutive frames
            continue
        assert torch.allclose(captured_observation[name], observation[name], atol=1)

    # Test send_action can run
    robot.send_action(action["action"])

    # Test disconnecting
    robot.disconnect()
    assert not robot.is_connected
    for name in robot.follower_arms:
        assert not robot.follower_arms[name].is_connected
    for name in robot.leader_arms:
        assert not robot.leader_arms[name].is_connected
    for name in robot.cameras:
        assert not robot.cameras[name].is_connected
    del robot

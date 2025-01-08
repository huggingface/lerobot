"""
Tests for physical robots and their mocked versions.
If the physical robots are not connected to the computer, or not working,
the test will be skipped.

Example of running a specific test:
```bash
pytest -sx tests/test_robots.py::test_robot
```

Example of running test on real robots connected to the computer:
```bash
pytest -sx 'tests/test_robots.py::test_robot[koch-False]'
pytest -sx 'tests/test_robots.py::test_robot[koch_bimanual-False]'
pytest -sx 'tests/test_robots.py::test_robot[aloha-False]'
```

Example of running test on a mocked version of robots:
```bash
pytest -sx 'tests/test_robots.py::test_robot[koch-True]'
pytest -sx 'tests/test_robots.py::test_robot[koch_bimanual-True]'
pytest -sx 'tests/test_robots.py::test_robot[aloha-True]'
```
"""

from pathlib import Path

import pytest
import torch

from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from tests.utils import TEST_ROBOT_TYPES, make_robot, mock_calibration_dir, require_robot


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
@require_robot
def test_robot(tmpdir, request, robot_type, mock):
    # TODO(rcadene): measure fps in nightly?
    # TODO(rcadene): test logs
    # TODO(rcadene): add compatibility with other robots
    robot_kwargs = {"robot_type": robot_type}

    if robot_type == "aloha" and mock:
        # To simplify unit test, we do not rerun manual calibration for Aloha mock=True.
        # Instead, we use the files from '.cache/calibration/aloha_default'
        overrides_calibration_dir = None
    else:
        if mock:
            request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        tmpdir = Path(tmpdir)
        calibration_dir = tmpdir / robot_type
        overrides_calibration_dir = [f"calibration_dir={calibration_dir}"]
        mock_calibration_dir(calibration_dir)
        robot_kwargs["calibration_dir"] = calibration_dir

    # Test connecting without devices raises an error
    robot = ManipulatorRobot(**robot_kwargs)
    with pytest.raises(ValueError):
        robot.connect()
    del robot

    # Test using robot before connecting raises an error
    robot = ManipulatorRobot(**robot_kwargs)
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

    # Test connecting (triggers manual calibration)
    robot = make_robot(robot_type, overrides=overrides_calibration_dir, mock=mock)
    robot.connect()
    assert robot.is_connected

    # Test connecting twice raises an error
    with pytest.raises(RobotDeviceAlreadyConnectedError):
        robot.connect()

    # TODO(rcadene, aliberts): Test disconnecting with `__del__` instead of `disconnect`
    # del robot
    robot.disconnect()

    # Test teleop can run
    robot = make_robot(robot_type, overrides=overrides_calibration_dir, mock=mock)
    if overrides_calibration_dir is not None:
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
        assert captured_observation[name].shape == observation[name].shape

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

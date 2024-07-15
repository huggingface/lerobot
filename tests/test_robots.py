import pickle
from pathlib import Path

import pytest
import torch

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from tests.utils import require_koch


@require_koch
def test_robot(tmpdir, request):
    # TODO(rcadene): measure fps in nightly?
    # TODO(rcadene): test logs
    # TODO(rcadene): add compatibility with other robots
    from lerobot.common.robot_devices.robots.koch import KochRobot

    # Save calibration preset
    calibration = {
        "follower_main": {
            "shoulder_pan": (-2048, False),
            "shoulder_lift": (2048, True),
            "elbow_flex": (-1024, False),
            "wrist_flex": (2048, True),
            "wrist_roll": (2048, True),
            "gripper": (2048, True),
        },
        "leader_main": {
            "shoulder_pan": (-2048, False),
            "shoulder_lift": (1024, True),
            "elbow_flex": (2048, True),
            "wrist_flex": (-2048, False),
            "wrist_roll": (2048, True),
            "gripper": (2048, True),
        },
    }
    tmpdir = Path(tmpdir)
    calibration_path = tmpdir / "calibration.pkl"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calibration_path, "wb") as f:
        pickle.dump(calibration, f)

    # Test connecting without devices raises an error
    robot = KochRobot()
    with pytest.raises(ValueError):
        robot.connect()
    del robot

    # Test using robot before connecting raises an error
    robot = KochRobot()
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
    robot = make_robot("koch")
    # TODO(rcadene): proper monkey patch
    robot.calibration_path = calibration_path
    robot.connect()  # run the manual calibration precedure
    assert robot.is_connected

    # Test connecting twice raises an error
    with pytest.raises(RobotDeviceAlreadyConnectedError):
        robot.connect()

    # Test disconnecting with `__del__`
    del robot

    # Test teleop can run
    robot = make_robot("koch")
    robot.calibration_path = calibration_path
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

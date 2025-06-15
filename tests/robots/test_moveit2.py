from unittest.mock import MagicMock, patch

import pytest

from lerobot.common.robots.moveit2 import MoveIt2, MoveIt2Config


def _make_moveit2_interface_mock() -> MagicMock:
    """Return a MoveIt2Interface mock with just the attributes used by the robot."""
    interface = MagicMock(name="MoveIt2InterfaceMock")
    interface.is_connected = False

    cfg = MoveIt2Config()

    # Mock joint state
    all_joint_names = cfg.moveit2_interface.arm_joint_names + [cfg.moveit2_interface.gripper_joint_name]
    interface.joint_state = {
        "position": dict.fromkeys(all_joint_names, 0.1),
        "velocity": dict.fromkeys(all_joint_names, 0.2),
    }

    # Mock config
    config_mock = MagicMock()
    config_mock.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    config_mock.gripper_joint_name = "gripper_jaw1_joint"
    interface.config = config_mock

    def _connect():
        interface.is_connected = True

    def _disconnect():
        interface.is_connected = False

    interface.connect.side_effect = _connect
    interface.disconnect.side_effect = _disconnect

    # Mock servo and gripper commands
    interface.servo.return_value = None
    interface.send_gripper_command.return_value = True

    return interface


@pytest.fixture
def moveit2_robot():
    interface_mock = _make_moveit2_interface_mock()

    with patch("lerobot.common.robots.moveit2.moveit2.MoveIt2Interface", return_value=interface_mock):
        cfg = MoveIt2Config()
        robot = MoveIt2(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(moveit2_robot):
    """Test basic connection and disconnection."""
    assert not moveit2_robot.is_connected

    moveit2_robot.connect()
    assert moveit2_robot.is_connected

    moveit2_robot.disconnect()
    assert not moveit2_robot.is_connected


def test_get_observation(moveit2_robot):
    """Test getting observations from the robot."""
    moveit2_robot.connect()
    obs = moveit2_robot.get_observation()

    # Check that all expected joint positions are in the observation
    expected_joints = moveit2_robot.config.moveit2_interface.arm_joint_names + [
        moveit2_robot.config.moveit2_interface.gripper_joint_name
    ]
    expected_keys = {f"{joint}.pos" for joint in expected_joints}

    # Only check motor keys since cameras might not be configured
    motor_keys = {key for key in obs if key.endswith(".pos")}
    assert motor_keys == expected_keys

    # Check that the values match the mocked joint state
    for joint in expected_joints:
        assert obs[f"{joint}.pos"] == 0.1


def test_send_action(moveit2_robot):
    """Test sending action commands to the robot."""
    moveit2_robot.connect()

    action = {
        "linear_vel_x": 0.1,
        "linear_vel_y": 0.2,
        "linear_vel_z": 0.3,
        "angular_vel_x": 0.4,
        "angular_vel_y": 0.5,
        "angular_vel_z": 0.6,
        "gripper_pos": 0.8,
    }

    returned_action = moveit2_robot.send_action(action)
    assert returned_action == action

    # Verify that the interface methods were called correctly
    moveit2_robot.moveit2_interface.servo.assert_called_once_with(
        linear=(0.1, 0.2, 0.3), angular=(0.4, 0.5, 0.6)
    )
    moveit2_robot.moveit2_interface.send_gripper_command.assert_called_once_with(0.8)


def test_send_action_with_max_relative_target(moveit2_robot):
    """Test sending action with safety limits applied."""
    # Configure with a small max relative target for testing
    moveit2_robot.config.max_relative_target = 0.1
    moveit2_robot.connect()

    action = {
        "linear_vel_x": 1.0,  # Large value that should be clipped
        "linear_vel_y": 0.0,
        "linear_vel_z": 0.0,
        "angular_vel_x": 0.0,
        "angular_vel_y": 0.0,
        "angular_vel_z": 0.0,
        "gripper_pos": 0.5,
    }

    returned_action = moveit2_robot.send_action(action)

    # The action should be clipped due to max_relative_target
    assert returned_action["linear_vel_x"] <= 0.1
    assert returned_action["gripper_pos"] <= 0.1


def test_keyboard_action_conversion(moveit2_robot):
    """Test conversion from keyboard input to action commands."""
    moveit2_robot.config.action_from_keyboard = True
    moveit2_robot.connect()

    pressed_keys = {"w": True, "d": True, "space": True}

    returned_action = moveit2_robot.send_action(pressed_keys)

    # Check that keyboard inputs were converted correctly
    assert returned_action["linear_vel_x"] == 1.0  # 'd' key
    assert returned_action["linear_vel_y"] == 1.0  # 'w' key
    assert returned_action["linear_vel_z"] == 0.0
    assert returned_action["gripper_pos"] == 1.0  # 'space' key


def test_from_keyboard_to_action(moveit2_robot):
    """Test the keyboard to action conversion method directly."""
    # Test all movement keys
    pressed_keys = {
        "w": True,  # +linear_vel_y
        "s": True,  # -linear_vel_y (should cancel with w)
        "a": True,  # -linear_vel_x
        "d": True,  # +linear_vel_x (should cancel with a)
        "q": True,  # -linear_vel_z
        "e": True,  # +linear_vel_z (should cancel with q)
        "i": True,  # +angular_vel_x
        "k": True,  # -angular_vel_x (should cancel with i)
        "j": True,  # -angular_vel_y
        "l": True,  # +angular_vel_y (should cancel with j)
        "u": True,  # +angular_vel_z
        "o": True,  # -angular_vel_z (should cancel with u)
        "space": True,  # gripper
    }

    action = moveit2_robot.from_keyboard_to_action(pressed_keys)

    # All opposing keys should cancel out to 0
    assert action["linear_vel_x"] == 0.0
    assert action["linear_vel_y"] == 0.0
    assert action["linear_vel_z"] == 0.0
    assert action["angular_vel_x"] == 0.0
    assert action["angular_vel_y"] == 0.0
    assert action["angular_vel_z"] == 0.0
    assert action["gripper_pos"] == 1.0

    # Test individual keys
    single_key_tests = [
        ({"w": True}, {"linear_vel_y": 1.0}),
        ({"s": True}, {"linear_vel_y": -1.0}),
        ({"a": True}, {"linear_vel_x": -1.0}),
        ({"d": True}, {"linear_vel_x": 1.0}),
        ({"q": True}, {"linear_vel_z": -1.0}),
        ({"e": True}, {"linear_vel_z": 1.0}),
        ({"i": True}, {"angular_vel_x": -1.0}),
        ({"k": True}, {"angular_vel_x": 1.0}),
        ({"j": True}, {"angular_vel_y": -1.0}),
        ({"l": True}, {"angular_vel_y": 1.0}),
        ({"u": True}, {"angular_vel_z": 1.0}),
        ({"o": True}, {"angular_vel_z": -1.0}),
    ]

    for pressed_keys, expected_non_zero in single_key_tests:
        action = moveit2_robot.from_keyboard_to_action(pressed_keys)
        for key, expected_value in expected_non_zero.items():
            assert action[key] == expected_value


def test_observation_features(moveit2_robot):
    """Test that observation features are correctly defined."""
    features = moveit2_robot.observation_features

    # Check that all joint position features are defined
    expected_joints = moveit2_robot.config.moveit2_interface.arm_joint_names + [
        moveit2_robot.config.moveit2_interface.gripper_joint_name
    ]
    for joint in expected_joints:
        assert f"{joint}.pos" in features
        assert features[f"{joint}.pos"] is float


def test_action_features(moveit2_robot):
    """Test that action features are correctly defined."""
    features = moveit2_robot.action_features

    expected_actions = [
        "linear_vel_x",
        "linear_vel_y",
        "linear_vel_z",
        "angular_vel_x",
        "angular_vel_y",
        "angular_vel_z",
        "gripper_pos",
    ]

    for action in expected_actions:
        assert action in features
        assert features[action] is float


def test_calibration_methods(moveit2_robot):
    """Test calibration-related methods."""
    # MoveIt2 robot should always be considered calibrated
    assert moveit2_robot.is_calibrated

    # Calibrate method should do nothing (no-op)
    moveit2_robot.calibrate()  # Should not raise any exception


def test_configure_method(moveit2_robot):
    """Test configure method."""
    # Configure method should do nothing (no-op)
    moveit2_robot.configure()  # Should not raise any exception


def test_error_handling_when_not_connected(moveit2_robot):
    """Test that appropriate errors are raised when robot is not connected."""
    from lerobot.common.errors import DeviceNotConnectedError

    # Should raise error when trying to get observation without connection
    with pytest.raises(DeviceNotConnectedError):
        moveit2_robot.get_observation()

    # Should raise error when trying to send action without connection
    with pytest.raises(DeviceNotConnectedError):
        moveit2_robot.send_action({"linear_vel_x": 0.1, "gripper_pos": 0.5})

    # Should raise error when trying to disconnect without connection
    with pytest.raises(DeviceNotConnectedError):
        moveit2_robot.disconnect()


def test_double_connect_error(moveit2_robot):
    """Test that connecting twice raises an error."""
    from lerobot.common.errors import DeviceAlreadyConnectedError

    moveit2_robot.connect()

    with pytest.raises(DeviceAlreadyConnectedError):
        moveit2_robot.connect()

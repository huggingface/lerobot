import math
from unittest.mock import MagicMock

import pytest

from lerobot.robots.jaka_zu7_follower import JAKAZu7Follower, JAKAZu7FollowerConfig
from lerobot.robots.jaka_zu7_follower.jaka_zu7_follower import JAKA_ZU7_JOINTS, _limit_joint_step


class FakeJakaRobot:
    def __init__(self, *_args):
        self.joints_rad = [0.0] * 6
        self.servo_mode = False
        self.servo_commands = []

    def login(self):
        return (0,)

    def logout(self):
        return (0,)

    def get_joint_position(self):
        return (0, self.joints_rad)

    def servo_move_enable(self, enable):
        self.servo_mode = enable
        return (0,)

    def servo_j_extend(self, joints_rad, move_mode, step_num):
        self.servo_commands.append((joints_rad, move_mode, step_num))
        return (0,)


@pytest.fixture
def robot(tmp_path, monkeypatch):
    fake_jaka = FakeJakaRobot()
    sdk_module = MagicMock(RC=MagicMock(return_value=fake_jaka))
    monkeypatch.setattr(
        "lerobot.robots.jaka_zu7_follower.jaka_zu7_follower.importlib.import_module",
        lambda _name: sdk_module,
    )
    config = JAKAZu7FollowerConfig(
        ip_address="192.168.1.10",
        calibration_dir=tmp_path,
        max_joint_velocity_deg_s=1000,
        max_joint_acceleration_deg_s2=1000,
    )
    instance = JAKAZu7Follower(config)
    instance.connect()
    yield instance, fake_jaka
    if instance.is_connected:
        instance.disconnect()


def test_features_match_six_so_arm101_arm_joints(robot):
    instance, _ = robot

    assert list(instance.action_features) == [f"{joint}.pos" for joint in JAKA_ZU7_JOINTS]
    assert "gripper.pos" not in instance.action_features
    assert set(instance.get_observation()) == set(instance.observation_features)


def test_feedback_is_reported_in_degrees(robot):
    instance, fake_jaka = robot
    fake_jaka.joints_rad = [0, math.pi / 2, math.pi, -math.pi / 2, 0, 0]

    observation = instance.get_observation()

    assert observation["shoulder_pan.pos"] == pytest.approx(0)
    assert observation["heavy_joint.pos"] == pytest.approx(90)
    assert observation["shoulder_lift.pos"] == pytest.approx(180)
    assert observation["elbow_flex.pos"] == pytest.approx(-90)


def test_joint_signs_and_offsets_map_between_leader_and_jaka(robot):
    instance, fake_jaka = robot
    instance.config.joint_signs = (-1, 1, 1, 1, 1, 1)
    instance.config.joint_offsets_deg = (10, 0, 0, 0, 0, 0)
    fake_jaka.joints_rad = [math.radians(10), 0, 0, 0, 0, 0]

    observation = instance.get_observation()
    assert observation["shoulder_pan.pos"] == pytest.approx(0)

    action = {f"{joint}.pos": 0.0 for joint in JAKA_ZU7_JOINTS}
    instance.send_action(action)
    commanded_rad = fake_jaka.servo_commands[-1][0]
    assert commanded_rad[0] == pytest.approx(math.radians(10))


def test_send_action_uses_joint_order_and_ignores_leader_gripper(robot):
    instance, fake_jaka = robot
    action = {f"{joint}.pos": 10.0 for joint in JAKA_ZU7_JOINTS}
    action["gripper.pos"] = 75.0

    sent = instance.send_action(action)

    assert list(sent) == [f"{joint}.pos" for joint in JAKA_ZU7_JOINTS]
    assert len(fake_jaka.servo_commands) == 1
    joints_rad, move_mode, step_num = fake_jaka.servo_commands[0]
    assert move_mode == 0
    assert step_num == 2
    assert joints_rad == pytest.approx([math.radians(sent[f"{joint}.pos"]) for joint in JAKA_ZU7_JOINTS])
    assert max(abs(value) for value in sent.values()) < 10.0


def test_send_action_rejects_missing_and_unknown_joints(robot):
    instance, _ = robot
    with pytest.raises(KeyError, match="Missing JAKA"):
        instance.send_action({"shoulder_pan.pos": 0.0})

    action = {f"{joint}.pos": 0.0 for joint in JAKA_ZU7_JOINTS}
    action["extra.pos"] = 0.0
    with pytest.raises(KeyError, match="Unknown JAKA"):
        instance.send_action(action)


def test_sdk_failure_is_reported(robot):
    instance, fake_jaka = robot
    fake_jaka.servo_j_extend = MagicMock(return_value=(-3,))
    action = {f"{joint}.pos": 0.0 for joint in JAKA_ZU7_JOINTS}

    with pytest.raises(RuntimeError, match="servo_j_extend.*-3"):
        instance.send_action(action)
    assert instance._command_position_deg == [0.0] * 6
    assert instance._command_velocity_deg_s == [0.0] * 6


def test_disconnect_exits_servo_mode(robot):
    instance, fake_jaka = robot

    instance.disconnect()

    assert not fake_jaka.servo_mode
    assert not instance.is_connected


def test_limit_joint_step_bounds_velocity_and_acceleration():
    position, velocity = _limit_joint_step(
        position=[0.0],
        velocity=[0.0],
        target=[90.0],
        dt=0.01,
        max_velocity=5.0,
        max_acceleration=10.0,
    )

    assert velocity == pytest.approx([0.1])
    assert position == pytest.approx([0.001])

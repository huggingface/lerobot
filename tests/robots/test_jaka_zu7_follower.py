import json
import math
from unittest.mock import MagicMock

import pytest

from lerobot.robots.jaka_zu7_follower import JAKAZu7Follower, JAKAZu7FollowerConfig
from lerobot.robots.jaka_zu7_follower.jaka_zu7_follower import JAKA_ZU7_JOINTS, _limit_joint_step


class FakeSocket:
    """让 connect() 不真的建立 socket 连接。"""

    def settimeout(self, _timeout):
        pass

    def makefile(self, _mode):
        return MagicMock()

    def sendall(self, _data):
        pass

    def close(self):
        pass


@pytest.fixture
def robot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lerobot.robots.jaka_zu7_follower.jaka_zu7_follower.socket.create_connection",
        lambda *_args, **_kwargs: FakeSocket(),
    )
    config = JAKAZu7FollowerConfig(
        calibration_dir=tmp_path,
        max_joint_velocity_deg_s=1000,
        max_joint_acceleration_deg_s2=1000,
    )
    instance = JAKAZu7Follower(config)

    state = {"joints_deg": [0.0] * 6, "sent": [], "fail_set": False}

    def fake_request(req):
        if req["cmd"] == "get":
            return {"joints": list(state["joints_deg"])}
        if req["cmd"] == "set":
            if state["fail_set"]:
                raise RuntimeError("JAKA bridge error: servo_j 错误码 -3")
            state["sent"].append(req["joints"])
            return {"status": "ok"}
        if req["cmd"] == "exit":
            return {"status": "bye"}
        raise AssertionError(f"unexpected cmd {req}")

    monkeypatch.setattr(instance, "_request", fake_request)
    instance.connect()
    yield instance, state
    if instance.is_connected:
        instance.disconnect()


def test_features_match_six_so_arm101_arm_joints(robot):
    instance, _ = robot

    assert list(instance.action_features) == [f"{joint}.pos" for joint in JAKA_ZU7_JOINTS]
    assert "gripper.pos" not in instance.action_features
    assert set(instance.get_observation()) == set(instance.observation_features)


def test_feedback_is_reported_in_degrees(robot):
    instance, state = robot
    state["joints_deg"] = [0, 90, 180, -90, 0, 0]

    observation = instance.get_observation()

    assert observation["shoulder_pan.pos"] == pytest.approx(0)
    assert observation["heavy_joint.pos"] == pytest.approx(90)
    assert observation["shoulder_lift.pos"] == pytest.approx(180)
    assert observation["elbow_flex.pos"] == pytest.approx(-90)


def test_joint_signs_and_offsets_map_between_leader_and_jaka(robot):
    instance, state = robot
    instance.config.joint_signs = (-1, 1, 1, 1, 1, 1)
    instance.config.joint_offsets_deg = (10, 0, 0, 0, 0, 0)
    state["joints_deg"] = [10, 0, 0, 0, 0, 0]

    observation = instance.get_observation()
    assert observation["shoulder_pan.pos"] == pytest.approx(0)

    action = {f"{joint}.pos": 0.0 for joint in JAKA_ZU7_JOINTS}
    instance.send_action(action)
    assert state["sent"][-1][0] == pytest.approx(10)


def test_send_action_uses_joint_order_and_ignores_leader_gripper(robot):
    instance, state = robot
    action = {f"{joint}.pos": 10.0 for joint in JAKA_ZU7_JOINTS}
    action["gripper.pos"] = 75.0

    sent = instance.send_action(action)

    assert list(sent) == [f"{joint}.pos" for joint in JAKA_ZU7_JOINTS]
    assert len(state["sent"]) == 1
    assert len(state["sent"][0]) == 6
    # 软限速:第一步到不了 10
    assert max(abs(value) for value in sent.values()) < 10.0


def test_send_action_rejects_missing_and_unknown_joints(robot):
    instance, _ = robot
    with pytest.raises(KeyError, match="Missing JAKA"):
        instance.send_action({"shoulder_pan.pos": 0.0})

    action = {f"{joint}.pos": 0.0 for joint in JAKA_ZU7_JOINTS}
    action["extra.pos"] = 0.0
    with pytest.raises(KeyError, match="Unknown JAKA"):
        instance.send_action(action)


def test_bridge_error_is_reported(robot):
    instance, state = robot
    state["fail_set"] = True
    action = {f"{joint}.pos": 0.0 for joint in JAKA_ZU7_JOINTS}

    with pytest.raises(RuntimeError, match="servo_j"):
        instance.send_action(action)


def test_disconnect_sends_exit(robot):
    instance, _ = robot

    instance.disconnect()

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


def test_request_raises_on_bridge_error():
    # 直接测真实的 _request:当 bridge 返回 {"status":"error"} 时必须抛 RuntimeError。
    class _ErrorFile:
        def readline(self):
            return json.dumps({"status": "error", "error": "servo_j 错误码 -3"}).encode("utf-8") + b"\n"

    class _ErrorSocket:
        def sendall(self, _data):
            pass

        def makefile(self, _mode):
            return _ErrorFile()

    instance = JAKAZu7Follower(JAKAZu7FollowerConfig(calibration_dir=None))
    instance._sock = _ErrorSocket()
    instance._recv_file = instance._sock.makefile("rb")

    with pytest.raises(RuntimeError, match="servo_j"):
        instance._request({"cmd": "set", "joints": [0.0] * 6})

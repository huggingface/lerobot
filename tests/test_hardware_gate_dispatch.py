from pathlib import Path

import pytest

from lerobot.rollout.hardware_gate import (
    send_action_checked_with_approval,
)


class FakeRobotWrapper:
    def __init__(self):
        self.actions = []

    def send_action_checked(self, action):
        self.actions.append(action)
        return "dispatched"


def test_hardware_gate_rejects_without_approval(monkeypatch):
    monkeypatch.delenv("S0101_HARDWARE_APPROVED", raising=False)
    robot = FakeRobotWrapper()

    with pytest.raises(RuntimeError, match="hardware dispatch disabled"):
        send_action_checked_with_approval(robot, {"joint": 1})

    assert robot.actions == []


def test_hardware_gate_dispatches_through_checked_wrapper(monkeypatch):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    monkeypatch.setenv("S0101_HARDWARE_APPROVED", "1")
    robot = FakeRobotWrapper()
    action = {"joint": 1}

    result = send_action_checked_with_approval(robot, action)

    assert result == "dispatched"
    assert robot.actions == [action]


def test_return_to_initial_position_uses_hardware_gate():
    source = (
        Path(__file__).parents[1]
        / "src/lerobot/rollout/strategies/core.py"
    ).read_text(encoding="utf-8")

    assert "send_action_checked_with_approval(robot, interp)" in source
    assert "robot.send_action(interp)" not in source


def test_dagger_action_dispatch_uses_hardware_gate():
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "src/lerobot/rollout/strategies/dagger.py"
    ).read_text(encoding="utf-8")

    assert (
        "send_action_checked_with_approval("
        "ctx.hardware.robot_wrapper, robot_action_to_send)"
    ) in source
    assert "robot.send_action(robot_action_to_send)" not in source


def test_episodic_action_dispatch_uses_hardware_gate():
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "src/lerobot/rollout/strategies/episodic.py"
    ).read_text(encoding="utf-8")

    assert (
        "send_action_checked_with_approval("
        "ctx.hardware.robot_wrapper, robot_action)"
    ) in source
    assert "robot.send_action(robot_action)" not in source



# SOFOLLOWER_SEND_ACTION_REGRESSION_TESTS

def test_real_sofollower_interface_fallback_requires_approval(monkeypatch):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")

    class FakeSOFollower:
        def __init__(self):
            self.actions = []

        def send_action(self, action):
            self.actions.append(action)
            return {"sent": action}

    robot = FakeSOFollower()
    action = {"joint": 1}

    result = send_action_checked_with_approval(robot, action)

    assert result == {"sent": action}
    assert robot.actions == [action]


def test_real_sofollower_interface_fallback_rejects_without_approval(
    monkeypatch,
):
    monkeypatch.delenv("SO101_HARDWARE_APPROVED", raising=False)

    class FakeSOFollower:
        def __init__(self):
            self.actions = []

        def send_action(self, action):
            self.actions.append(action)
            return {"sent": action}

    robot = FakeSOFollower()

    with pytest.raises(
        RuntimeError, match="hardware dispatch disabled"
    ):
        send_action_checked_with_approval(robot, {"joint": 1})

    assert robot.actions == []

import pytest

from lerobot.rollout.hardware_gate import send_action_checked_with_approval


class FakeRobot:
    """纯软件模拟对象，不连接硬件。"""

    def __init__(self):
        self.sent_actions = []

    def send_action_checked(self, action):
        self.sent_actions.append(action)
        return action


def test_dispatch_is_rejected_without_approval(monkeypatch):
    monkeypatch.delenv("SO101_HARDWARE_APPROVED", raising=False)
    robot = FakeRobot()
    action = {"joint_1": 0.0}

    with pytest.raises(RuntimeError, match="hardware dispatch disabled"):
        send_action_checked_with_approval(robot, action)

    assert robot.sent_actions == []


def test_approved_dispatch_uses_checked_send_action(monkeypatch):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    robot = FakeRobot()
    action = {"joint_1": 0.0}

    result = send_action_checked_with_approval(robot, action)

    assert result == action
    assert robot.sent_actions == [action]


def test_identified_entrypoints_use_authorization_gate():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    calls = {
        "src/lerobot/scripts/lerobot_teleoperate.py":
            "send_action_checked_with_approval(robot, robot_action_to_send)",
        "src/lerobot/scripts/lerobot_replay.py":
            "send_action_checked_with_approval(robot, processed_action)",
        "src/lerobot/teleoperators/keyboard/teleop_keyboard.py":
            "send_action_checked_with_approval(robot, action)",
    }

    for filename, expected in calls.items():
        source = (root / filename).read_text(encoding="utf-8")
        assert expected in source, f"安全门未接入：{filename}"



def test_action_interpolator_performs_no_robot_dispatch():
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src/lerobot/utils/action_interpolator.py"
    ).read_text(encoding="utf-8")

    tree = ast.parse(source)

    forbidden = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"send_action", "send_action_checked"}
    ]

    assert not forbidden

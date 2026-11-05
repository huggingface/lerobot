from threading import Lock

import pytest

from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.rollout.gripper_safety import GripperSafetyError


class FakeBus:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def read(self, name, motor, normalize=True):
        self.calls.append((name.lower(), motor, normalize))
        return self.values[name.lower()]


class FakeRobot:
    robot_type = "so101_follower"

    def __init__(self, values):
        self.bus = FakeBus(values)
        self.sent_actions = []

    def send_action(self, action):
        self.sent_actions.append(action)
        return action


def make_wrapper(values):
    robot = FakeRobot(values)
    wrapper = ThreadSafeRobot.__new__(ThreadSafeRobot)
    wrapper._robot = robot
    wrapper._lock = Lock()
    wrapper._position_hold_enabled = False
    wrapper._hardware_failure = None
    wrapper._validate_joint_limits_locked = lambda action: None
    return wrapper, robot


@pytest.fixture
def normal_feedback():
    return {
        "present_load": -300,
        "present_voltage": 120,
        "present_temperature": 35,
        "present_current": 100,
        "status": 0,
    }


def test_wrapper_uses_all_five_feedback_registers(
    normal_feedback, monkeypatch
):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    wrapper, robot = make_wrapper(normal_feedback)
    action = {"joint": 100}

    result = wrapper.send_action_checked(action)

    assert result == action
    assert robot.sent_actions == [action]
    assert [x[0] for x in robot.bus.calls] == [
        "present_load",
        "present_voltage",
        "present_temperature",
        "present_current",
        "status",
    ]
    assert not wrapper._gripper_safety_guard.fault_latched


def test_invalid_feedback_latches_fault_before_dispatch(
    normal_feedback, monkeypatch
):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    normal_feedback["present_load"] = 1024
    wrapper, robot = make_wrapper(normal_feedback)

    with pytest.raises(GripperSafetyError):
        wrapper.send_action_checked({"joint": 100})

    assert robot.sent_actions == []
    assert wrapper._gripper_safety_guard.fault_latched


def test_feedback_read_failure_latches_fault(
    normal_feedback, monkeypatch
):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    wrapper, robot = make_wrapper(normal_feedback)

    def failed_read(name, motor, normalize=True):
        raise ConnectionError("simulated offline read failure")

    robot.bus.read = failed_read

    with pytest.raises(GripperSafetyError):
        wrapper.send_action_checked({"joint": 100})

    assert robot.sent_actions == []
    assert wrapper._gripper_safety_guard.fault_latched

def prepare_hold(wrapper):
    wrapper._position_hold_enabled = True
    wrapper._last_applied_action = {"joint": 100}
    wrapper._validated_positions = lambda action: action.copy()


def test_hold_reads_fresh_feedback_before_dispatch(
    normal_feedback, monkeypatch
):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    wrapper, robot = make_wrapper(normal_feedback)
    prepare_hold(wrapper)

    wrapper.hold()

    assert len(robot.sent_actions) == 1
    assert [call[0] for call in robot.bus.calls] == [
        "present_load",
        "present_voltage",
        "present_temperature",
        "present_current",
        "status",
    ]


def test_hold_feedback_read_failure_blocks_dispatch(
    normal_feedback, monkeypatch
):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    wrapper, robot = make_wrapper(normal_feedback)
    prepare_hold(wrapper)

    def failed_read(name, motor, normalize=True):
        raise ConnectionError("simulated offline read failure")

    robot.bus.read = failed_read

    with pytest.raises(GripperSafetyError):
        wrapper.hold()

    assert robot.sent_actions == []
    assert wrapper._gripper_safety_guard.fault_latched


def test_hold_rejects_latched_gripper_fault(
    normal_feedback, monkeypatch
):
    from lerobot.rollout.gripper_safety import GripperSafetyGuard

    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    wrapper, robot = make_wrapper(normal_feedback)
    prepare_hold(wrapper)

    guard = GripperSafetyGuard()
    guard.fault_latched = True
    guard.fault_reason = "test overload"
    wrapper._gripper_safety_guard = guard

    with pytest.raises(GripperSafetyError):
        wrapper.hold()

    assert robot.bus.calls == []
    assert robot.sent_actions == []


def test_hold_rejects_missing_hardware_approval(
    normal_feedback, monkeypatch
):
    monkeypatch.delenv("SO101_HARDWARE_APPROVED", raising=False)
    wrapper, robot = make_wrapper(normal_feedback)
    prepare_hold(wrapper)

    with pytest.raises(RuntimeError, match="hardware dispatch disabled"):
        wrapper.hold()

    assert robot.bus.calls == []
    assert robot.sent_actions == []


def test_hold_rejects_unconfigured_position_hold(
    normal_feedback, monkeypatch
):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    wrapper, robot = make_wrapper(normal_feedback)

    with pytest.raises(RuntimeError, match="Local hold was not configured"):
        wrapper.hold()

    assert robot.bus.calls == []
    assert robot.sent_actions == []

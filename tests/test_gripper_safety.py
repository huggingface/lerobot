import pytest

from lerobot.rollout.gripper_safety import (
    GripperSafetyError,
    GripperSafetyGuard,
    GripperStatus,
)


def test_normal_gripper_status_passes():
    """正常反馈应通过安全检查。"""
    guard = GripperSafetyGuard()

    result = guard.check(GripperStatus(0, 0, 25, 12))

    assert result is True
    assert guard.fault_latched is False


def test_missing_status_latches_fault():
    """缺少夹爪反馈时应触发并锁定故障。"""
    guard = GripperSafetyGuard()

    with pytest.raises(GripperSafetyError):
        guard.check(None)

    assert guard.fault_latched is True


def test_non_finite_feedback_latches_fault():
    """反馈包含 NaN 时应触发保护。"""
    guard = GripperSafetyGuard()
    status = GripperStatus(float("nan"), 0, 25, 12)

    with pytest.raises(GripperSafetyError):
        guard.check(status)

    assert guard.fault_latched is True


def test_overload_latches_fault():
    """检测到夹爪过载时应触发并锁定故障。"""
    guard = GripperSafetyGuard()
    status = GripperStatus(
        0, 0, 25, 12,
        overload_detected=True,
    )

    with pytest.raises(GripperSafetyError):
        guard.check(status)

    assert guard.fault_latched is True


def test_invalid_overload_flag_latches_fault():
    """过载标志不是布尔值时应触发保护。"""
    guard = GripperSafetyGuard()
    status = GripperStatus(
        0, 0, 25, 12,
        overload_detected=1,
    )

    with pytest.raises(GripperSafetyError):
        guard.check(status)

    assert guard.fault_latched is True


def test_latched_fault_blocks_subsequent_checks():
    """故障锁定后，即使反馈恢复正常，也不能直接解除锁定。"""
    guard = GripperSafetyGuard()

    with pytest.raises(GripperSafetyError):
        guard.check(None)

    with pytest.raises(GripperSafetyError):
        guard.check(GripperStatus(0, 0, 25, 12))

    assert guard.fault_latched is True

from unittest.mock import MagicMock

import pytest

from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import (
    SOLeaderTeleopConfig,
)
from lerobot.robots.so_follower.so101_safe_bus import (
    SO101SafeSerialMotorsBus,
)


@pytest.fixture
def leader():
    config = SOLeaderTeleopConfig(port="/dev/null")
    instance = SOLeader(config)

    assert isinstance(instance.bus, SO101SafeSerialMotorsBus)
    assert not instance.is_connected

    return instance


def test_leader_uses_safe_bus(leader):
    assert isinstance(leader.bus, SO101SafeSerialMotorsBus)


def test_leader_motor_ids(leader):
    assert {
        name: motor.id
        for name, motor in leader.bus.motors.items()
    } == {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "wrist_roll": 5,
        "gripper": 6,
    }


def test_leader_connect_rejects_missing_connect_authorization(
    leader, monkeypatch
):
    monkeypatch.delenv(
        "SO101_HARDWARE_CONNECT_APPROVED", raising=False
    )

    with pytest.raises(RuntimeError, match="SO101_HARDWARE_CONNECT_APPROVED"):
        leader.connect(calibrate=False)


def test_leader_connect_rejects_missing_setup_authorization(
    leader, monkeypatch
):
    monkeypatch.setenv("SO101_HARDWARE_CONNECT_APPROVED", "1")
    monkeypatch.delenv(
        "SO101_HARDWARE_SETUP_APPROVED", raising=False
    )

    with pytest.raises(RuntimeError, match="SO101_HARDWARE_SETUP_APPROVED"):
        leader.connect(calibrate=False)


def test_leader_enable_torque_rejects_missing_authorization(
    leader, monkeypatch
):
    monkeypatch.delenv(
        "SO101_HARDWARE_TORQUE_APPROVED", raising=False
    )

    with pytest.raises(RuntimeError, match="SO101_HARDWARE_TORQUE_APPROVED"):
        leader.enable_torque()


def test_leader_disable_torque_does_not_require_enable_authorization(
    leader, monkeypatch
):
    monkeypatch.delenv(
        "SO101_HARDWARE_TORQUE_APPROVED", raising=False
    )

    # Disconnected bus: verify safe shutdown remains callable only
    # through the normal connection-state contract.
    from lerobot.utils.errors import DeviceNotConnectedError

    with pytest.raises(DeviceNotConnectedError):
        leader.disable_torque()


def test_leader_setup_requires_setup_authorization(leader, monkeypatch):
    monkeypatch.delenv(
        "SO101_HARDWARE_SETUP_APPROVED", raising=False
    )

    with pytest.raises(RuntimeError, match="SO101_HARDWARE_SETUP_APPROVED"):
        leader.bus.configure_motors()


def test_leader_calibration_write_requires_setup_authorization(
    leader, monkeypatch
):
    monkeypatch.delenv(
        "SO101_HARDWARE_SETUP_APPROVED", raising=False
    )

    with pytest.raises(RuntimeError, match="SO101_HARDWARE_SETUP_APPROVED"):
        leader.bus.write_calibration({})


def test_leader_action_write_requires_action_authorization(
    leader, monkeypatch
):
    monkeypatch.delenv(
        "SO101_HARDWARE_APPROVED", raising=False
    )

    with pytest.raises(RuntimeError, match="SO101_HARDWARE_APPROVED"):
        leader.bus.sync_write("Goal_Position", {"shoulder_pan": 100})


def test_leader_unknown_register_is_rejected(leader, monkeypatch):
    monkeypatch.setenv("SO101_HARDWARE_SETUP_APPROVED", "1")

    with pytest.raises(RuntimeError, match="unclassified register"):
        leader.bus.write("Unknown_Register", "shoulder_pan", 100)


def test_leader_get_action_requires_connection(leader):
    from lerobot.utils.errors import DeviceNotConnectedError

    with pytest.raises(DeviceNotConnectedError):
        leader.get_action()


def test_leader_feedback_requires_connection(leader):
    from lerobot.utils.errors import DeviceNotConnectedError

    with pytest.raises(DeviceNotConnectedError):
        leader.send_feedback({"shoulder_pan.pos": 100.0})


def test_leader_connect_does_not_implicitly_enable_torque():
    # Static method-level contract: connect/configure contains no
    # call to enable_torque in the Leader implementation.
    import inspect

    source = inspect.getsource(SOLeader.connect)
    assert "enable_torque" not in source


def test_leader_configuration_does_not_enable_torque():
    import inspect

    source = inspect.getsource(SOLeader.configure)
    assert "enable_torque" not in source


def test_leader_safe_bus_import_is_explicit():
    assert SOLeader.__init__.__globals__[
        "SO101SafeSerialMotorsBus"
    ] is SO101SafeSerialMotorsBus

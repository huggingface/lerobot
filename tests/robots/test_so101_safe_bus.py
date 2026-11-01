import pytest

from lerobot.robots.so_follower.so101_safe_bus import (
    SO101SafeSerialMotorsBus,
)


@pytest.fixture
def bus():
    return object.__new__(SO101SafeSerialMotorsBus)


@pytest.mark.parametrize(
    "method,env",
    [
        ("connect", "SO101_HARDWARE_CONNECT_APPROVED"),
        ("_connect", "SO101_HARDWARE_CONNECT_APPROVED"),
        ("_open_bus", "SO101_HARDWARE_CONNECT_APPROVED"),
        ("sync_write", "SO101_HARDWARE_APPROVED"),
        ("enable_torque", "SO101_HARDWARE_TORQUE_APPROVED"),
        ("setup_motor", "SO101_HARDWARE_SETUP_APPROVED"),
        ("setup_motors", "SO101_HARDWARE_SETUP_APPROVED"),
        ("record_calibration", "SO101_HARDWARE_SETUP_APPROVED"),
        ("write_calibration", "SO101_HARDWARE_SETUP_APPROVED"),
        ("configure_motors", "SO101_HARDWARE_SETUP_APPROVED"),
        ("set_operating_mode", "SO101_HARDWARE_SETUP_APPROVED"),
        ("write", "SO101_HARDWARE_SETUP_APPROVED"),
        ("set_baudrate", "SO101_HARDWARE_SETUP_APPROVED"),
    ],
)
def test_unauthorized_operations_are_rejected(bus, monkeypatch, method, env):
    monkeypatch.delenv(env, raising=False)

    with pytest.raises(RuntimeError, match="SO101 safety gate"):
        getattr(bus, method)()


def test_connect_requires_setup_authorization(bus, monkeypatch):
    monkeypatch.setenv("SO101_HARDWARE_CONNECT_APPROVED", "1")
    monkeypatch.delenv("SO101_HARDWARE_SETUP_APPROVED", raising=False)

    with pytest.raises(RuntimeError, match="motor setup"):
        bus.connect()


def test_torque_disabled_never_restores_torque(bus):
    calls = []

    bus.disable_torque = lambda motors=None: calls.append("disable")
    bus.enable_torque = lambda *a, **k: calls.append("enable")

    with bus.torque_disabled():
        calls.append("inside")

    assert calls == ["disable", "inside"]


def test_torque_disabled_exception_never_restores_torque(bus):
    calls = []

    bus.disable_torque = lambda motors=None: calls.append("disable")
    bus.enable_torque = lambda *a, **k: calls.append("enable")

    with pytest.raises(ValueError):
        with bus.torque_disabled():
            raise ValueError("offline test")

    assert calls == ["disable"]


def test_torque_disabled_preserves_motor_selection(bus):
    calls = []

    bus.disable_torque = lambda motors=None: calls.append(motors)

    with bus.torque_disabled(["gripper"]):
        pass

    assert calls == [["gripper"]]


def test_write_goal_position_requires_action(bus, monkeypatch):
    monkeypatch.setenv("SO101_HARDWARE_SETUP_APPROVED", "1")
    monkeypatch.delenv("SO101_HARDWARE_APPROVED", raising=False)

    with pytest.raises(RuntimeError, match="motor action dispatch"):
        bus.write("Goal_Position", "shoulder", 0)


def test_write_configuration_requires_setup(bus, monkeypatch):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    monkeypatch.delenv("SO101_HARDWARE_SETUP_APPROVED", raising=False)

    with pytest.raises(RuntimeError, match="motor register write"):
        bus.write("P_Coefficient", "shoulder", 10)


def test_sync_write_goal_position_requires_action(bus, monkeypatch):
    monkeypatch.setenv("SO101_HARDWARE_SETUP_APPROVED", "1")
    monkeypatch.delenv("SO101_HARDWARE_APPROVED", raising=False)

    with pytest.raises(RuntimeError, match="motor action dispatch"):
        bus.sync_write("Goal_Position", {"shoulder": 0})


def test_sync_write_configuration_requires_setup(bus, monkeypatch):
    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    monkeypatch.delenv("SO101_HARDWARE_SETUP_APPROVED", raising=False)

    with pytest.raises(RuntimeError, match="motor register write"):
        bus.sync_write("P_Coefficient", {"shoulder": 10})


# B43.5.4: fail-closed register and torque authorization tests.

def test_b4354_unknown_register_rejected_even_with_setup_approval(monkeypatch):
    from lerobot.robots.so_follower.so101_safe_bus import SO101SafeSerialMotorsBus

    monkeypatch.setenv("SO101_HARDWARE_SETUP_APPROVED", "1")
    bus = object.__new__(SO101SafeSerialMotorsBus)

    with pytest.raises(RuntimeError, match="unclassified register"):
        bus._authorize_register_write("Unknown_Register", "test")


@pytest.mark.parametrize(
    "register",
    [
        "Goal_Position",
        "Goal_Velocity",
        "Goal_Current",
        "Goal_Torque",
        "Goal_PWM",
    ],
)
def test_b4354_motion_register_requires_action_approval(monkeypatch, register):
    from lerobot.robots.so_follower.so101_safe_bus import SO101SafeSerialMotorsBus

    monkeypatch.delenv("SO101_HARDWARE_APPROVED", raising=False)
    bus = object.__new__(SO101SafeSerialMotorsBus)

    with pytest.raises(RuntimeError, match="disabled"):
        bus._authorize_register_write(register, "test")


def test_b4354_motion_register_accepts_action_approval(monkeypatch):
    from lerobot.robots.so_follower.so101_safe_bus import SO101SafeSerialMotorsBus

    monkeypatch.setenv("SO101_HARDWARE_APPROVED", "1")
    bus = object.__new__(SO101SafeSerialMotorsBus)

    bus._authorize_register_write("Goal_Position", "test")


def test_b4354_configuration_register_requires_setup_approval(monkeypatch):
    from lerobot.robots.so_follower.so101_safe_bus import SO101SafeSerialMotorsBus

    monkeypatch.delenv("SO101_HARDWARE_SETUP_APPROVED", raising=False)
    bus = object.__new__(SO101SafeSerialMotorsBus)

    with pytest.raises(RuntimeError, match="disabled"):
        bus._authorize_register_write("P_Coefficient", "test")


def test_b4354_configuration_register_accepts_setup_approval(monkeypatch):
    from lerobot.robots.so_follower.so101_safe_bus import SO101SafeSerialMotorsBus

    monkeypatch.setenv("SO101_HARDWARE_SETUP_APPROVED", "1")
    bus = object.__new__(SO101SafeSerialMotorsBus)

    bus._authorize_register_write("P_Coefficient", "test")


def test_b4354_low_level_torque_enable_rejected(monkeypatch):
    from lerobot.robots.so_follower.so101_safe_bus import SO101SafeSerialMotorsBus

    monkeypatch.delenv("SO101_HARDWARE_TORQUE_APPROVED", raising=False)
    bus = object.__new__(SO101SafeSerialMotorsBus)

    with pytest.raises(RuntimeError, match="disabled"):
        bus._set_torque(None, True, 0)


def test_b4354_invalid_torque_state_rejected(monkeypatch):
    from lerobot.robots.so_follower.so101_safe_bus import SO101SafeSerialMotorsBus

    bus = object.__new__(SO101SafeSerialMotorsBus)

    with pytest.raises(RuntimeError, match="explicit boolean"):
        bus._set_torque(None, 1, 0)


class FakeDisconnectBus:
    def __init__(self, fail_close=False):
        self.calls = []
        self.fail_close = fail_close

    def close(self):
        self.calls.append("close")
        if self.fail_close:
            raise OSError("simulated close failure")


def test_disconnect_disables_torque_before_close(bus):
    raw = FakeDisconnectBus()
    bus._bus = raw

    def disable_torque(*args, **kwargs):
        raw.calls.append("disable_torque")

    bus.disable_torque = disable_torque

    bus.disconnect()

    assert raw.calls == ["disable_torque", "close"]
    assert bus.is_connected is False


def test_disconnect_closes_port_when_torque_disable_fails(bus):
    raw = FakeDisconnectBus()
    bus._bus = raw

    def disable_torque(*args, **kwargs):
        raise RuntimeError("simulated torque disable failure")

    bus.disable_torque = disable_torque

    with pytest.raises(RuntimeError, match="torque disable failure"):
        bus.disconnect()

    assert raw.calls == ["close"]
    assert bus.is_connected is False


def test_disconnect_without_torque_disable(bus):
    raw = FakeDisconnectBus()
    bus._bus = raw

    def unexpected_disable(*args, **kwargs):
        raise AssertionError("torque disable must not be called")

    bus.disable_torque = unexpected_disable

    bus.disconnect(disable_torque=False)

    assert raw.calls == ["close"]
    assert bus.is_connected is False


def test_repeated_disconnect_is_safe(bus):
    raw = FakeDisconnectBus()
    bus._bus = raw

    bus.disable_torque = lambda *args, **kwargs: None

    bus.disconnect()
    bus.disconnect()

    assert raw.calls == ["close"]
    assert bus.is_connected is False


def test_close_failure_preserves_connection_reference(bus):
    raw = FakeDisconnectBus(fail_close=True)
    bus._bus = raw

    bus.disable_torque = lambda *args, **kwargs: None

    with pytest.raises(OSError, match="close failure"):
        bus.disconnect()

    assert raw.calls == ["close"]
    # The close operation failed, so retain the reference rather
    # than falsely claiming that the bus is disconnected.
    assert bus._bus is raw
    assert bus.is_connected is True

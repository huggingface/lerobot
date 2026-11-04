import pytest

from lerobot.robots.so_follower.so101_safe_bus import (
    SO101SafeSerialMotorsBus,
    _SO101RustyPotBusProxy,
)


class FakeRawBus:
    def __init__(self):
        self.calls = []

    def set_torque(self, ids, enabled, retries=0):
        self.calls.append(("set_torque", ids, enabled, retries))
        return True

    def set_baudrate(self, baudrate):
        self.calls.append(("set_baudrate", baudrate))
        return True

    def write_register(self, id, name, value, retries=0):
        self.calls.append(("write_register", id, name, value))
        return True

    def sync_write_register(self, ids, name, values, retries=0):
        self.calls.append(("sync_write_register", ids, name, values))
        return True

    def read_register(self, id, name, retries=0):
        self.calls.append(("read_register", id, name))
        return 123

    def close(self):
        self.calls.append(("close",))
        return None


@pytest.fixture
def proxy():
    owner = object.__new__(SO101SafeSerialMotorsBus)
    raw = FakeRawBus()
    return _SO101RustyPotBusProxy(raw, owner), raw


def test_proxy_rejects_unauthorized_torque_enable(proxy, monkeypatch):
    p, raw = proxy

    monkeypatch.delenv(
        "SO101_HARDWARE_TORQUE_APPROVED",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="torque"):
        p.set_torque([1], True)

    assert raw.calls == []


def test_proxy_accepts_authorized_torque_enable(proxy, monkeypatch):
    p, raw = proxy

    monkeypatch.setenv(
        "SO101_HARDWARE_TORQUE_APPROVED",
        "1",
    )

    assert p.set_torque([1], True) is True

    assert raw.calls == [
        ("set_torque", [1], True, 0)
    ]


def test_proxy_allows_torque_disable_without_enable_authorization(
    proxy, monkeypatch
):
    p, raw = proxy

    monkeypatch.delenv(
        "SO101_HARDWARE_TORQUE_APPROVED",
        raising=False,
    )

    assert p.set_torque([1], False) is True

    assert raw.calls == [
        ("set_torque", [1], False, 0)
    ]


def test_proxy_rejects_invalid_torque_state(proxy):
    p, raw = proxy

    with pytest.raises(RuntimeError, match="invalid torque state"):
        p.set_torque([1], 1)

    assert raw.calls == []


def test_proxy_rejects_unauthorized_setup(proxy, monkeypatch):
    p, raw = proxy

    monkeypatch.delenv(
        "SO101_HARDWARE_SETUP_APPROVED",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="SETUP"):
        p.set_baudrate(1000000)

    assert raw.calls == []


def test_proxy_accepts_authorized_setup(proxy, monkeypatch):
    p, raw = proxy

    monkeypatch.setenv(
        "SO101_HARDWARE_SETUP_APPROVED",
        "1",
    )

    assert p.set_baudrate(1000000) is True

    assert raw.calls == [
        ("set_baudrate", 1000000)
    ]


def test_proxy_rejects_unauthorized_action(proxy, monkeypatch):
    p, raw = proxy

    monkeypatch.delenv(
        "SO101_HARDWARE_APPROVED",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="SO101_HARDWARE_APPROVED=1"):
        p.write_register(1, "Goal_Position", 2048)

    assert raw.calls == []


def test_proxy_accepts_authorized_action(proxy, monkeypatch):
    p, raw = proxy

    monkeypatch.setenv(
        "SO101_HARDWARE_APPROVED",
        "1",
    )

    assert p.write_register(
        1, "Goal_Position", 2048
    ) is True

    assert raw.calls == [
        ("write_register", 1, "Goal_Position", 2048)
    ]


def test_proxy_sync_action_requires_authorization(proxy, monkeypatch):
    p, raw = proxy

    monkeypatch.delenv(
        "SO101_HARDWARE_APPROVED",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="SO101_HARDWARE_APPROVED=1"):
        p.sync_write_register(
            [1, 2],
            "Goal_Position",
            [2048, 2048],
        )

    assert raw.calls == []


def test_proxy_accepts_authorized_sync_action(proxy, monkeypatch):
    p, raw = proxy

    monkeypatch.setenv(
        "SO101_HARDWARE_APPROVED",
        "1",
    )

    p.sync_write_register(
        [1, 2],
        "Goal_Position",
        [2048, 2048],
    )

    assert len(raw.calls) == 1
    assert raw.calls[0][0] == "sync_write_register"


@pytest.mark.parametrize(
    "name",
    [
        "_owner",
        "_raw",
        "__owner",
        "__raw_bus",
        "_SO101RustyPotBusProxy__owner",
        "_SO101RustyPotBusProxy__raw_bus",
    ],
)
def test_proxy_hides_internal_attributes(proxy, name):
    p, _ = proxy

    with pytest.raises(RuntimeError, match="underlying Bus method rejected"):
        getattr(p, name)


def test_proxy_rejects_unknown_method(proxy):
    p, raw = proxy

    with pytest.raises(RuntimeError, match="underlying Bus method rejected"):
        p.dangerous_method()

    assert raw.calls == []


def test_proxy_allows_read_without_hardware_write_authorization(
    proxy, monkeypatch
):
    p, raw = proxy

    monkeypatch.delenv(
        "SO101_HARDWARE_APPROVED",
        raising=False,
    )

    assert p.read_register(1, "Present_Position") == 123

    assert raw.calls == [
        ("read_register", 1, "Present_Position")
    ]


def test_proxy_allows_close(proxy):
    p, raw = proxy

    p.close()

    assert raw.calls == [
        ("close",)
    ]

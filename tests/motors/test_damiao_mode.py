"""Mode switching regressions using python-can's in-process virtual transport."""

import threading
from uuid import uuid4

import pytest

import lerobot.motors.damiao.damiao as damiao_module
from lerobot.motors import Motor
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import CAN_CMD_WRITE_PARAM, CAN_PARAM_ID, ControlMode

can = pytest.importorskip("can")


def state_frame(motor_id):
    return can.Message(
        arbitration_id=0x10 + motor_id,
        data=[0x10 | motor_id, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF, 30, 30],
        is_extended_id=False,
    )


@pytest.fixture
def virtual_bus():
    channel = str(uuid4())
    peer = can.Bus(interface="virtual", channel=channel)
    bus = DamiaoMotorsBus(
        port=channel,
        can_interface="virtual",
        use_can_fd=False,
        motors={
            f"joint_{i}": Motor(
                id=i, model="damiao", norm_mode="degrees", motor_type_str="dm4310", recv_id=0x10 + i
            )
            for i in (1, 2)
        },
    )
    bus.connect(handshake=False)
    try:
        yield bus, peer
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        peer.shutdown()


def install_responder(bus, peer, monkeypatch, on_write):
    """Reply through the real virtual transport, with configurable parameter ACKs."""
    sent = []
    original_send = bus.canbus.send

    def send(msg):
        original_send(msg)
        sent.append(msg)
        if msg.arbitration_id == CAN_PARAM_ID and msg.data[2] == CAN_CMD_WRITE_PARAM:
            on_write(msg)
        else:
            peer.send(state_frame(msg.arbitration_id))

    monkeypatch.setattr(bus.canbus, "send", send)
    return sent


def ack_frame(msg):
    return can.Message(arbitration_id=0x10 + msg.data[0], data=msg.data, is_extended_id=False)


def test_delayed_ack_does_not_accept_state_or_corrupt_cache(virtual_bus, monkeypatch):
    bus, peer = virtual_bus
    timers = []
    modes_before_ack = []

    def on_write(msg):
        # A late DISABLE state reply arrives before the actual mode ACK.
        peer.send(state_frame(msg.data[0]))

        def send_ack():
            modes_before_ack.append(bus.operation_mode.get(f"joint_{msg.data[0]}"))
            peer.send(ack_frame(msg))

        timer = threading.Timer(0.003, send_ack)
        timers.append(timer)
        timer.start()

    install_responder(bus, peer, monkeypatch, on_write)
    try:
        with bus.torque_disabled():
            bus.configure_motors()
        assert modes_before_ack == [None, None]
        assert all(mode == ControlMode.MIT for mode in bus.operation_mode.values())
        assert all(abs(state["position"]) < 0.1 for state in bus._last_known_states.values())
        assert bus.canbus.recv(0) is None
    finally:
        for timer in timers:
            timer.join()


@pytest.mark.parametrize(
    "bad_field", ["motor", "command", "register", "value", "length", "can_id", "missing"]
)
def test_invalid_or_missing_ack_fails_before_enable(virtual_bus, monkeypatch, bad_field):
    bus, peer = virtual_bus
    monkeypatch.setattr(damiao_module, "LONG_TIMEOUT_SEC", 0.01)
    bus.operation_mode["joint_1"] = ControlMode.MIT

    def on_write(msg):
        # Same-ID state replies are insufficient, even after a prior successful setup.
        peer.send(state_frame(msg.data[0]))
        ack = ack_frame(msg)
        if bad_field == "missing":
            return
        if bad_field == "can_id":
            ack.arbitration_id += 1
        elif bad_field == "length":
            ack.data = ack.data[:-1]
        else:
            index = {"motor": 0, "command": 2, "register": 3, "value": 4}[bad_field]
            ack.data[index] += 1
        peer.send(ack)

    sent = install_responder(bus, peer, monkeypatch, on_write)
    with pytest.raises(TimeoutError, match="joint_1.*CTRL_MODE=MIT"):
        bus.configure_motors()
    assert bus.operation_mode["joint_1"] is None
    assert len(sent) == 1
    assert sent[0].arbitration_id == CAN_PARAM_ID


@pytest.mark.parametrize("motor_keys", [("joint_1", "joint_2"), (1, 2)])
def test_batch_checks_all_modes_before_sending(virtual_bus, motor_keys):
    bus, peer = virtual_bus
    bus.operation_mode["joint_1"] = ControlMode.MIT
    commands = dict.fromkeys(motor_keys, (10.0, 0.5, 0.0, 0.0, 0.0))
    with pytest.raises(RuntimeError):
        bus._mit_control_batch(commands)
    assert peer.recv(0) is None


def test_configure_allows_single_and_batch_control(virtual_bus, monkeypatch):
    bus, peer = virtual_bus
    sent = install_responder(bus, peer, monkeypatch, lambda msg: peer.send(ack_frame(msg)))
    with pytest.raises(RuntimeError):
        bus._mit_control("joint_1", 10.0, 0.5, 0.0, 0.0, 0.0)
    assert not sent
    bus.configure_motors()
    sent.clear()
    bus._mit_control("joint_1", 10.0, 0.5, 0.0, 0.0, 0.0)
    bus._mit_control_batch(dict.fromkeys(bus.motors, (10.0, 0.5, 0.0, 0.0, 0.0)))
    assert [msg.arbitration_id for msg in sent] == [1, 1, 2]


def test_disconnect_and_reconnect_require_configuration(virtual_bus, monkeypatch):
    bus, peer = virtual_bus
    install_responder(bus, peer, monkeypatch, lambda msg: peer.send(ack_frame(msg)))
    bus.configure_motors()
    bus.disconnect(disable_torque=False)
    assert all(mode is None for mode in bus.operation_mode.values())
    bus.connect(handshake=False)
    with pytest.raises(RuntimeError, match="configure_motors"):
        bus._mit_control_batch({"joint_1": (10.0, 0.5, 0.0, 0.0, 0.0)})

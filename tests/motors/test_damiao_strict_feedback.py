"""CAN-frame regression tests for fresh feedback and public MIT batch writes."""

from collections import deque
from unittest.mock import Mock

import pytest

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus

can = pytest.importorskip("can")


def make_bus():
    bus = DamiaoMotorsBus(
        "virtual",
        {"joint": Motor(1, "dm4310", MotorNormMode.DEGREES, motor_type_str="dm4310", recv_id=17)},
        use_can_fd=False,
    )
    bus.canbus = Mock()
    bus._is_connected = True
    return bus


def frame(data):
    return can.Message(arbitration_id=17, data=data, is_extended_id=False)


def respond_on_send(bus, response):
    queue = deque()
    bus.canbus.send.side_effect = lambda _: queue.append(response) if response is not None else None
    bus.canbus.recv.side_effect = lambda **_: queue.popleft() if queue else None


def test_handshake_decodes_response_not_enable_packet():
    bus = make_bus()
    respond_on_send(bus, frame([1, 0x80, 0, 0x80, 0, 0, 25, 26]))
    bus._handshake()
    assert abs(bus._last_known_states["joint"]["position"]) < 0.02
    assert bus._last_known_states["joint"]["temp_mos"] == 25


@pytest.mark.parametrize(
    "payload", [None, [1], [0x21, 128, 0, 128, 0, 0, 25, 25], [2, 128, 0, 128, 0, 0, 25, 25]]
)
def test_strict_read_refuses_missing_malformed_faulted_or_wrong_id(payload):
    bus = make_bus()
    respond_on_send(bus, frame(payload) if payload is not None else None)
    with pytest.raises(ConnectionError):
        bus.sync_read_all_states(strict=True)


def test_strict_read_discards_stale_queued_feedback():
    bus = make_bus()
    queue = deque([frame([1, 128, 0, 128, 0, 0, 25, 25])])
    bus.canbus.recv.side_effect = lambda **_: queue.popleft() if queue else None
    with pytest.raises(ConnectionError, match="Missing fresh"):
        bus.sync_read_all_states(strict=True)


def test_legacy_read_retains_fallback():
    bus = make_bus()
    respond_on_send(bus, None)
    assert bus.sync_read_all_states()["joint"]["position"] == 0


@pytest.mark.parametrize(
    "bad", [(1, 1, float("nan"), 0, 0), (501, 1, 0, 0, 0), (1, 6, 0, 0, 0), (1, 1, 0, 0, 11)]
)
def test_mit_validation_happens_before_any_transmission(bad):
    bus = make_bus()
    with pytest.raises(ValueError):
        bus.sync_write_mit({"joint": bad})
    bus.canbus.send.assert_not_called()


def test_mit_packet_encodes_degrees_as_radians():
    bus = make_bus()
    respond_on_send(bus, None)
    bus.sync_write_mit({"joint": (20, 0.5, 90, 0, 0.2)})
    message = bus.canbus.send.call_args.args[0]
    expected_position = int(((3.141592653589793 / 2 + 12.5) / 25) * 65535)
    assert message.data[:2] == expected_position.to_bytes(2, "big")
    assert not message.is_fd


def test_receive_collects_queued_replies_after_scheduler_pause(monkeypatch):
    bus = make_bus()
    now = [0.0]
    monkeypatch.setattr("lerobot.motors.damiao.damiao.time.monotonic", lambda: now[0])
    replies = deque(can.Message(arbitration_id=i, data=[0] * 8) for i in range(17, 24))

    def receive(**kwargs):
        # All replies arrived while Python was descheduled for 20 ms.
        now[0] = 0.02
        return replies.popleft() if replies else None

    bus.canbus.recv.side_effect = receive
    assert set(bus._recv_all_responses(list(range(17, 24)), timeout=0.01)) == set(range(17, 24))
    assert bus.canbus.recv.call_args.kwargs["timeout"] == 0


def test_receive_deadline_still_rejects_missing_reply(monkeypatch):
    bus = make_bus()
    now = [0.0]
    monkeypatch.setattr("lerobot.motors.damiao.damiao.time.monotonic", lambda: now[0])

    def receive(**kwargs):
        now[0] = 0.02
        return None

    bus.canbus.recv.side_effect = receive
    assert bus._recv_all_responses([17], timeout=0.01) == {}
    assert bus.canbus.recv.call_count == 2


def test_receive_drain_is_bounded_with_unrelated_traffic(monkeypatch):
    bus = make_bus()
    monkeypatch.setattr("lerobot.motors.damiao.damiao.time.monotonic", lambda: 0.0)
    bus.canbus.recv.return_value = can.Message(arbitration_id=99, data=[0] * 8)
    assert bus._recv_all_responses([17], timeout=0) == {}
    assert bus.canbus.recv.call_count == 1024


def test_fault_reports_interface_and_communication_loss():
    bus = make_bus()
    respond_on_send(bus, frame([0xD1, 128, 0, 128, 0, 0, 25, 25]))
    with pytest.raises(ConnectionError, match="virtual: Motor joint.*0xD.*communication lost"):
        bus.sync_read_all_states(strict=True)

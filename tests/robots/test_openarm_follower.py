"""OpenArm configuration failure handling, without physical hardware."""

from unittest.mock import MagicMock, call
from uuid import uuid4

import pytest

import lerobot.motors.damiao.damiao as damiao_module
from lerobot.motors import Motor
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import CAN_CMD_DISABLE, CAN_CMD_ENABLE, CAN_PARAM_ID, ControlMode
from lerobot.robots.openarm_follower.openarm_follower import OpenArmFollower


@pytest.mark.parametrize("missing_ack", [None, 1, 2])
def test_configure_motor_commands(monkeypatch: pytest.MonkeyPatch, missing_ack: int | None) -> None:
    can = pytest.importorskip("can")
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
    # Exercise the real configure method without cameras, calibration files or hardware discovery.
    follower = object.__new__(OpenArmFollower)
    follower.bus = bus
    sent: list[tuple[int, bytes]] = []
    monkeypatch.setattr(damiao_module, "LONG_TIMEOUT_SEC", 0.01)
    try:
        bus.connect(handshake=False)
        original_send = bus.canbus.send

        def send(msg: can.Message) -> None:
            original_send(msg)
            sent.append((msg.arbitration_id, bytes(msg.data)))
            if msg.arbitration_id == CAN_PARAM_ID:
                motor_id = msg.data[0]
                if motor_id == missing_ack:
                    return
                payload = msg.data
            else:
                motor_id = msg.arbitration_id
                payload = [0x10 | motor_id, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF, 30, 30]
            peer.send(can.Message(arbitration_id=0x10 + motor_id, data=payload, is_extended_id=False))

        monkeypatch.setattr(bus.canbus, "send", send)
        if missing_ack is None:
            follower.configure()
            assert all(mode == ControlMode.MIT for mode in bus.operation_mode.values())
        else:
            with pytest.raises(TimeoutError, match=f"joint_{missing_ack}.*CTRL_MODE=MIT"):
                follower.configure()
            assert bus.operation_mode[f"joint_{missing_ack}"] is None

        commands = [(motor_id, data[-1]) for motor_id, data in sent if data[:7] == b"\xff" * 7]
        initial_disable = [(1, CAN_CMD_DISABLE), (2, CAN_CMD_DISABLE)]
        enabled = [(motor_id, CAN_CMD_ENABLE) for motor_id in range(1, missing_ack or 3)]
        expected = initial_disable + enabled
        if missing_ack is not None:
            expected += initial_disable
        assert commands == expected
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        peer.shutdown()


def test_configure_preserves_error_when_disable_fails(caplog: pytest.LogCaptureFixture) -> None:
    follower = object.__new__(OpenArmFollower)
    bus = MagicMock(spec=DamiaoMotorsBus)
    follower.bus = bus
    original_error = TimeoutError("joint_2 did not acknowledge CTRL_MODE=MIT")
    bus.configure_motors.side_effect = original_error
    bus.disable_torque.side_effect = [None, OSError("CAN transport unavailable")]

    with pytest.raises(TimeoutError) as caught:
        follower.configure()

    assert caught.value is original_error
    assert bus.mock_calls == [call.disable_torque(), call.configure_motors(), call.disable_torque()]
    assert "Failed to disable torque after motor configuration failed" in caplog.text
    assert "CAN transport unavailable" in caplog.text

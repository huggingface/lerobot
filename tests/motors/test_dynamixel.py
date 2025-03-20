import sys
from unittest.mock import patch

import dynamixel_sdk as dxl
import pytest

from lerobot.common.motors.dynamixel.dynamixel import DynamixelMotorsBus
from tests.mocks.mock_dynamixel import MockInstructionPacket, MockMotors, MockPortHandler


@pytest.fixture(autouse=True)
def patch_port_handler():
    if sys.platform == "darwin":
        with patch.object(dxl, "PortHandler", MockPortHandler):
            yield
    else:
        yield


@pytest.mark.skipif(sys.platform != "darwin", reason=f"No patching needed on {sys.platform=}")
def test_autouse_patch():
    """Ensures that the autouse fixture correctly patches dxl.PortHandler with MockPortHandler."""
    assert dxl.PortHandler is MockPortHandler


@pytest.mark.parametrize(
    "value, n_bytes, expected",
    [
        (0x12,       1, [0x12]),                    # Single byte
        (0x1234,     2, [0x34, 0x12]),              # Two bytes
        (0x12345678, 4, [0x78, 0x56, 0x34, 0x12]),  # Four bytes
        (0,          1, [0x00]),                    # Zero with 1 byte
        (0,          2, [0x00, 0x00]),              # Zero with 2 bytes
        (0,          4, [0x00, 0x00, 0x00, 0x00]),  # Zero with 4 bytes
        (255,        1, [0xFF]),                    # Max single byte
        (65535,      2, [0xFF, 0xFF]),              # Max two bytes
        (4294967295, 4, [0xFF, 0xFF, 0xFF, 0xFF]),  # Max four bytes
    ],
)  # fmt: skip
def test_split_int_bytes(value, n_bytes, expected):
    assert DynamixelMotorsBus.split_int_bytes(value, n_bytes) == expected


def test_split_int_bytes_invalid_n_bytes():
    with pytest.raises(NotImplementedError):
        DynamixelMotorsBus.split_int_bytes(100, 3)


def test_split_int_bytes_negative_numbers():
    with pytest.raises(ValueError):
        neg = DynamixelMotorsBus.split_int_bytes(-1, 1)
        print(neg)


def test_split_int_bytes_large_number():
    with pytest.raises(ValueError):
        DynamixelMotorsBus.split_int_bytes(2**32, 4)  # 4-byte max is 0xFFFFFFFF


def test_abc_implementation():
    # Instantiation should raise an error if the class doesn't implements abstract methods/properties
    DynamixelMotorsBus(port="/dev/dummy-port", motors={"dummy": (1, "xl330-m077")})


@pytest.mark.parametrize(
    "motors",
    [
        None,
        [1, 2, 3],
        ["dummy_1", "dummy_2", "dummy_3"],
        [1, "dummy_2", 3],  # Mixed
    ],
)
def test_read_all_motors(motors):
    mock_motors = MockMotors([1, 2, 3])
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors={
            "dummy_1": (1, "xl330-m077"),
            "dummy_2": (2, "xl330-m077"),
            "dummy_3": (3, "xl330-m077"),
        },
    )
    motors_bus.connect()

    pos_dict = motors_bus.read("Present_Position", motors=motors)

    assert mock_motors.stubs["SyncRead_Present_Position_all"].called
    assert len(pos_dict) == 3
    assert all(pos >= 0 and pos <= 4095 for pos in pos_dict.values())


@pytest.mark.parametrize("idx", [1, 2, 3])
def test_read_single_motor_name(idx):
    mock_motors = MockMotors([1, 2, 3])
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors={
            "dummy_1": (1, "xl330-m077"),
            "dummy_2": (2, "xl330-m077"),
            "dummy_3": (3, "xl330-m077"),
        },
    )
    motors_bus.connect()

    pos_dict = motors_bus.read("Present_Position", f"dummy_{idx}")

    assert mock_motors.stubs[f"SyncRead_Present_Position_{idx}"].called
    assert len(pos_dict) == 1
    assert all(pos >= 0 and pos <= 4095 for pos in pos_dict.values())


@pytest.mark.parametrize("idx", [1, 2, 3])
def test_read_single_motor_id(idx):
    mock_motors = MockMotors([1, 2, 3])
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors={
            "dummy_1": (1, "xl330-m077"),
            "dummy_2": (2, "xl330-m077"),
            "dummy_3": (3, "xl330-m077"),
        },
    )
    motors_bus.connect()

    pos_dict = motors_bus.read("Present_Position", idx)

    assert mock_motors.stubs[f"SyncRead_Present_Position_{idx}"].called
    assert len(pos_dict) == 1
    assert all(pos >= 0 and pos <= 4095 for pos in pos_dict.values())


@pytest.mark.parametrize(
    "num_retry, num_invalid_try",
    [
        [1, 2],
        [2, 3],
        [3, 2],
        [2, 1],
    ],
)
def test_read_num_retry(num_retry, num_invalid_try):
    mock_motors = MockMotors([1, 2, 3], default_stubs=None)
    address, length = mock_motors.ctrl_table["Present_Position"]
    receive_bytes = MockInstructionPacket.sync_read([1], address, length)
    send_fn = mock_motors._create_present_pos_send_fn(
        [1], "Present_Position", num_invalid_try=num_invalid_try
    )
    mock_motors.stub(name="num_retry", receive_bytes=receive_bytes, send_fn=send_fn)

    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors={
            "dummy_1": (1, "xl330-m077"),
            "dummy_2": (2, "xl330-m077"),
            "dummy_3": (3, "xl330-m077"),
        },
    )
    motors_bus.connect()

    if num_retry >= num_invalid_try:
        pos_dict = motors_bus.read("Present_Position", 1, num_retry=num_retry)
        assert len(pos_dict) == 1
        assert all(pos >= 0 and pos <= 4095 for pos in pos_dict.values())
    else:
        with pytest.raises(ConnectionError):
            _ = motors_bus.read("Present_Position", 1, num_retry=num_retry)

    assert mock_motors.stubs["num_retry"].calls == num_retry

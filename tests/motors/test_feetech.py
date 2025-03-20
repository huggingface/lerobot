import sys
from unittest.mock import patch

import pytest
import scservo_sdk as scs

from lerobot.common.motors.feetech import FeetechMotorsBus
from tests.mocks.mock_feetech import MockMotors, MockPortHandler


@pytest.fixture(autouse=True)
def patch_port_handler():
    if sys.platform == "darwin":
        with patch.object(scs, "PortHandler", MockPortHandler):
            yield
    else:
        yield


@pytest.mark.skipif(sys.platform != "darwin", reason=f"No patching needed on {sys.platform=}")
def test_autouse_patch():
    """Ensures that the autouse fixture correctly patches scs.PortHandler with MockPortHandler."""
    assert scs.PortHandler is MockPortHandler


@pytest.mark.parametrize(
    "value, n_bytes, expected",
    [
        (0x12,       1, [0x12]),
        (0x1234,     2, [0x34, 0x12]),
        (0x12345678, 4, [0x78, 0x56, 0x34, 0x12]),
        (0,          1, [0x00]),
        (0,          2, [0x00, 0x00]),
        (0,          4, [0x00, 0x00, 0x00, 0x00]),
        (255,        1, [0xFF]),
        (65535,      2, [0xFF, 0xFF]),
        (4294967295, 4, [0xFF, 0xFF, 0xFF, 0xFF]),
    ],
    ids=[
        "1 byte",
        "2 bytes",
        "4 bytes",
        "0 with 1 byte",
        "0 with 2 bytes",
        "0 with 4 bytes",
        "max single byte",
        "max two bytes",
        "max four bytes",
    ],
)  # fmt: skip
def test_split_int_bytes(value, n_bytes, expected):
    assert FeetechMotorsBus.split_int_bytes(value, n_bytes) == expected


def test_split_int_bytes_invalid_n_bytes():
    with pytest.raises(NotImplementedError):
        FeetechMotorsBus.split_int_bytes(100, 3)


def test_split_int_bytes_negative_numbers():
    with pytest.raises(ValueError):
        neg = FeetechMotorsBus.split_int_bytes(-1, 1)
        print(neg)


def test_split_int_bytes_large_number():
    with pytest.raises(ValueError):
        FeetechMotorsBus.split_int_bytes(2**32, 4)  # 4-byte max is 0xFFFFFFFF


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    FeetechMotorsBus(port="/dev/dummy-port", motors={"dummy": (1, "sts3215")})


@pytest.mark.parametrize(
    "motors",
    [
        None,
        [1, 2, 3],
        ["dummy_1", "dummy_2", "dummy_3"],
        [1, "dummy_2", 3],
    ],
    ids=["None", "by ids", "by names", "mixed"],
)
def test_read_all_motors(motors):
    mock_motors = MockMotors([1, 2, 3])
    positions = [1337, 42, 4016]
    mock_motors.build_all_motors_stub("Present_Position", return_values=positions)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors={
            "dummy_1": (1, "sts3215"),
            "dummy_2": (2, "sts3215"),
            "dummy_3": (3, "sts3215"),
        },
    )
    motors_bus.connect()

    pos_dict = motors_bus.read("Present_Position", motors=motors)

    assert mock_motors.stubs["SyncRead_Present_Position_all"].called
    assert all(returned_pos == pos for returned_pos, pos in zip(pos_dict.values(), positions, strict=True))
    assert set(pos_dict) == {"dummy_1", "dummy_2", "dummy_3"}
    assert all(pos >= 0 and pos <= 4095 for pos in pos_dict.values())


@pytest.mark.parametrize(
    "idx, pos",
    [
        [1, 1337],
        [2, 42],
        [3, 4016],
    ],
)
def test_read_single_motor_by_name(idx, pos):
    mock_motors = MockMotors([1, 2, 3])
    mock_motors.build_single_motor_stubs("Present_Position", return_value=pos)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors={
            "dummy_1": (1, "sts3215"),
            "dummy_2": (2, "sts3215"),
            "dummy_3": (3, "sts3215"),
        },
    )
    motors_bus.connect()

    pos_dict = motors_bus.read("Present_Position", f"dummy_{idx}")

    assert mock_motors.stubs[f"SyncRead_Present_Position_{idx}"].called
    assert pos_dict == {f"dummy_{idx}": pos}
    assert all(pos >= 0 and pos <= 4095 for pos in pos_dict.values())


@pytest.mark.parametrize(
    "idx, pos",
    [
        [1, 1337],
        [2, 42],
        [3, 4016],
    ],
)
def test_read_single_motor_by_id(idx, pos):
    mock_motors = MockMotors([1, 2, 3])
    mock_motors.build_single_motor_stubs("Present_Position", return_value=pos)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors={
            "dummy_1": (1, "sts3215"),
            "dummy_2": (2, "sts3215"),
            "dummy_3": (3, "sts3215"),
        },
    )
    motors_bus.connect()

    pos_dict = motors_bus.read("Present_Position", idx)

    assert mock_motors.stubs[f"SyncRead_Present_Position_{idx}"].called
    assert pos_dict == {f"dummy_{idx}": pos}
    assert all(pos >= 0 and pos <= 4095 for pos in pos_dict.values())


@pytest.mark.parametrize(
    "num_retry, num_invalid_try, pos",
    [
        [1, 2, 1337],
        [2, 3, 42],
        [3, 2, 4016],
        [2, 1, 999],
    ],
)
def test_read_num_retry(num_retry, num_invalid_try, pos):
    mock_motors = MockMotors([1, 2, 3])
    mock_motors.build_single_motor_stubs(
        "Present_Position", return_value=pos, num_invalid_try=num_invalid_try
    )
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors={
            "dummy_1": (1, "sts3215"),
            "dummy_2": (2, "sts3215"),
            "dummy_3": (3, "sts3215"),
        },
    )
    motors_bus.connect()

    if num_retry >= num_invalid_try:
        pos_dict = motors_bus.read("Present_Position", 1, num_retry=num_retry)
        assert pos_dict == {"dummy_1": pos}
        assert all(pos >= 0 and pos <= 4095 for pos in pos_dict.values())
    else:
        with pytest.raises(ConnectionError):
            _ = motors_bus.read("Present_Position", 1, num_retry=num_retry)

    assert mock_motors.stubs["SyncRead_Present_Position_1"].calls == num_retry

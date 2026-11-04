import pytest

from lerobot.rollout.so101_gripper_feedback import (
    SO101GripperFeedback,
    parse_sts3215_feedback,
)


@pytest.fixture
def normal_feedback():
    return {
        "present_load": 300,
        "present_voltage": 120,
        "present_temperature": 35,
        "present_current": 100,
        "status": 0,
    }


def test_normal_feedback_parsing(normal_feedback):
    result = parse_sts3215_feedback(normal_feedback)

    assert isinstance(result, SO101GripperFeedback)
    assert result.load_value == 300
    assert result.load_magnitude == 300
    assert result.load_direction == "positive"
    assert result.voltage_v == 12.0
    assert result.temperature_c == 35
    assert result.current_raw == 100
    assert result.status_raw == 0


def test_negative_load():
    raw = {
        "present_load": -300,
        "present_voltage": 120,
        "present_temperature": 35,
        "present_current": 100,
        "status": 0,
    }

    result = parse_sts3215_feedback(raw)

    assert result.load_value == -300
    assert result.load_magnitude == 300
    assert result.load_direction == "negative"


def test_zero_load():
    raw = {
        "present_load": 0,
        "present_voltage": 120,
        "present_temperature": 35,
        "present_current": 100,
        "status": 0,
    }

    result = parse_sts3215_feedback(raw)

    assert result.load_magnitude == 0
    assert result.load_direction == "zero"


@pytest.mark.parametrize("value", [-1023, -1, 1, 1023])
def test_load_signed_boundaries(normal_feedback, value):
    normal_feedback["present_load"] = value

    result = parse_sts3215_feedback(normal_feedback)

    assert result.load_value == value
    assert result.load_magnitude == abs(value)


@pytest.mark.parametrize("value", [-1024, 1024, 1324, 65535])
def test_invalid_load_range_rejected(normal_feedback, value):
    normal_feedback["present_load"] = value

    with pytest.raises(ValueError):
        parse_sts3215_feedback(normal_feedback)


def test_status_preserved_without_unverified_mapping(normal_feedback):
    normal_feedback["status"] = 0xFF

    result = parse_sts3215_feedback(normal_feedback)

    assert result.status_raw == 0xFF


def test_current_preserved_without_unverified_conversion(normal_feedback):
    normal_feedback["present_current"] = 65535

    result = parse_sts3215_feedback(normal_feedback)

    assert result.current_raw == 65535


@pytest.mark.parametrize(
    "field",
    [
        "present_load",
        "present_voltage",
        "present_temperature",
        "present_current",
        "status",
    ],
)
def test_missing_register_rejected(normal_feedback, field):
    del normal_feedback[field]

    with pytest.raises(ValueError, match="Missing feedback"):
        parse_sts3215_feedback(normal_feedback)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("present_voltage", 256),
        ("present_temperature", -1),
        ("present_current", 65536),
        ("status", 256),
    ],
)
def test_out_of_range_values_rejected(normal_feedback, field, value):
    normal_feedback[field] = value

    with pytest.raises(ValueError):
        parse_sts3215_feedback(normal_feedback)


@pytest.mark.parametrize(
    "value",
    [True, False, 1.5, "100", None],
)
def test_invalid_value_types_rejected(normal_feedback, value):
    normal_feedback["present_current"] = value

    with pytest.raises(ValueError):
        parse_sts3215_feedback(normal_feedback)


def test_input_is_not_modified(normal_feedback):
    original = normal_feedback.copy()

    parse_sts3215_feedback(normal_feedback)

    assert normal_feedback == original

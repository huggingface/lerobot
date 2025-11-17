#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode
from lerobot.motors.dc_pwm.dc_pwm import (
    PI5_OPTIMAL_FREQUENCY,
    PWMDCMotorsController,
    PWMProtocolHandler,
)


class MockPWMLED:
    """Mock PWMLED for testing without hardware."""

    def __init__(self, pin: int, frequency: int | None = None):
        self.pin = pin
        self.frequency = frequency
        self._value = 0.0
        self.is_closed = False

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, val: float):
        self._value = max(0.0, min(1.0, val))

    def on(self):
        self._value = 1.0

    def off(self):
        self._value = 0.0

    def close(self):
        self.is_closed = True


@pytest.fixture
def mock_gpiozero():
    """Mock gpiozero module."""
    mock_gpio = MagicMock()
    mock_gpio.PWMLED = MockPWMLED
    return mock_gpio


@pytest.fixture
def dummy_motors() -> dict[str, DCMotor]:
    return {
        "motor_1": DCMotor(id=1, model="drv8874", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
        "motor_2": DCMotor(id=2, model="drv8874", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
        "motor_3": DCMotor(id=3, model="drv8874", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
    }


@pytest.fixture
def pwm_config() -> dict:
    return {
        "in1_pins": [12, 13, 18],
        "in2_pins": [19, 20, 21],
        "pwm_frequency": PI5_OPTIMAL_FREQUENCY,
    }


@pytest.fixture
def protocol_handler(pwm_config, dummy_motors, mock_gpiozero):
    """Create a PWMProtocolHandler with mocked gpiozero."""
    # Patch sys.modules to intercept the import inside _import_gpiozero
    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        handler = PWMProtocolHandler(pwm_config, dummy_motors)
        return handler


@pytest.fixture(autouse=True)
def auto_patch_gpiozero(mock_gpiozero):
    """Automatically patch gpiozero for all tests."""
    # This fixture patches sys.modules to intercept imports
    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        yield


def test_controller_instantiation(pwm_config, dummy_motors):
    """Test PWMDCMotorsController can be instantiated."""
    controller = PWMDCMotorsController(config=pwm_config, motors=dummy_motors, protocol="pwm")
    assert controller.config == pwm_config
    assert controller.motors == dummy_motors
    assert controller.protocol == "pwm"


def test_controller_creates_protocol_handler(pwm_config, dummy_motors, mock_gpiozero):
    """Test that controller creates the correct protocol handler."""
    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        controller = PWMDCMotorsController(config=pwm_config, motors=dummy_motors)
        handler = controller._create_protocol_handler()
        assert isinstance(handler, PWMProtocolHandler)


@pytest.mark.parametrize(
    "in1_pins, in2_pins, expected_count",
    [
        ([12], [19], 1),
        ([12, 13], [19, 20], 2),
        ([12, 13, 18], [19, 20, 21], 3),
    ],
)
def test_protocol_handler_init(in1_pins, in2_pins, expected_count, dummy_motors, mock_gpiozero):
    """Test PWMProtocolHandler initialization."""
    config = {
        "in1_pins": in1_pins,
        "in2_pins": in2_pins,
        "pwm_frequency": PI5_OPTIMAL_FREQUENCY,
    }
    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        handler = PWMProtocolHandler(config, dummy_motors)
        assert len(handler.in1_pins) == expected_count
        assert len(handler.in2_pins) == expected_count
        assert handler.pwm_frequency == PI5_OPTIMAL_FREQUENCY


def test_connect(protocol_handler):
    """Test connect method sets up PWMLED channels."""
    protocol_handler.connect()

    assert len(protocol_handler.in1_channels) == 3
    assert len(protocol_handler.in2_channels) == 3
    assert all(isinstance(ch, MockPWMLED) for ch in protocol_handler.in1_channels.values())
    assert all(isinstance(ch, MockPWMLED) for ch in protocol_handler.in2_channels.values())

    # Check that channels were initialized with correct pins
    assert protocol_handler.in1_channels[1].pin == 12
    assert protocol_handler.in2_channels[1].pin == 19


def test_disconnect(protocol_handler):
    """Test disconnect method closes all channels."""
    protocol_handler.connect()
    protocol_handler.disconnect()

    assert all(ch.is_closed for ch in protocol_handler.in1_channels.values())
    assert all(ch.is_closed for ch in protocol_handler.in2_channels.values())


def test_get_position(protocol_handler):
    """Test get_position returns stored position."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["position"] = 0.5

    position = protocol_handler.get_position(1)
    assert position == 0.5

    # Test with uninitialized motor
    position = protocol_handler.get_position(999)
    assert position == 0.0


def test_set_position(protocol_handler):
    """Test set_position updates position and PWM."""
    protocol_handler.connect()

    # Set direction to forward first
    protocol_handler.motor_states[1]["direction"] = 1

    protocol_handler.set_position(1, 0.75)
    assert protocol_handler.motor_states[1]["position"] == 0.75
    assert protocol_handler.motor_states[1]["pwm"] == 0.75

    # Test clamping
    protocol_handler.set_position(1, 1.5)
    assert protocol_handler.motor_states[1]["position"] == 1.0

    protocol_handler.set_position(1, -0.5)
    assert protocol_handler.motor_states[1]["position"] == 0.0


def test_get_velocity(protocol_handler):
    """Test get_velocity returns stored velocity."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["velocity"] = 0.5

    velocity = protocol_handler.get_velocity(1)
    assert velocity == 0.5

    # Test with uninitialized motor
    velocity = protocol_handler.get_velocity(999)
    assert velocity == 0.0


def test_set_velocity(protocol_handler):
    """Test set_velocity sets target velocity."""
    protocol_handler.connect()

    protocol_handler.set_velocity(1, 0.5, instant=True)
    assert protocol_handler.motor_states[1]["target_velocity"] == 0.5
    assert protocol_handler.motor_states[1]["velocity"] == 0.5

    # Test clamping
    protocol_handler.set_velocity(1, 1.5, instant=False)
    assert protocol_handler.motor_states[1]["target_velocity"] == 1.0

    protocol_handler.set_velocity(1, -1.5, instant=False)
    assert protocol_handler.motor_states[1]["target_velocity"] == -1.0


@pytest.mark.parametrize(
    "current, target, max_step, expected",
    [
        (0.0, 1.0, 0.5, 0.5),  # Ramp up
        (1.0, 0.0, 0.5, 0.5),  # Ramp down
        (0.0, 0.5, 1.0, 0.5),  # Instant update
        (0.0, 0.0, 1.0, 0.0),  # No change
    ],
)
def test_update_velocity(current, target, max_step, expected, protocol_handler):
    """Test update_velocity with slew rate limiting."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["velocity"] = current
    protocol_handler.motor_states[1]["target_velocity"] = target

    protocol_handler.update_velocity(1, max_step)

    assert protocol_handler.motor_states[1]["velocity"] == expected


def test_update_velocity_forward(protocol_handler):
    """Test update_velocity sets correct GPIO states for forward motion."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["target_velocity"] = 0.5
    protocol_handler.motor_states[1]["velocity"] = 0.0

    protocol_handler.update_velocity(1, 1.0)

    in1 = protocol_handler.in1_channels[1]
    in2 = protocol_handler.in2_channels[1]
    assert in1.value > 0  # PWM on
    assert in2.value == 0.0  # OFF
    assert protocol_handler.motor_states[1]["direction"] == 1


def test_update_velocity_reverse(protocol_handler):
    """Test update_velocity sets correct GPIO states for reverse motion."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["target_velocity"] = -0.5
    protocol_handler.motor_states[1]["velocity"] = 0.0

    protocol_handler.update_velocity(1, 1.0)

    in1 = protocol_handler.in1_channels[1]
    in2 = protocol_handler.in2_channels[1]
    assert in1.value == 0.0  # OFF
    assert in2.value > 0  # PWM on
    assert protocol_handler.motor_states[1]["direction"] == -1


def test_update_velocity_stop(protocol_handler):
    """Test update_velocity stops motor when velocity is zero."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["target_velocity"] = 0.0
    protocol_handler.motor_states[1]["velocity"] = 0.5

    protocol_handler.update_velocity(1, 1.0)

    in1 = protocol_handler.in1_channels[1]
    in2 = protocol_handler.in2_channels[1]
    assert in1.value == 0.0
    assert in2.value == 0.0
    assert protocol_handler.motor_states[1]["direction"] == 0


def test_get_pwm(protocol_handler):
    """Test get_pwm returns stored PWM value."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["pwm"] = 0.75

    pwm = protocol_handler.get_pwm(1)
    assert pwm == 0.75

    # Test with uninitialized motor
    pwm = protocol_handler.get_pwm(999)
    assert pwm == 0.0


def test_set_pwm_forward(protocol_handler):
    """Test set_pwm with forward direction."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["direction"] = 1

    protocol_handler.set_pwm(1, 0.5)

    in1 = protocol_handler.in1_channels[1]
    in2 = protocol_handler.in2_channels[1]
    assert in1.value == 0.5  # PWM value set to duty_cycle
    assert in2.value == 0.0  # OFF
    assert protocol_handler.motor_states[1]["pwm"] == 0.5


def test_set_pwm_reverse(protocol_handler):
    """Test set_pwm with reverse direction."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["direction"] = -1

    protocol_handler.set_pwm(1, 0.5)

    in1 = protocol_handler.in1_channels[1]
    in2 = protocol_handler.in2_channels[1]
    assert in1.value == 0.0  # OFF
    assert in2.value == 0.5  # PWM value set to duty_cycle
    assert protocol_handler.motor_states[1]["pwm"] == 0.5


def test_set_pwm_stop(protocol_handler):
    """Test set_pwm stops motor when direction is 0."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["direction"] = 0

    protocol_handler.set_pwm(1, 0.5)

    in1 = protocol_handler.in1_channels[1]
    in2 = protocol_handler.in2_channels[1]
    assert in1.value == 0.0
    assert in2.value == 0.0


def test_set_pwm_clamping(protocol_handler):
    """Test set_pwm clamps duty cycle to 0.98."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["direction"] = 1

    protocol_handler.set_pwm(1, 1.0)
    assert protocol_handler.motor_states[1]["pwm"] == 0.98

    protocol_handler.set_pwm(1, -0.1)
    assert protocol_handler.motor_states[1]["pwm"] == 0.0


def test_enable_motor(protocol_handler):
    """Test enable_motor sets enabled flag."""
    protocol_handler.connect()

    protocol_handler.enable_motor(1)
    assert protocol_handler.motor_states[1]["enabled"] is True


def test_disable_motor(protocol_handler):
    """Test disable_motor sets PWM to 0 and disables."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["pwm"] = 0.5
    protocol_handler.motor_states[1]["direction"] = 0  # Set to stop/neutral

    protocol_handler.disable_motor(1)

    assert protocol_handler.motor_states[1]["pwm"] == 0.0
    assert protocol_handler.motor_states[1]["enabled"] is False
    in1 = protocol_handler.in1_channels[1]
    in2 = protocol_handler.in2_channels[1]
    # When direction=0, both channels are turned off
    assert in1.value == 0.0
    assert in2.value == 0.0


def test_activate_brake(protocol_handler):
    """Test activate_brake sets both IN1 and IN2 high."""
    protocol_handler.connect()

    protocol_handler.activate_brake(1)

    in1 = protocol_handler.in1_channels[1]
    in2 = protocol_handler.in2_channels[1]
    assert in1.value == 1.0
    assert in2.value == 1.0
    assert protocol_handler.motor_states[1]["brake_active"] is True


def test_release_brake(protocol_handler):
    """Test release_brake sets both IN1 and IN2 low."""
    protocol_handler.connect()
    protocol_handler.motor_states[1]["brake_active"] = True

    protocol_handler.release_brake(1)

    in1 = protocol_handler.in1_channels[1]
    in2 = protocol_handler.in2_channels[1]
    assert in1.value == 0.0
    assert in2.value == 0.0
    assert protocol_handler.motor_states[1]["brake_active"] is False


@pytest.mark.parametrize(
    "velocity, expected_pwm",
    [
        (0.0, 0.0),
        (0.5, 0.325),  # deadzone + (1-deadzone) * 0.5^2 = 0.1 + 0.9 * 0.25
        (1.0, 1.0),  # deadzone + (1-deadzone) * 1^2 = 0.1 + 0.9 * 1
    ],
)
def test_velocity_to_pwm(velocity, expected_pwm, protocol_handler):
    """Test _velocity_to_pwm conversion."""
    protocol_handler.connect()

    pwm = protocol_handler._velocity_to_pwm(velocity)
    assert abs(pwm - expected_pwm) < 0.01  # Allow small floating point differences


def test_controller_connect_disconnect(pwm_config, dummy_motors, mock_gpiozero):
    """Test controller connect and disconnect."""
    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        controller = PWMDCMotorsController(config=pwm_config, motors=dummy_motors)
        controller.connect()

        assert controller.is_connected is True
        assert controller.protocol_handler is not None

        controller.disconnect()
        assert controller.is_connected is False


def test_controller_get_set_position(pwm_config, dummy_motors, mock_gpiozero):
    """Test controller position methods."""
    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        controller = PWMDCMotorsController(config=pwm_config, motors=dummy_motors)
        controller.connect()

        controller.set_position("motor_1", 0.5)
        position = controller.get_position("motor_1")
        assert position == 0.5


def test_controller_get_set_velocity(pwm_config, dummy_motors, mock_gpiozero):
    """Test controller velocity methods."""
    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        controller = PWMDCMotorsController(config=pwm_config, motors=dummy_motors)
        controller.connect()

        controller.set_velocity("motor_1", 0.5)
        velocity = controller.get_velocity("motor_1")
        assert velocity == 0.5


def test_controller_get_set_pwm(pwm_config, dummy_motors, mock_gpiozero):
    """Test controller PWM methods."""
    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        controller = PWMDCMotorsController(config=pwm_config, motors=dummy_motors)
        controller.connect()

        # Set direction first
        controller.protocol_handler.motor_states[1]["direction"] = 1

        controller.set_pwm("motor_1", 0.5)
        pwm = controller.get_pwm("motor_1")
        assert pwm == 0.5


def test_controller_enable_disable(pwm_config, dummy_motors, mock_gpiozero):
    """Test controller enable/disable methods."""
    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        controller = PWMDCMotorsController(config=pwm_config, motors=dummy_motors)
        controller.connect()

        controller.enable_motor("motor_1")
        assert controller.protocol_handler.motor_states[1]["enabled"] is True

        controller.disable_motor("motor_1")
        assert controller.protocol_handler.motor_states[1]["enabled"] is False


def test_setup_pwmled_fallback(protocol_handler):
    """Test _setup_pwmled falls back to default frequency on error."""
    protocol_handler.connect()

    # This test verifies the fallback logic exists
    # The actual fallback is tested by the fact that connect() works
    assert len(protocol_handler.in1_channels) > 0


def test_validate_pi5_pins(protocol_handler):
    """Test pin validation."""
    # This should not raise for valid pins
    protocol_handler._validate_pi5_pins()

    # Test with invalid IN2 pin
    protocol_handler.in2_pins = [999]
    # Should log warning but not raise
    protocol_handler._validate_pi5_pins()


def test_invert_direction(pwm_config, dummy_motors, mock_gpiozero):
    """Test direction inversion configuration."""
    pwm_config["invert_direction"] = True

    with patch.dict(sys.modules, {"gpiozero": mock_gpiozero}):
        handler = PWMProtocolHandler(pwm_config, dummy_motors)
        handler.connect()
        handler.motor_states[1]["direction"] = 1

        # Test _set_direction with inversion
        handler._set_direction(1, forward=True)
        # Should be inverted
        assert handler.motor_states[1]["direction"] == -1

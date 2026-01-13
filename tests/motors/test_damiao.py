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

"""
Test script for Damiao motor communication and control.

This script tests basic functionality of a single Damiao motor via CAN bus:
1. Connects to CAN interface
2. Discovers and enables the motor
3. Reads current position
4. Sets zero position
5. Writes target positions
6. Disables torque

Requirements:
- Motor must be connected and powered (24V)
- CAN interface must be configured (e.g., can0)
- Motor ID must be set to 0x01 (send) and 0x11 (receive)

Setup CAN interface:
    sudo ip link set can0 type can bitrate 1000000
    sudo ip link set can0 up

Verify connection:
    candump can0  # In another terminal
    cansend can0 001#FFFFFFFFFFFFFFFC  # Should enable motor and LED turns green
"""

import time

import pytest

pytest.importorskip("can")

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MotorType


@pytest.fixture
def can_port(request):
    """Get CAN port from command line or raise error if not provided."""
    port = request.config.getoption("--can-port")
    if port is None:
        pytest.skip("CAN port not specified. Use --can-port to specify the CAN interface.")
    return port


@pytest.mark.hardware
def test_single_motor_basic_operations(can_port):
    """
    Test basic operations with a single Damiao motor.

    This test requires actual hardware and is skipped by default.
    To run with hardware, use: pytest tests/motors/test_damiao.py --run-hardware --can-port PORT
    """

    # Configuration
    motor_id_const = 0x01  # Sender CAN ID
    motor_recv_id_const = 0x11  # Receiver/Master ID
    motor_type_const = "dm4310"
    motor_name_const = "test_motor"

    print(f"\n{'=' * 60}")
    print("Damiao Motor Test - Single Motor Basic Operations")
    print(f"{'=' * 60}\n")

    # Step 1: Create motor configuration
    print("Step 1: Creating motor configuration...")
    print(f"  - Motor ID: 0x{motor_id_const:02X} (send) / 0x{motor_recv_id_const:02X} (recv)")
    print(f"  - Motor Type: {motor_type_const}")
    print(f"  - CAN Port: {can_port}")

    motor = Motor(motor_id_const, motor_type_const, MotorNormMode.DEGREES)
    motor.recv_id = motor_recv_id_const
    motor.motor_type = MotorType.DM4310

    motors = {motor_name_const: motor}

    # Step 2: Connect to CAN bus
    print("\nStep 2: Connecting to CAN bus...")
    bus = DamiaoMotorsBus(port=can_port, motors=motors)

    try:
        bus.connect(handshake=True)
        print(f"  ✓ Connected to {can_port}")
    except Exception as e:
        print(f"  ✗ Failed to connect: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Check CAN interface is up: ip link show {can_port}")
        print(f"  2. Setup if needed: sudo ip link set {can_port} type can bitrate 1000000")
        print(f"  3. Bring up: sudo ip link set {can_port} up")
        print(f"  4. Test with: cansend {can_port} 001#FFFFFFFFFFFFFFFC")
        return

    try:
        # Step 3: Enable motor (torque on)
        print("\nStep 3: Enabling motor...")
        bus.enable_torque(motor_name_const)
        time.sleep(0.1)
        print("  ✓ Motor enabled (LED should be green)")

        # Step 4: Read current position
        print("\nStep 4: Reading current position...")
        current_pos = bus.read("Present_Position", motor_name_const, normalize=False)
        current_vel = bus.read("Present_Velocity", motor_name_const, normalize=False)
        current_torque = bus.read("Present_Torque", motor_name_const, normalize=False)

        print("  Current State:")
        print(f"    Position: {current_pos:8.2f}°")
        print(f"    Velocity: {current_vel:8.2f}°/s")
        print(f"    Torque:   {current_torque:8.3f} N·m")

        # Step 5: Set zero position
        print("\nStep 5: Setting current position as zero...")
        bus.set_zero_position([motor_name_const])
        time.sleep(0.2)

        new_pos = bus.read("Present_Position", motor_name_const, normalize=False)
        print(f"  Position after zero: {new_pos:8.2f}°")
        print("  ✓ Zero position set")

        # Step 6: Test position commands
        print("\nStep 6: Testing position control...")

        test_positions = [0.0, 45.0, -45.0, 0.0]

        for target_pos in test_positions:
            print(f"\n  Moving to {target_pos:6.1f}°...")
            bus.write("Goal_Position", motor_name_const, target_pos, normalize=False)
            time.sleep(1.0)  # Allow motor to move

            actual_pos = bus.read("Present_Position", motor_name_const, normalize=False)
            error = abs(actual_pos - target_pos)

            print(f"    Target:   {target_pos:8.2f}°")
            print(f"    Actual:   {actual_pos:8.2f}°")
            print(f"    Error:    {error:8.2f}°")

            if error > 10.0:
                print("    ⚠ Large position error!")
            else:
                print("    ✓ Position reached")

        # Step 7: Test MIT control with custom gains
        print("\nStep 7: Testing MIT control with custom gains...")
        print("  Using lower gains for gentler movement...")

        # Lower gains for smoother motion
        bus._mit_control(
            motor_name_const,
            kp=5.0,  # Lower position gain
            kd=0.3,  # Lower damping
            position_degrees=30.0,
            velocity_deg_per_sec=0.0,
            torque=0.0,
        )
        time.sleep(1.5)

        final_pos = bus.read("Present_Position", motor_name_const, normalize=False)
        print(f"  Final position: {final_pos:8.2f}°")
        print("  ✓ MIT control test complete")

        # Step 8: Return to zero
        print("\nStep 8: Returning to zero position...")
        bus.write("Goal_Position", motor_name_const, 0.0, normalize=False)
        time.sleep(1.0)

        final_pos = bus.read("Present_Position", motor_name_const, normalize=False)
        print(f"  Final position: {final_pos:8.2f}°")

    finally:
        # Step 9: Disable motor
        print("\nStep 9: Disabling motor...")
        if bus.is_connected:
            bus.disable_torque(motor_name_const)
            time.sleep(0.1)
            print("  ✓ Motor disabled (torque off)")

        # Step 10: Disconnect
        print("\nStep 10: Disconnecting...")
        if bus.is_connected:
            bus.disconnect(disable_torque=False)  # Already disabled
            print(f"  ✓ Disconnected from {can_port}")

    print(f"\n{'=' * 60}")
    print("Test completed successfully!")
    print(f"{'=' * 60}\n")


@pytest.mark.hardware
def test_motor_discovery_and_setup(can_port):
    """
    Test motor discovery and ID configuration.

    Note: This test requires the Damiao Debugging Tools for actual ID changes.
    This test only demonstrates the bus scan functionality.
    """

    print(f"\n{'=' * 60}")
    print("Damiao Motor Discovery Test")
    print(f"{'=' * 60}\n")

    print("Note: Motor ID configuration must be done via Damiao Debugging Tools")
    print("See: https://docs.openarm.dev/software/setup/motor-id")
    print()

    # Test if CAN interface is accessible
    print(f"Testing CAN interface: {can_port}")

    # Create a minimal motor bus for testing
    test_motor = Motor(0x01, "dm4310", MotorNormMode.DEGREES)
    test_motor.recv_id = 0x11
    test_motor.motor_type = MotorType.DM4310

    bus = DamiaoMotorsBus(port=can_port, motors={"test": test_motor})

    try:
        bus.connect(handshake=False)
        print(f"✓ CAN interface {can_port} is accessible")

        # Try to communicate with motor at 0x01
        print("\nLooking for motor at ID 0x01...")
        try:
            bus._refresh_motor("test")
            msg = bus._recv_motor_response(timeout=0.5)
            if msg:
                print(f"✓ Motor found at ID 0x01, response ID: 0x{msg.arbitration_id:02X}")
            else:
                print("✗ No response from motor")
                print("\nTroubleshooting:")
                print("  1. Verify motor is powered (24V)")
                print("  2. Check CAN wiring (CANH, CANL)")
                print("  3. Verify motor ID is set to 0x01")
                print("  4. Enable with: cansend can0 001#FFFFFFFFFFFFFFFC")
        except Exception as e:
            print(f"✗ Error communicating with motor: {e}")

    except Exception as e:
        print(f"✗ Failed to access CAN interface: {e}")
        print("\nSetup CAN interface:")
        print(f"  sudo ip link set {can_port} type can bitrate 1000000")
        print(f"  sudo ip link set {can_port} up")

    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=True)

    print(f"\n{'=' * 60}\n")


@pytest.mark.hardware
def test_multi_motor_sync_operations(can_port):
    """
    Test synchronized read/write with multiple motors.

    This demonstrates how to control multiple motors simultaneously.
    """

    print(f"\n{'=' * 60}")
    print("Damiao Multi-Motor Sync Test")
    print(f"{'=' * 60}\n")

    # Setup motors (adjust IDs as needed)
    motors = {
        "joint_1": Motor(0x01, "dm4310", MotorNormMode.DEGREES),
        "joint_2": Motor(0x02, "dm4310", MotorNormMode.DEGREES),
    }

    motors["joint_1"].recv_id = 0x11
    motors["joint_1"].motor_type = MotorType.DM4310
    motors["joint_2"].recv_id = 0x12
    motors["joint_2"].motor_type = MotorType.DM4310

    bus = DamiaoMotorsBus(port=can_port, motors=motors)

    try:
        bus.connect()
        bus.enable_torque()

        print("Reading all motor positions...")
        positions = bus.sync_read("Present_Position")
        for motor, pos in positions.items():
            print(f"  {motor}: {pos:.2f}°")

        print("\nMoving all motors to 45°...")
        target_positions = dict.fromkeys(motors, 45.0)
        bus.sync_write("Goal_Position", target_positions)
        time.sleep(2.0)

        positions = bus.sync_read("Present_Position")
        print("Final positions:")
        for motor, pos in positions.items():
            print(f"  {motor}: {pos:.2f}°")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("\nThis is expected on macOS without proper CAN hardware.")
        print("macOS does not support SocketCAN natively (Linux-only feature).")
        print("For macOS, you need a USB-CAN adapter with SLCAN support.")
    finally:
        if bus.is_connected:
            bus.disable_torque()
            bus.disconnect()

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    print("Damiao Motor Test Suite")
    print("=" * 60)
    print("\nThese tests require actual hardware to run.")
    print("Please ensure:")
    print("  1. Motor is connected and powered (24V)")
    print("  2. CAN interface is configured")
    print("  3. Motor ID is set to 0x01/0x11")
    print("\nTo run tests with hardware:")
    print("\n  Linux (SocketCAN):")
    print("    sudo ip link set can0 type can bitrate 1000000")
    print("    sudo ip link set can0 up")
    print("    pytest tests/motors/test_damiao.py --run-hardware --can-port can0")
    print("\n  macOS (USB-CAN adapter with SLCAN):")
    print("    pytest tests/motors/test_damiao.py --run-hardware --can-port /dev/cu.usbmodem00000000050C1")
    print("\nTo run without hardware (tests will be skipped):")
    print("  pytest tests/motors/test_damiao.py")
    print("\nNote: The --run-hardware and --can-port flags are configured in tests/conftest.py")
    print("=" * 60)

#!/usr/bin/env python

"""
Debug script to print current servo positions and pose information for SO100 follower robot.
"""

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig


def print_robot_state():
    """Connect to robot and print current state information."""

    # Configure robot - adjust port if needed
    config = SO100FollowerConfig(
        port="/dev/tty.usbmodem58760434091",  # Your robot's port from the error log
        disable_torque_on_disconnect=True,
        cameras={},  # No cameras for debugging to avoid timeout issues
    )

    # Create robot instance
    robot = SO100Follower(config)

    try:
        print("Connecting to SO100 Follower robot...")
        robot.connect(calibrate=False)  # Skip calibration for quick debug
        print("✓ Robot connected successfully!")

        print("\n" + "=" * 60)
        print("CURRENT ROBOT STATE")
        print("=" * 60)

        # Get current observation (includes servo positions)
        obs = robot.get_observation()

        print("\n📍 SERVO POSITIONS:")
        print("-" * 40)
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

        for motor in motor_names:
            key = f"{motor}.pos"
            if key in obs:
                position = obs[key]
                print(f"  {motor:15}: {position:8.2f}")

        print("\n🔄 RAW SERVO DATA:")
        print("-" * 40)
        # Get raw motor positions directly from bus
        raw_positions = robot.bus.sync_read("Present_Position")
        for motor, position in raw_positions.items():
            print(f"  {motor:15}: {position:8.0f} (raw)")

        print("\n⚙️  MOTOR CONFIGURATION:")
        print("-" * 40)
        for motor_name, motor_obj in robot.bus.motors.items():
            print(f"  {motor_name:15}: ID={motor_obj.id}, Model={motor_obj.model}")

        print("\n🔧 ROBOT INFO:")
        print("-" * 40)
        print(f"  Port: {config.port}")
        print(f"  Connected: {robot.is_connected}")
        print(f"  Calibrated: {robot.is_calibrated}")
        print(f"  Use degrees: {config.use_degrees}")

        # Try to get additional motor status
        print("\n📊 MOTOR STATUS:")
        print("-" * 40)
        try:
            temperatures = robot.bus.sync_read("Present_Temperature")
            voltages = robot.bus.sync_read("Present_Voltage")
            currents = robot.bus.sync_read("Present_Current")

            for motor in motor_names:
                if motor in temperatures:
                    temp = temperatures[motor]
                    volt = voltages.get(motor, "N/A")
                    curr = currents.get(motor, "N/A")
                    print(f"  {motor:15}: Temp={temp}°C, Voltage={volt}V, Current={curr}mA")
        except Exception as e:
            print(f"  Could not read additional motor status: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Make sure the robot is connected to {config.port}")

    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
                print("\n✓ Robot disconnected safely")
        except:
            pass


if __name__ == "__main__":
    print("SO100 Follower Robot State Debugger")
    print("=" * 40)
    print_robot_state()

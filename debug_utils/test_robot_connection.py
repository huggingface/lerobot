#!/usr/bin/env python

"""
Quick diagnostic to test robot connection before recording
"""

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus


def test_connection(port, robot_name):
    """Test connection to a robot arm"""
    print(f"\n🔌 Testing connection to {robot_name} on {port}...")

    try:
        # Define motors
        motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }

        # Create bus
        bus = FeetechMotorsBus(port=port, motors=motors)

        # Connect
        print("  → Connecting...")
        bus.connect()
        print("  ✅ Connected!")

        # Test read
        print("  → Reading positions...")
        positions = bus.sync_read("Present_Position")
        print("  ✅ Read successful!")

        # Display positions
        print("\n  Motor Positions:")
        for motor, pos in positions.items():
            print(f"    {motor:15}: {pos:6.1f}")

        # Disconnect
        bus.disconnect()
        print("\n  ✅ Disconnected successfully!")
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Test both leader and follower connections"""

    print("🤖 Robot Connection Diagnostic")
    print("=" * 50)

    # Test follower
    follower_ok = test_connection(port="/dev/tty.usbmodem58760434091", robot_name="SO100 Follower")

    # Test leader
    leader_ok = test_connection(port="/dev/tty.usbmodem58CD1771421", robot_name="SO100 Leader")

    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Follower: {'✅ OK' if follower_ok else '❌ FAILED'}")
    print(f"  Leader:   {'✅ OK' if leader_ok else '❌ FAILED'}")

    if not follower_ok or not leader_ok:
        print("\n💡 Troubleshooting tips:")
        print("  1. Unplug and replug USB cables")
        print("  2. Check motor power is ON")
        print("  3. Wait 2-3 seconds after power on")
        print("  4. Try different USB ports")
        print("  5. Check cables aren't damaged")


if __name__ == "__main__":
    main()

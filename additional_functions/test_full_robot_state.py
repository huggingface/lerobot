#!/usr/bin/env python3
"""
Test script for get_full_robot_state() function.
Displays IMU data and motor states from the G1 robot.

Usage:
    python test_full_robot_state.py
"""

import time
import logging
import numpy as np
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def print_robot_state(state: dict):
    """Print robot state in a readable format"""
    
    print("\n" + "=" * 70)
    print("G1 FULL ROBOT STATE")
    print("=" * 70)
    
    # IMU Data
    imu = state['imu']
    print(f"\n🧭 IMU:")
    print(f"  Orientation (deg): Roll={np.degrees(imu['rpy'][0]):+.1f}°, "
          f"Pitch={np.degrees(imu['rpy'][1]):+.1f}°, Yaw={np.degrees(imu['rpy'][2]):+.1f}°")
    print(f"  Gyroscope (rad/s): x={imu['gyroscope'][0]:+.3f}, "
          f"y={imu['gyroscope'][1]:+.3f}, z={imu['gyroscope'][2]:+.3f}")
    print(f"  Accel (m/s²):      x={imu['accelerometer'][0]:+.3f}, "
          f"y={imu['accelerometer'][1]:+.3f}, z={imu['accelerometer'][2]:+.3f}")
    print(f"  Quaternion:        w={imu['quaternion'][0]:.3f}, "
          f"x={imu['quaternion'][1]:+.3f}, y={imu['quaternion'][2]:+.3f}, z={imu['quaternion'][3]:+.3f}")
    print(f"  Temperature:       {imu['temperature']}°C")
    
    # Motor Data - show first 5 motors
    motors = state['motors']
    print(f"\n🦾 Motors (showing first 5 of {len(motors)}):")
    for motor in motors[:5]:
        print(f"  Motor {motor['id']:2d}: pos={motor['q']:+.3f} rad, "
              f"vel={motor['dq']:+.3f} rad/s, "
              f"torque={motor['tau_est']:+.2f} Nm, "
              f"temp={motor['temperature']}°C")
    
    # Motor Statistics
    print(f"\n📊 Motor Statistics (all {len(motors)} motors):")
    temps = [m['temperature'] for m in motors]
    torques = [abs(m['tau_est']) for m in motors]
    velocities = [abs(m['dq']) for m in motors]
    
    print(f"  Temperature: min={min(temps)}°C, max={max(temps)}°C, avg={sum(temps)/len(temps):.1f}°C")
    print(f"  Torque:      min={min(torques):.2f}Nm, max={max(torques):.2f}Nm, avg={sum(torques)/len(torques):.2f}Nm")
    print(f"  Velocity:    min={min(velocities):.3f}rad/s, max={max(velocities):.3f}rad/s")
    
    # High torque warning
    high_torque = [(m['id'], m['tau_est']) for m in motors if abs(m['tau_est']) > 5.0]
    if high_torque:
        print(f"\n⚠️  Motors with high torque (>5.0 Nm):")
        for motor_id, torque in high_torque[:10]:  # Show up to 10
            print(f"     Motor {motor_id:2d}: {torque:+.2f} Nm")
    
    print("\nPress Ctrl+C to stop")


def main():
    print("="*70)
    print("G1 Robot Full State Test")
    print("="*70)
    
    # Create robot config
    config = UnitreeG1Config(
        cameras={},  # No cameras needed for state test
        motion_mode=False,
        simulation_mode=False,
    )
    
    # Initialize robot
    print("\nInitializing robot...")
    robot = UnitreeG1(config)
    
    # Give it a moment to start receiving data
    time.sleep(1)
    
    try:
        print("\nReading robot state (updating every 0.5 seconds)...")
        print("This demonstrates the get_full_robot_state() function")
        
        while True:
            # Get full robot state
            state = robot.get_full_robot_state()
            
            # Print it
            print_robot_state(state)
            
            # Wait before next update
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nDisconnecting robot...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()


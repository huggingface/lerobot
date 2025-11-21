#!/usr/bin/env python3
"""
Quick G1 sensor test script.
Displays IMU, motor states, and system info.

Usage:
    python test_sensors.py <network_interface>
    Example: python test_sensors.py en7
"""

import sys
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


def print_sensor_data(msg: LowState_):
    """Print sensor data in a readable format"""
    
    print("\n" + "=" * 70)
    print(f"G1 SENSOR DATA @ tick {msg.tick}")
    print("=" * 70)
    
    # IMU
    imu = msg.imu_state
    print(f"\n🧭 IMU:")
    print(f"  Orientation (deg): Roll={np.degrees(imu.rpy[0]):+.1f}°, "
          f"Pitch={np.degrees(imu.rpy[1]):+.1f}°, Yaw={np.degrees(imu.rpy[2]):+.1f}°")
    print(f"  Gyroscope (rad/s): x={imu.gyroscope[0]:+.3f}, "
          f"y={imu.gyroscope[1]:+.3f}, z={imu.gyroscope[2]:+.3f}")
    print(f"  Accel (m/s²):      x={imu.accelerometer[0]:+.3f}, "
          f"y={imu.accelerometer[1]:+.3f}, z={imu.accelerometer[2]:+.3f}")
    print(f"  Quaternion:        w={imu.quaternion[0]:.3f}, "
          f"x={imu.quaternion[1]:+.3f}, y={imu.quaternion[2]:+.3f}, z={imu.quaternion[3]:+.3f}")
    print(f"  Temperature:       {imu.temperature}°C")
    
    # Motors - show first 5 and summary
    print(f"\n🦾 Motors (showing first 5 of 35):")
    for i in range(min(5, 35)):
        motor = msg.motor_state[i]
        print(f"  Motor {i:2d}: pos={motor.q:+.3f} rad, "
              f"vel={motor.dq:+.3f} rad/s, "
              f"torque={motor.tau_est:+.2f} Nm, "
              f"temp={motor.temperature[0]}°C")
    
    # Motor statistics
    print(f"\n📊 Motor Statistics (all 35):")
    temps = [msg.motor_state[i].temperature[0] for i in range(35)]
    torques = [abs(msg.motor_state[i].tau_est) for i in range(35)]
    velocities = [abs(msg.motor_state[i].dq) for i in range(35)]
    
    print(f"  Temperature: min={min(temps)}°C, max={max(temps)}°C, avg={sum(temps)/len(temps):.1f}°C")
    print(f"  Torque:      min={min(torques):.2f}Nm, max={max(torques):.2f}Nm, avg={sum(torques)/len(torques):.2f}Nm")
    print(f"  Velocity:    min={min(velocities):.3f}rad/s, max={max(velocities):.3f}rad/s")
    
    # High torque warning
    high_torque = [(i, msg.motor_state[i].tau_est) for i in range(35) 
                   if abs(msg.motor_state[i].tau_est) > 5.0]
    if high_torque:
        print(f"\n⚠️  Motors with high torque (>5.0 Nm):")
        for motor_id, torque in high_torque[:10]:  # Show up to 10
            print(f"     Motor {motor_id:2d}: {torque:+.2f} Nm")
    
    # System
    print(f"\n⚙️  System Info:")
    print(f"  Mode: machine={msg.mode_machine}, pr={msg.mode_pr}")
    print(f"  Version: {msg.version[0]}.{msg.version[1]}")
    
    print("\nPress Ctrl+C to stop")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <network_interface>")
        print(f"Example: python3 {sys.argv[0]} en7")
        sys.exit(1)
    
    network_interface = sys.argv[1]
    
    print(f"Initializing DDS on {network_interface}...")
    ChannelFactoryInitialize(0, network_interface)
    
    print("Subscribing to rt/lowstate...")
    lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_sub.Init(print_sensor_data, 10)
    
    print("Receiving sensor data...")
    print("(Data will print each time it's received)")
    
    try:
        while True:
            time.sleep(1)  # Keep alive, handler prints on each message
    except KeyboardInterrupt:
        print("\n\nStopped")


if __name__ == "__main__":
    main()


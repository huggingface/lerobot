#!/usr/bin/env python3
"""
G1 Robot Monitor - IMU, Force/Torque, Temperature, and Battery

Displays:
- IMU data (orientation, gyro, accelerometer)
- Motor force/torque (estimated from current)
- Motor temperatures
- Battery status (requires BMS topic, may not be available)

Usage:
    python monitor_robot.py <network_interface>
    Example: python monitor_robot.py en7
"""

import sys
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# Try to import BMS (may not be available on all systems)
try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_
    HAS_BMS = True
except:
    HAS_BMS = False
    print("Note: BMS not available in this SDK version")


class RobotMonitor:
    def __init__(self):
        self.latest_state = None
        self.latest_bms = None
        self.update_count = 0
        
    def state_handler(self, msg: LowState_):
        """Handle low-level state updates (IMU, motors)"""
        self.latest_state = msg
        self.update_count += 1
    
    def bms_handler(self, msg):
        """Handle battery management system updates"""
        self.latest_bms = msg
    
    def print_status(self):
        """Print formatted sensor data"""
        if not self.latest_state:
            print("Waiting for robot data...")
            return
        
        state = self.latest_state
        
        # Clear screen and print header
        print("\033[2J\033[H")  # Clear screen
        print("=" * 80)
        print(f"G1 ROBOT MONITOR - Updates: {self.update_count}")
        print("=" * 80)
        
        # ============ IMU DATA ============
        print("\n🧭 IMU DATA:")
        imu = state.imu_state
        
        # Orientation (Euler angles)
        roll_deg = np.degrees(imu.rpy[0])
        pitch_deg = np.degrees(imu.rpy[1])
        yaw_deg = np.degrees(imu.rpy[2])
        print(f"  Orientation:")
        print(f"    Roll:  {roll_deg:+7.2f}°")
        print(f"    Pitch: {pitch_deg:+7.2f}°")
        print(f"    Yaw:   {yaw_deg:+7.2f}°")
        
        # Angular velocity (gyroscope)
        print(f"  Angular Velocity (rad/s):")
        print(f"    X: {imu.gyroscope[0]:+7.3f}")
        print(f"    Y: {imu.gyroscope[1]:+7.3f}")
        print(f"    Z: {imu.gyroscope[2]:+7.3f}")
        
        # Linear acceleration
        print(f"  Linear Acceleration (m/s²):")
        print(f"    X: {imu.accelerometer[0]:+7.3f}")
        print(f"    Y: {imu.accelerometer[1]:+7.3f}")
        print(f"    Z: {imu.accelerometer[2]:+7.3f}")
        
        # Gravity magnitude (should be ~9.81 when stationary)
        accel_mag = np.linalg.norm(imu.accelerometer)
        print(f"  Acceleration Magnitude: {accel_mag:.2f} m/s² (gravity ~9.81)")
        
        print(f"  IMU Temperature: {imu.temperature}°C")
        
        # ============ ALL MOTORS: TORQUE & TEMPERATURE ============
        print("\n⚡ ALL MOTORS - TORQUE & TEMPERATURE:")
        print(f"  {'Motor':<8} {'Torque (Nm)':<14} {'Temp (°C)':<12} {'Status'}")
        print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*20}")
        
        for i in range(35):
            motor = state.motor_state[i]
            torque = motor.tau_est
            temp = motor.temperature[0]
            
            # Status indicators
            status = []
            if abs(torque) > 5.0:
                status.append("⚡HIGH TORQUE")
            if temp > 60:
                status.append("🔥HOT")
            elif temp > 50:
                status.append("⚠️ WARM")
            
            status_str = " ".join(status) if status else "✓"
            
            print(f"  {i:2d}       {torque:+7.3f}        {temp:3d}          {status_str}")
        
        # Summary statistics
        torques = [state.motor_state[i].tau_est for i in range(35)]
        temps = [state.motor_state[i].temperature[0] for i in range(35)]
        
        print(f"\n  Statistics:")
        print(f"    Torque:  min={min([abs(t) for t in torques]):.2f}  "
              f"max={max([abs(t) for t in torques]):.2f}  "
              f"mean={np.mean([abs(t) for t in torques]):.2f} Nm")
        print(f"    Temp:    min={min(temps)}  max={max(temps)}  mean={np.mean(temps):.1f}°C")
        
        # ============ BATTERY ============
        print("\n🔋 BATTERY:")
        if self.latest_bms:
            bms = self.latest_bms
            print(f"  State of Charge: {bms.soc}%")
            print(f"  State of Health: {bms.soh}%")
            print(f"  Voltage: {bms.bmsvoltage[0] / 1000.0:.2f} V")
            print(f"  Current: {bms.current / 1000.0:.2f} A")
            
            # Power calculation
            power = (bms.bmsvoltage[0] / 1000.0) * (bms.current / 1000.0)
            print(f"  Power: {power:.2f} W")
            
            # Temperature range
            temps = [t / 10.0 for t in bms.temperature]
            print(f"  Battery Temp: {min(temps):.1f}°C to {max(temps):.1f}°C")
            print(f"  Charge Cycles: {bms.cycle}")
        else:
            print(f"  Battery data not available")
            print(f"  (BMS topic may not be published or accessible)")
        
        # ============ SYSTEM INFO ============
        print(f"\n⚙️  SYSTEM:")
        print(f"  Tick: {state.tick}")
        print(f"  Mode: machine={state.mode_machine}, pr={state.mode_pr}")
        
        print("\n" + "=" * 80)
        print("Press Ctrl+C to stop")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <network_interface>")
        print(f"Example: python3 {sys.argv[0]} en7")
        sys.exit(1)
    
    network_interface = sys.argv[1]
    
    print("=" * 80)
    print("G1 ROBOT MONITOR")
    print("=" * 80)
    print(f"Initializing DDS on {network_interface}...")
    ChannelFactoryInitialize(0, network_interface)
    
    # Create monitor
    monitor = RobotMonitor()
    
    # Subscribe to low-level state (IMU, motors)
    print("Subscribing to rt/lowstate...")
    lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_sub.Init(monitor.state_handler, 10)
    
    # Try to subscribe to BMS (may not be available)
    if HAS_BMS:
        print("Attempting to subscribe to battery data...")
        try:
            # Common BMS topic names
            for topic in ["rt/bms", "rt/battery", "rt/bms_state"]:
                try:
                    bms_sub = ChannelSubscriber(topic, BmsState_)
                    bms_sub.Init(monitor.bms_handler, 10)
                    print(f"  Subscribed to {topic}")
                    break
                except:
                    pass
        except Exception as e:
            print(f"  Battery data not available: {e}")
    
    print("\nWaiting for data...")
    time.sleep(1)
    
    print("Starting monitor...\n")
    
    try:
        while True:
            monitor.print_status()
            time.sleep(0.5)  # Update display at 2 Hz
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Monitor stopped")
        print("=" * 80)


if __name__ == "__main__":
    main()


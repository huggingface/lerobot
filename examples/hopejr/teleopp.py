import time
import serial
import numpy as np
import matplotlib.pyplot as plt

# Import the motor bus (adjust the import path as needed)
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

def main():
    # -------------------------------
    # Setup the motor bus (ACM0)
    # -------------------------------
    arm_bus = FeetechMotorsBus(
        port="/dev/ttyACM0",
        motors={
            "wrist_pitch": [7, "sts3215"],
        },
        protocol_version=0,
    )
    arm_bus.connect()
    
    # -------------------------------
    # Setup the serial connection for sensor (ACM1)
    # -------------------------------
    try:
        ser = serial.Serial("/dev/ttyACM1", 115200, timeout=1)
    except Exception as e:
        print(f"Error opening serial port /dev/ttyACM1: {e}")
        return

    # Lists to store the motor positions and sensor values.
    positions = []
    sensor_values = []
    
    # -------------------------------
    # Loop: move motor and collect sensor data
    # -------------------------------
    # We assume that 2800 > 1480 so we decrement by 10 each step.
    for pos in range(2800, 1500, -10):  # 2800 down to 1480 (inclusive)
        # Command the motor to go to position 'pos'
        arm_bus.write("Goal_Position", pos, ["wrist_pitch"])
        
        # Wait a short period for the motor to move and the sensor to update.
        time.sleep(0.01)
        
        # Read one line from the sensor device.
        sensor_val = np.nan  # default if reading fails
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                # Split the line into parts and convert each part to int.
                parts = line.split()
                # Ensure there are enough values (we expect at least 15 values)
                if len(parts) >= 15:
                    values = [int(x) for x in parts]
                    # Use the 15th value (index 14)
                    sensor_val = values[14]
        except Exception as e:
            print(f"Error parsing sensor data: {e}")
        
        positions.append(pos)
        sensor_values.append(sensor_val)
        print(f"Motor pos: {pos} | Sensor 15th value: {sensor_val}")
    
    #move it back to 
    arm_bus.write("Goal_Position", 2800, ["wrist_pitch"])
    # -------------------------------
    # Plot the data: Motor Angle vs. Sensor 15th Value
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(positions, sensor_values, marker='o', linestyle='-')
    plt.xlabel("Motor Angle")
    plt.ylabel("Sensor 15th Value")
    plt.title("Motor Angle vs Sensor 15th Value")
    plt.grid(True)
    plt.savefig("asd.png", dpi=300)
    plt.close()
    print("Plot saved as asd.png")
    
    # Close the serial connection.
    ser.close()

if __name__ == "__main__":
    main()

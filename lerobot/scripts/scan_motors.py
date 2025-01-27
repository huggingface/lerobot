"""
This script is helpful to diagnose issues with motors during calibration or later operation. 
It scans all motors on a given port and displays their settings.

Example of usage:
```bash
python lerobot/scripts/scan_motors.py \
  --port /dev/tty.usbmodem585A0080521 \
  --brand feetech \
  --model sts3215 \
```
"""

import argparse
import time


def scan_motors(port, brand, model):
    if brand == "feetech":
        from lerobot.common.robot_devices.motors.feetech import MODEL_BAUDRATE_TABLE
        from lerobot.common.robot_devices.motors.feetech import (
            SCS_SERIES_BAUDRATE_TABLE as SERIES_BAUDRATE_TABLE,
        )
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus as MotorsBusClass
    elif brand == "dynamixel":
        from lerobot.common.robot_devices.motors.dynamixel import MODEL_BAUDRATE_TABLE
        from lerobot.common.robot_devices.motors.dynamixel import (
            X_SERIES_BAUDRATE_TABLE as SERIES_BAUDRATE_TABLE,
        )
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus as MotorsBusClass
    else:
        raise ValueError(
            f"Currently we do not support this motor brand: {brand}. We currently support feetech and dynamixel motors."
        )

    # Check if the provided model exists in the model_baud_rate_table
    if model not in MODEL_BAUDRATE_TABLE:
        raise ValueError(
            f"Invalid model '{model}' for brand '{brand}'. Supported models: {list(MODEL_BAUDRATE_TABLE.keys())}"
        )

    # Setup motor names, indices, and models
    motor_name = "motor"
    motor_index_arbitrary = -1  # Use an arbitrary out of range motor ID
    motor_model = model  # Use the motor model passed via argument

    # Initialize the MotorBus with the correct port and motor configurations
    motor_bus = MotorsBusClass(port=port, motors={motor_name: (motor_index_arbitrary, motor_model)})

    # Try to connect to the motor bus and handle any connection-specific errors
    try:
        motor_bus.connect()
        print(f"Connected on port {motor_bus.port}")
    except OSError as e:
        print(f"Error occurred when connecting to the motor bus: {e}")
        return

    # Motor bus is connected, proceed with the rest of the operations
    try:
        print("Scanning all baudrates and motor indices")
        all_baudrates = set(SERIES_BAUDRATE_TABLE.values())
        motors_detected = False

        for baudrate in all_baudrates:
            motor_bus.set_bus_baudrate(baudrate)
            present_ids = motor_bus.find_motor_indices(list(range(1, 10)))
            if len(present_ids) > 0:
                print(f"{len(present_ids)} motor ID(s) detected for baud rate {baudrate}. Motor IDs: {present_ids}.")
                motors_detected = True

            for motor_idx in present_ids:
                present_idx = motor_bus.read_with_motor_ids(motor_bus.motor_models, motor_idx, "ID", num_retry=2)
                if present_idx != motor_idx:
                    raise OSError(f"Failed to access motor index {motor_idx}.")

                if brand == "feetech":
                    print("Present Position", motor_bus.read_with_motor_ids(motor_bus.motor_models, motor_idx, "Present_Position", num_retry=2))
                    print("Offset", motor_bus.read_with_motor_ids(motor_bus.motor_models, motor_idx, "Offset", num_retry=2))

        if not motors_detected:
            print("No motors detected.")

        print("Scan finished.")

    finally:
        motor_bus.disconnect()
        print("Disconnected from motor bus.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True, help="Motors bus port (e.g. dynamixel,feetech)")
    parser.add_argument("--brand", type=str, required=True, help="Motor brand (e.g. dynamixel,feetech)")
    parser.add_argument("--model", type=str, required=True, help="Motor model (e.g. xl330-m077,sts3215)")
    args = parser.parse_args()

    scan_motors(args.port, args.brand, args.model)

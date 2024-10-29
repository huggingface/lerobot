"""
This script configure a single motor at a time to a given ID and baudrate.

Example of usage:
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem585A0080521 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```
"""

import argparse
import time


def configure_motor(port, brand, model, motor_idx_des, baudrate_des):
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
    motor_index_arbitrary = motor_idx_des  # Use the motor ID passed via argument
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
        motor_index = -1  # Set the motor index to an out-of-range value.

        for baudrate in all_baudrates:
            motor_bus.set_bus_baudrate(baudrate)
            present_ids = motor_bus.find_motor_indices(list(range(1, 10)))
            if len(present_ids) > 1:
                raise ValueError(
                    "Error: More than one motor ID detected. This script is designed to only handle one motor at a time. Please disconnect all but one motor."
                )

            if len(present_ids) == 1:
                if motor_index != -1:
                    raise ValueError(
                        "Error: More than one motor ID detected. This script is designed to only handle one motor at a time. Please disconnect all but one motor."
                    )
                motor_index = present_ids[0]

        if motor_index == -1:
            raise ValueError("No motors detected. Please ensure you have one motor connected.")

        print(f"Motor index found at: {motor_index}")

        if brand == "feetech":
            # Allows ID and BAUDRATE to be written in memory
            motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "Lock", 0)

        if baudrate != baudrate_des:
            print(f"Setting its baudrate to {baudrate_des}")
            baudrate_idx = list(SERIES_BAUDRATE_TABLE.values()).index(baudrate_des)

            # The write can fail, so we allow retries
            motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "Baud_Rate", baudrate_idx)
            time.sleep(0.5)
            motor_bus.set_bus_baudrate(baudrate_des)
            present_baudrate_idx = motor_bus.read_with_motor_ids(
                motor_bus.motor_models, motor_index, "Baud_Rate", num_retry=2
            )

            if present_baudrate_idx != baudrate_idx:
                raise OSError("Failed to write baudrate.")

        print(f"Setting its index to desired index {motor_idx_des}")
        motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "Lock", 0)
        motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "ID", motor_idx_des)

        present_idx = motor_bus.read_with_motor_ids(motor_bus.motor_models, motor_idx_des, "ID", num_retry=2)
        if present_idx != motor_idx_des:
            raise OSError("Failed to write index.")

        if brand == "feetech":
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            motor_bus.write("Lock", 0)
            motor_bus.write("Maximum_Acceleration", 254)

            motor_bus.write("Goal_Position", 2048)
            time.sleep(4)
            print("Present Position", motor_bus.read("Present_Position"))

            motor_bus.write("Offset", 0)
            time.sleep(4)
            print("Offset", motor_bus.read("Offset"))

    except Exception as e:
        print(f"Error occurred during motor configuration: {e}")

    finally:
        motor_bus.disconnect()
        print("Disconnected from motor bus.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True, help="Motors bus port (e.g. dynamixel,feetech)")
    parser.add_argument("--brand", type=str, required=True, help="Motor brand (e.g. dynamixel,feetech)")
    parser.add_argument("--model", type=str, required=True, help="Motor model (e.g. xl330-m077,sts3215)")
    parser.add_argument("--ID", type=int, required=True, help="Desired ID of the current motor (e.g. 1,2,3)")
    parser.add_argument(
        "--baudrate", type=int, default=1000000, help="Desired baudrate for the motor (default: 1000000)"
    )
    args = parser.parse_args()

    configure_motor(args.port, args.brand, args.model, args.ID, args.baudrate)

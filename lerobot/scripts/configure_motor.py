import argparse
import importlib
import time


def configure_motor(brand, model, motor_idx_des, baudrate_des):
    if brand == "feetech":
        motor_bus_class = importlib.import_module(
            "lerobot.common.robot_devices.motors.feetech"
        ).FeetechMotorsBus
        baudrate_table = importlib.import_module(
            "lerobot.common.robot_devices.motors.feetech"
        ).SCS_SERIES_BAUDRATE_TABLE
        num_write_retry = importlib.import_module(
            "lerobot.common.robot_devices.motors.feetech"
        ).NUM_WRITE_RETRY
        model_baud_rate_table = importlib.import_module(
            "lerobot.common.robot_devices.motors.feetech"
        ).MODEL_BAUDRATE_TABLE
    elif brand == "dynamixel":
        motor_bus_class = importlib.import_module(
            "lerobot.common.robot_devices.motors.dynamixel"
        ).DynamixelMotorsBus
        baudrate_table = importlib.import_module(
            "lerobot.common.robot_devices.motors.dynamixel"
        ).X_SERIES_BAUDRATE_TABLE
        num_write_retry = importlib.import_module(
            "lerobot.common.robot_devices.motors.dynamixel"
        ).NUM_WRITE_RETRY
        model_baud_rate_table = importlib.import_module(
            "lerobot.common.robot_devices.motors.dynamixel"
        ).MODEL_BAUDRATE_TABLE
    else:
        raise ValueError(
            f"Currently we do not support this motor brand: {brand}. We currently support feetech and dynamixel motors."
        )

    # Check if the provided model exists in the model_baud_rate_table
    if model not in model_baud_rate_table:
        raise ValueError(
            f"Invalid model '{model}' for brand '{brand}'. Supported models: {list(model_baud_rate_table.keys())}"
        )

    # Setup motor names, indices, and models
    motor_name = "motor"
    motor_index_arbitrary = motor_idx_des  # Use the motor ID passed via argument
    motor_model = model  # Use the motor model passed via argument

    # Initialize the MotorBus with the correct port and motor configurations
    motor_bus = motor_bus_class(
        port="/dev/ttyACM0", motors={motor_name: (motor_index_arbitrary, motor_model)}
    )

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
        all_baudrates = set(baudrate_table.values())
        motor_index = -1  # Set the motor index to an out-of-range value.

        for baudrate in all_baudrates:
            motor_bus.set_bus_baudrate(baudrate)
            present_ids = motor_bus.find_motor_indices()
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

        if baudrate != baudrate_des:
            print(f"Setting its baudrate to {baudrate_des}")
            baudrate_idx = list(baudrate_table.values()).index(baudrate_des)

            # The write can fail, so we allow retries
            for _ in range(num_write_retry):
                motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "Baud_Rate", baudrate_idx)
                time.sleep(0.5)
                motor_bus.set_bus_baudrate(baudrate_des)
                try:
                    present_baudrate_idx = motor_bus.read_with_motor_ids(
                        motor_bus.motor_models, motor_index, "Baud_Rate"
                    )
                except ConnectionError:
                    print("Failed to write baudrate. Retrying.")
                    motor_bus.set_bus_baudrate(baudrate)
                    continue
                break
            else:
                raise OSError("Failed to write baudrate.")

            if present_baudrate_idx != baudrate_idx:
                raise OSError("Failed to write baudrate.")

        print(f"Setting its index to desired index {motor_idx_des}")
        motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "ID", motor_idx_des)

        present_idx = motor_bus.read_with_motor_ids(motor_bus.motor_models, motor_idx_des, "ID")
        if present_idx != motor_idx_des:
            raise OSError("Failed to write index.")

    except Exception as e:
        print(f"Error occurred during motor configuration: {e}")

    finally:
        # Disconnect the motor bus
        motor_bus.disconnect()
        print("Disconnected from motor bus.")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="This script is used to configure a single motor at a time to the ID and baudrate you desire."
    )
    parser.add_argument("--brand", type=str, required=True, help="Motor brand (e.g., dynamixel, feetech)")
    parser.add_argument("--model", type=str, required=True, help="Motor model (e.g., xl330-m077, sts3215)")
    parser.add_argument("--ID", type=int, required=True, help="Desired ID of the current motor (e.g., 1)")
    parser.add_argument(
        "--baudrate", type=int, default=1000000, help="Desired baudrate for the motor (default: 1000000)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the configure_motor function with the parsed arguments
    configure_motor(args.brand, args.model, args.ID, args.baudrate)

import time


def calibrate_motor(motor_id, motor_bus):
    """
    1) Prompts user to move the motor to the "home" position.
    2) Reads servo ticks.
    3) Calculates the offset so that 'home_ticks' becomes 0.
    4) Returns the offset
    """
    print(f"\n--- Calibrating Motor {motor_id} ---")
    input("Move the motor to middle (home) position, then press Enter...")
    home_ticks = motor_bus.read("Present_Position")[0]
    print(f"  [Motor {motor_id}] middle position recorded: {home_ticks}")

    # Calculate how many ticks to shift so that 'home_ticks' becomes 0
    raw_offset = -home_ticks  # negative of home_ticks
    encoder_offset = raw_offset % 4096  # wrap to [0..4095]

    # Convert to a signed range [-2048..2047]
    if encoder_offset > 2047:
        encoder_offset -= 4096

    print(f"Encoder offset: {encoder_offset}")

    return encoder_offset


def adjusted__to_homing_ticks(raw_motor_ticks, encoder_offset):
    shifted = (raw_motor_ticks + encoder_offset) % 4096
    if shifted > 2047:
        shifted -= 4096
    return shifted


def adjusted_to_motor_ticks(adjusted_pos, encoder_offset):
    """
    Inverse of read_adjusted_position().

    adjusted_pos : int in [-2048 .. +2047]  (the homed, shifted value)
    encoder_offset : int (the offset computed so that `home` becomes zero)

    Returns the raw servo ticks in [0..4095].
    """
    if adjusted_pos < 0:
        adjusted_pos += 4096

    raw_ticks = (adjusted_pos - encoder_offset) % 4096
    return raw_ticks


def configure_and_calibrate():
    from lerobot.common.robot_devices.motors.feetech import (
        SCS_SERIES_BAUDRATE_TABLE as SERIES_BAUDRATE_TABLE,
    )
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus as MotorsBusClass

    motor_idx_des = 1  # index you want to assign

    # Setup motor names, indices, and models
    motor_name = "motor"
    motor_index_arbitrary = motor_idx_des  # Use the motor ID passed via argument
    motor_model = "sts3215"  # Use the motor model passed via argument

    # Initialize the MotorBus with the correct port and motor configurations
    motor_bus = MotorsBusClass(
        port="/dev/tty.usbmodem585A0078271", motors={motor_name: (motor_index_arbitrary, motor_model)}
    )

    # Try to connect to the motor bus and handle any connection-specific errors
    try:
        motor_bus.connect()
        print(f"Connected on port {motor_bus.port}")
    except OSError as e:
        print(f"Error occurred when connecting to the motor bus: {e}")
        return

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

        motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "Lock", 0)

        baudrate_des = 1000000

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

        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        motor_bus.write("Lock", 0)
        motor_bus.write("Maximum_Acceleration", 254)

        motor_bus.write("Max_Angle_Limit", 4095)  # default 4095
        motor_bus.write("Min_Angle_Limit", 0)  # default 0
        motor_bus.write("Offset", 0)
        motor_bus.write("Mode", 0)
        motor_bus.write("Goal_Position", 0)
        motor_bus.write("Torque_Enable", 0)

        motor_bus.write("Lock", 1)

        # Calibration
        print("Offset", motor_bus.read("Offset"))
        print("Max_Angle_Limit", motor_bus.read("Max_Angle_Limit"))
        print("Min_Angle_Limit", motor_bus.read("Min_Angle_Limit"))
        print("Goal_Position", motor_bus.read("Goal_Position"))
        print("Present Position", motor_bus.read("Present_Position"))

        encoder_offset = calibrate_motor(motor_idx_des, motor_bus)

        try:
            while True:
                raw_motor_ticks = motor_bus.read("Present_Position")[0]
                adjusted_ticks = adjusted__to_homing_ticks(raw_motor_ticks, encoder_offset)
                inverted_adjusted_ticks = adjusted_to_motor_ticks(adjusted_ticks, encoder_offset)
                print(
                    f"Raw Motor ticks: {raw_motor_ticks} | Adjusted ticks: {adjusted_ticks} | Invert adjusted ticks: {inverted_adjusted_ticks}"
                )
                time.sleep(0.3)
        except KeyboardInterrupt:
            print("Stopped reading positions.")

    except Exception as e:
        print(f"Error occurred during motor configuration: {e}")

    finally:
        motor_bus.disconnect()
        print("Disconnected from motor bus.")


if __name__ == "__main__":
    configure_and_calibrate()

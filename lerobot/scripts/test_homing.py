from lerobot.common.robot_devices.motors.feetech import (
    adjusted_to_homing_ticks,
    adjusted_to_motor_ticks,
    convert_steps_to_degrees,
)

# TODO(pepijn): remove this file!

MODEL_RESOLUTION = {
    "scs_series": 4096,
    "sts3215": 4096,
}


def calibrate_homing_motor(motor_id, motor_bus):
    """
    1) Reads servo ticks.
    2) Calculates the offset so that 'home_ticks' becomes 0.
    3) Returns the offset
    """

    home_ticks = motor_bus.read("Present_Position")[0]  # Read index starts at 0

    # Calculate how many ticks to shift so that 'home_ticks' becomes 0
    raw_offset = -home_ticks  # negative of home_ticks
    encoder_offset = raw_offset % 4096  # wrap to [0..4095]

    # Convert to a signed range [-2048..2047]
    if encoder_offset > 2047:
        encoder_offset -= 4096

    print(f"Encoder offset: {encoder_offset}")

    return encoder_offset


def configure_and_calibrate():
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus as MotorsBusClass

    motor_idx_des = 2  # index of motor

    # Setup motor names, indices, and models
    motor_name = "motor"
    motor_index_arbitrary = motor_idx_des  # Use the motor ID passed via argument
    motor_model = "sts3215"  # Use the motor model passed via argument

    # Initialize the MotorBus with the correct port and motor configurations
    motor_bus = MotorsBusClass(
        port="/dev/tty.usbmodem58760431631", motors={motor_name: (motor_index_arbitrary, motor_model)}
    )

    # Try to connect to the motor bus and handle any connection-specific errors
    try:
        motor_bus.connect()
        print(f"Connected on port {motor_bus.port}")
    except OSError as e:
        print(f"Error occurred when connecting to the motor bus: {e}")
        return

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

    input("Move the motor to middle (home) position, then press Enter...")
    encoder_offset = calibrate_homing_motor(motor_idx_des, motor_bus)

    try:
        while True:
            name = motor_bus.motor_names[0]
            _, model = motor_bus.motors[name]
            raw_motor_ticks = motor_bus.read("Present_Position")[0]
            adjusted_ticks = adjusted_to_homing_ticks(raw_motor_ticks, encoder_offset, model, motor_bus, 1)
            inverted_ticks = adjusted_to_motor_ticks(adjusted_ticks, encoder_offset, model, motor_bus, 1)
            adjusted_degrees = convert_steps_to_degrees([adjusted_ticks], [model])
            print(
                f"Raw Motor ticks: {raw_motor_ticks} | Adjusted ticks: {adjusted_ticks} | Adjusted degrees: {adjusted_degrees} | Invert adjusted ticks: {inverted_ticks}"
            )
            # time.sleep(0.3)
    except KeyboardInterrupt:
        print("Stopped reading positions.")

    except Exception as e:
        print(f"Error occurred during motor configuration: {e}")

    finally:
        motor_bus.disconnect()
        print("Disconnected from motor bus.")


if __name__ == "__main__":
    configure_and_calibrate()

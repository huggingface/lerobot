import argparse
import json
import time
from pathlib import Path

from lerobot.common.robot_devices.motors.feetech import (
    FeetechMotorsBus,
    adjusted_to_homing_ticks,
    adjusted_to_motor_ticks,
    convert_degrees_to_ticks,
    convert_ticks_to_degrees,
)

# Replace this import with your real config class import
# (e.g., So100RobotConfig or any other).
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config


def apply_feetech_offsets_from_calibration(motors_bus: FeetechMotorsBus, calibration_dict: dict):
    """
    Reads 'calibration_dict' containing 'homing_offset' and 'motor_names',
    then writes each motor's offset to the servo's internal Offset (0x1F) in EPROM,
    except for any motor whose name contains "gripper" (skipped).

    This version is modified so each homed position (originally 0) will now read
    2047, i.e. 180° away from 0 in the 4096-count circle. Offsets are permanently
    stored in EEPROM, so the servo's Present_Position is hardware-shifted even
    after power cycling.

    Steps:
      1) Subtract 2047 from the old offset (so 0 -> 2047).
      2) Clamp to [-2047..+2047].
      3) Encode sign bit and magnitude into a 12-bit number.
      4) Skip "gripper" motors, as they do not require this shift.
    """

    homing_offsets = calibration_dict["homing_offset"]
    motor_names = calibration_dict["motor_names"]

    # 1) Open the write lock => Lock=1 => changes to EEPROM do NOT persist yet
    motors_bus.write("Lock", 1)

    # 2) For each motor, set the 'Offset' parameter
    for m_name, old_offset in zip(motor_names, homing_offsets, strict=False):
        # If bus doesn’t have a motor named m_name, skip
        if m_name not in motors_bus.motors:
            print(f"Warning: '{m_name}' not found in motors_bus.motors; skipping offset.")
            continue

        if m_name == "gripper":
            print("Skipping gripper")
            continue

        # Shift the offset so the homed position reads 2047
        new_offset = old_offset - 2047

        # Clamp to [-2047..+2047]
        if new_offset > 2047:
            new_offset = 2047
            print("CLAMPED! -> shouldn't happen")
        elif new_offset < -2047:
            new_offset = -2047
            print("CLAMPED! -> shouldn't happen")

        # Determine the direction (sign) bit and magnitude
        direction_bit = 1 if new_offset < 0 else 0
        magnitude = abs(new_offset)

        # Combine sign bit (bit 11) with the magnitude (bits 0..10)
        servo_offset = (direction_bit << 11) | magnitude

        # Write to servo
        motors_bus.write("Offset", servo_offset, motor_names=m_name)
        print(
            f"Set offset for {m_name}: "
            f"old_offset={old_offset}, new_offset={new_offset}, servo_encoded={magnitude} + direction={direction_bit}"
        )

    motors_bus.write("Lock", 0)
    print("Offsets have been saved to EEPROM successfully.")


def debug_feetech_positions(cfg, arm_arg: str):
    """
    Reads each joint’s (1) raw ticks, (2) homed ticks, (3) degrees, and (4) invert-adjusted ticks.
    Optionally writes offsets from the calibration file into servo EEPROM.

    :param cfg: A config object (e.g. So100RobotConfig).
    :param arm_arg: One of "main_leader" or "main_follower".
    :param apply_offsets: If True, calls apply_feetech_offsets_from_calibration() to save offsets to EEPROM.
    """
    # 1) Make the Robot from your config
    robot = make_robot_from_config(cfg)

    # 2) Parse which arm we want: 'main_leader' or 'main_follower'
    if arm_arg not in ("main_leader", "main_follower"):
        raise ValueError("Please specify --arm=main_leader or --arm=main_follower")

    # Grab the relevant bus config from your robot’s arms
    bus_config = robot.leader_arms["main"] if arm_arg == "main_leader" else robot.follower_arms["main"]

    # 3) Create the Feetech bus from that config
    bus = FeetechMotorsBus(bus_config)

    # 4) Load calibration if it exists
    calib_file = Path(robot.calibration_dir) / f"{arm_arg}.json"
    if calib_file.exists():
        with open(calib_file) as f:
            calibration_dict = json.load(f)
        bus.set_calibration(calibration_dict)
        print(f"Loaded calibration from {calib_file}")
    else:
        print(f"No calibration file found at {calib_file}, skipping calibration set.")

    # 5) Connect to the bus
    bus.connect()
    print(f"Connected to Feetech bus on port: {bus.port}")

    # 5b) Apply offset to servo EEPROM so the servo’s Present_Position is hardware-shifted
    if calibration_dict is not None:
        print("Applying offsets from calibration object to servo EEPROM. Be careful—this is permanent!")
        apply_feetech_offsets_from_calibration(bus, calibration_dict)

    # 6) Disable torque on all motors so you can move them freely by hand
    bus.write("Torque_Enable", 0, motor_names=bus.motor_names)
    print("Torque disabled on all joints.")

    try:
        print("\nPress Ctrl+C to quit.\n")
        while True:
            # (a) Read *raw* positions (no calibration). This bypasses bus.apply_calibration
            raw_positions = bus.read_with_motor_ids(
                bus.motor_models, bus.motor_indices, data_name="Present_Position"
            )

            # (b) Read *already-homed* positions (calibration is applied automatically inside bus.read())
            homed_positions = bus.read("Present_Position")

            # Print them side by side
            for i, name in enumerate(bus.motor_names):
                motor_idx, model = bus.motors[name]

                raw_ticks = raw_positions[i]  # 0..4095
                homed_val = homed_positions[i]  # degrees or % if linear

                # Manually compute "adjusted ticks" from raw ticks
                manual_adjusted = adjusted_to_homing_ticks(raw_ticks, model, bus, motor_idx)
                # Convert to degrees
                manual_degs = convert_ticks_to_degrees(manual_adjusted, model)

                # Convert that deg back to ticks
                manual_ticks = convert_degrees_to_ticks(manual_degs, model)
                # Then invert them using offset & bus drive mode
                inv_ticks = adjusted_to_motor_ticks(manual_ticks, model, bus, motor_idx)

                print(
                    f"{name:15s} | "
                    f"RAW={raw_ticks:4d} | "
                    f"HOMED_FROM_READ={homed_val:7.2f} | "
                    f"HOMED_TICKS={manual_adjusted:6d} | "
                    f"MANUAL_ADJ_DEG={manual_degs:7.2f} | "
                    f"MANUAL_ADJ_TICKS={manual_ticks:6d} | "
                    f"INV_TICKS={inv_ticks:4d} "
                )
            print("----------------------------------------------------")
            time.sleep(0.25)  # slow down loop
    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting. Disconnecting bus...")
        bus.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Feetech positions.")
    parser.add_argument(
        "--arm",
        type=str,
        choices=["main_leader", "main_follower"],
        default="main_leader",
        help="Which arm to debug: 'main_leader' or 'main_follower'.",
    )
    args = parser.parse_args()

    # 1) Instantiate the config you want (So100RobotConfig, or whichever your project uses).
    cfg = So100RobotConfig()

    # 2) Call the function with (cfg, args.arm, args.apply_offsets)
    debug_feetech_positions(cfg, arm_arg=args.arm)

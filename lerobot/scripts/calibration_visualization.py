import argparse
import json
import time
from pathlib import Path

from lerobot.common.robot_devices.motors.feetech import (
    FeetechMotorsBus,
    adjusted_to_homing_ticks,
    adjusted_to_motor_ticks,
    convert_steps_to_degrees,
)

# Replace this import with your real config class import
# (e.g., So100RobotConfig or any other)
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config


def debug_feetech_positions(cfg, arm_arg: str):
    """
    Reads each joint’s (1) raw ticks, (2) homed ticks, (3) degrees, and (4) invert-adjusted ticks.

    :param cfg: A config object (e.g. So100RobotConfig).
    :param arm_arg: One of "main_leader" or "main_follower".
    """
    # 1) Make the Robot from your config
    #    e.g. `So100RobotConfig` might already be the "robot object",
    #    or if it is purely a config structure, do: robot = make_robot_from_config(cfg).
    robot = make_robot_from_config(cfg)

    # 2) Parse which arm we want: 'main_leader' or 'main_follower'
    if arm_arg not in ("main_leader", "main_follower"):
        raise ValueError("Please specify --arm=main_leader or --arm=main_follower")

    bus_config = robot.leader_arms["main"] if arm_arg == "main_leader" else robot.follower_arms["main"]

    # 3) Create the Feetech bus from that config
    bus = FeetechMotorsBus(bus_config)

    # 4) (Optional) Load calibration if it exists
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

    # 6) Disable torque on all motors so you can move them freely by hand
    bus.write("Torque_Enable", 0, motor_names=bus.motor_names)
    print("Torque disabled on all joints.")

    try:
        print("\nPress Ctrl+C to quit.\n")
        while True:
            # (a) Read *raw* positions (no calibration)
            raw_positions = bus.read_with_motor_ids(
                bus.motor_models, bus.motor_indices, data_name="Present_Position"
            )

            # (b) Read *already-homed* positions (calibration is applied automatically)
            homed_positions = bus.read("Present_Position")

            # Print them side by side
            for i, name in enumerate(bus.motor_names):
                motor_idx, model = bus.motors[name]

                raw_ticks = raw_positions[i]  # 0..4095
                homed_val = homed_positions[i]  # degrees or % if linear

                # If you want to see how offset is used inside bus.read(), do it manually:
                offset = 0
                if bus.calibration and name in bus.calibration["motor_names"]:
                    offset_idx = bus.calibration["motor_names"].index(name)
                    offset = bus.calibration["homing_offset"][offset_idx]

                # Manually compute "adjusted ticks" from raw ticks
                manual_adjusted = adjusted_to_homing_ticks(raw_ticks, offset, model, bus, motor_idx)
                # Convert to degrees
                manual_degs = convert_steps_to_degrees(manual_adjusted, [model])[0]
                # Invert
                inv_ticks = adjusted_to_motor_ticks(manual_adjusted, offset, model, bus, motor_idx)

                print(
                    f"{name:15s} | "
                    f"RAW={raw_ticks:4d} | "
                    f"HOMED={homed_val:7.2f} | "
                    f"MANUAL_ADJ={manual_adjusted:6d} | "
                    f"DEG={manual_degs:7.2f} | "
                    f"INV={inv_ticks:4d}"
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

    # 2) Call the function with (cfg, args.arm)
    debug_feetech_positions(cfg, arm_arg=args.arm)

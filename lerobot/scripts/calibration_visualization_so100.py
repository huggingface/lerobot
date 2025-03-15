"""
usage:

```python
python lerobot/scripts/calibration_visualization_so100.py \
    --teleop.type=so100 \
    --teleop.port=/dev/tty.usbmodem58760430541

python lerobot/scripts/calibration_visualization_so100.py \
    --robot.type=so100 \
    --robot.port=/dev/tty.usbmodem585A0084711
```
"""

import time
from dataclasses import dataclass

import draccus

from lerobot.common.motors.feetech.feetech import (
    adjusted_to_homing_ticks,
    adjusted_to_motor_ticks,
    convert_degrees_to_ticks,
    convert_ticks_to_degrees,
)
from lerobot.common.robots import RobotConfig
from lerobot.common.robots.so100 import SO100Robot, SO100RobotConfig  # noqa: F401
from lerobot.common.teleoperators import TeleoperatorConfig
from lerobot.common.teleoperators.so100 import SO100Teleop, SO100TeleopConfig  # noqa: F401


@dataclass
class DebugFeetechConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Select a single device.")


@draccus.wrap()
def debug_feetech_positions(cfg: DebugFeetechConfig):
    """
    Reads each joint's (1) raw ticks, (2) homed ticks, (3) degrees, and (4) invert-adjusted ticks.
    """
    device = SO100Teleop(cfg.teleop) if cfg.teleop else SO100Robot(cfg.robot)
    device.connect()

    # Disable torque on all motors so you can move them freely by hand
    device.arm.write("Torque_Enable", 0, motor_names=device.arm.motor_names)
    print("Torque disabled on all joints.")

    try:
        print("\nPress Ctrl+C to quit.\n")
        while True:
            # Read *raw* positions (no calibration).
            raw_positions = device.arm.read_with_motor_ids(
                device.arm.motor_models, device.arm.motor_indices, data_name="Present_Position"
            )

            # Read *already-homed* positions
            homed_positions = device.arm.read("Present_Position")

            for i, name in enumerate(device.arm.motor_names):
                motor_idx, model = device.arm.motors[name]

                raw_ticks = raw_positions[i]  # 0..4095
                homed_val = homed_positions[i]  # degrees or % if linear

                # Manually compute "adjusted ticks" from raw ticks
                manual_adjusted = adjusted_to_homing_ticks(raw_ticks, model, device.arm, motor_idx)
                # Convert to degrees
                manual_degs = convert_ticks_to_degrees(manual_adjusted, model)

                # Convert that deg back to ticks
                manual_ticks = convert_degrees_to_ticks(manual_degs, model)
                # Then invert them using offset & bus drive mode
                inv_ticks = adjusted_to_motor_ticks(manual_ticks, model, device.arm, motor_idx)

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
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting. Disconnecting bus...")
        device.disconnect()


if __name__ == "__main__":
    debug_feetech_positions()

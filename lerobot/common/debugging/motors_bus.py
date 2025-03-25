"""
Usage example

```python
from lerobot.common.debugging.motors_bus import visualize_motors_bus
from lerobot.common.robots.so100 import SO100Robot, SO100RobotConfig

cfg = SO100RobotConfig(port="/dev/tty.usbmodem58760430541")
so100 = SO100Robot(cfg)

visualize_motors_bus(so100.arm)
```
"""

import time

from lerobot.common.motors import MotorsBus
from lerobot.common.motors.feetech.feetech_calibration import (
    adjusted_to_homing_ticks,
    adjusted_to_motor_ticks,
    convert_degrees_to_ticks,
    convert_ticks_to_degrees,
)


def visualize_motors_bus(motors_bus: MotorsBus):
    """
    Reads each joint's (1) raw ticks, (2) homed ticks, (3) degrees, and (4) invert-adjusted ticks.
    """
    if not motors_bus.is_connected:
        motors_bus.connect()

    # Disable torque on all motors so you can move them freely by hand
    for id_ in motors_bus.ids:
        motors_bus.write("Torque_Enable", id_, 0)

    print("Torque disabled on all joints.")

    try:
        print("\nPress Ctrl+C to quit.\n")
        while True:
            # Read *raw* positions (no calibration).
            start = time.perf_counter()
            raw_positions = motors_bus.sync_read("Present_Position", raw_values=True)
            read_s = time.perf_counter() - start

            # # Read *already-homed* positions
            # homed_positions = motor_bus.read("Present_Position")

            print(f"read_s: {read_s * 1e3:.2f}ms ({1 / read_s:.0f} Hz)")
            for name, raw_ticks in raw_positions.items():
                idx = motors_bus.motors[name].id
                model = motors_bus.motors[name].model

                # homed_val = homed_positions[i]  # degrees or % if linear

                # Manually compute "adjusted ticks" from raw ticks
                manual_adjusted = adjusted_to_homing_ticks(raw_ticks, model, motors_bus, idx)
                # Convert to degrees
                manual_degs = convert_ticks_to_degrees(manual_adjusted, model)

                # Convert that deg back to ticks
                manual_ticks = convert_degrees_to_ticks(manual_degs, model)
                # Then invert them using offset & bus drive mode
                inv_ticks = adjusted_to_motor_ticks(manual_ticks, model, motors_bus, idx)

                print(
                    f"{name:15s} | "
                    f"RAW={raw_ticks:4d} | "
                    # f"HOMED_FROM_READ={homed_val:7.2f} | "
                    f"HOMED_TICKS={manual_adjusted:6d} | "
                    f"MANUAL_ADJ_DEG={manual_degs:7.2f} | "
                    f"MANUAL_ADJ_TICKS={manual_ticks:6d} | "
                    f"INV_TICKS={inv_ticks:4d} "
                )
            print("----------------------------------------------------")
            # time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting. Disconnecting bus...")
        motors_bus.disconnect()


if __name__ == "__main__":
    visualize_motors_bus()

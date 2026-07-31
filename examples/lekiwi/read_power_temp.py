#!/usr/bin/env python
"""Read-only telemetry: temperature / voltage / current of every LeKiwi servo.

Does NOT enable torque or move anything. Run with the serial port free (no host owning it):

    uv run python -m examples.lekiwi.read_power_temp --port /dev/ttyACM0
"""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

MOTORS = {
    "arm_shoulder_pan": 1,
    "arm_shoulder_lift": 2,
    "arm_elbow_flex": 3,
    "arm_wrist_flex": 4,
    "arm_wrist_roll": 5,
    "arm_gripper": 6,
    "base_left_wheel": 7,
    "base_back_wheel": 8,
    "base_right_wheel": 9,
}

# STS3215 scaling: current register is ~6.5 mA per LSB.
CURRENT_LSB_MA = 6.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=LeKiwiConfig().port)
    args = parser.parse_args()

    bus = FeetechMotorsBus(
        port=args.port,
        motors={name: Motor(mid, "sts3215", MotorNormMode.RANGE_M100_100) for name, mid in MOTORS.items()},
    )
    # Skip the strict handshake and pin the servo bus baudrate; these STS3215 run at 1 Mbaud.
    bus.connect(handshake=False)
    bus.set_baudrate(1_000_000)

    print(f"\n== LeKiwi servo telemetry ({args.port}) ==")
    header = f"{'motor':20s} {'id':>3s} {'temp':>7s} {'voltage':>9s} {'current':>9s}"
    print(header)
    print("-" * len(header))

    total_power_w = 0.0
    reachable = 0
    for name, mid in MOTORS.items():
        try:
            temp_c = bus.read("Present_Temperature", name, normalize=False, num_retry=3)
            volt = bus.read("Present_Voltage", name, normalize=False, num_retry=3) / 10.0
            raw_i = bus.read("Present_Current", name, normalize=False, num_retry=3)
            # 16-bit signed
            if raw_i > 32767:
                raw_i -= 65536
            curr_a = abs(raw_i) * CURRENT_LSB_MA / 1000.0
            power_w = volt * curr_a
            total_power_w += power_w
            reachable += 1
            print(f"{name:20s} {mid:>3d} {temp_c:>5.0f}C {volt:>7.1f}V {curr_a * 1000:>6.0f}mA  ({power_w:4.1f} W)")
        except Exception as e:
            print(f"{name:20s} {mid:>3d}   -- unreachable: {type(e).__name__}: {e}")

    print("-" * len(header))
    print(f"reachable servos: {reachable}/{len(MOTORS)}   approx total servo power: {total_power_w:.1f} W")

    bus.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

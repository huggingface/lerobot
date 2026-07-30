#!/usr/bin/env python
"""Quick motion test for the LeKiwi Orbbec camera pan/tilt servos.

Pans `camera_pan` (10) back and forth, then tilts `camera_tilt` (11) back and forth,
then returns both to center. Only the two camera servos need to be connected
(board -> camera_pan -> camera_tilt); the arm/wheels don't have to be present.

Requires the camera servos to already have IDs assigned (setup_camera_motors.py) and
be calibrated (calibrate_camera_motors.py), so goal positions are clamped to the
calibrated range by the servo firmware.

Usage:
    uv run python wiggle_camera_motors.py --port /dev/tty.usbmodem58760431551 --id my_lekiwi
"""

import argparse
import math
import time


def sweep(bus, motor: str, amplitude_deg: float, cycles: int, period_s: float, hz: float) -> None:
    """Oscillate `motor` as a sine around its center (0 deg) then return to center."""
    print(f"  {motor}: +/-{amplitude_deg:.1f} deg, {cycles} cycle(s)")
    t0 = time.perf_counter()
    total_s = cycles * period_s
    dt = 1.0 / hz
    while True:
        t = time.perf_counter() - t0
        if t >= total_s:
            break
        deg = amplitude_deg * math.sin(2 * math.pi * t / period_s)
        bus.write("Goal_Position", motor, deg)
        time.sleep(dt)
    bus.write("Goal_Position", motor, 0.0)
    time.sleep(0.5)


def calibrated_amplitude_deg(bus, motor: str, fraction: float = 0.8) -> float:
    """Half the calibrated range (in degrees) scaled by `fraction`, so we stay off the limits."""
    cal = bus.calibration[motor]
    model = bus.motors[motor].model
    max_res = bus.model_resolution_table[model] - 1
    span_deg = (cal.range_max - cal.range_min) * 360.0 / max_res
    return (span_deg / 2.0) * fraction


def main():
    parser = argparse.ArgumentParser(description="Pan/tilt motion test for the LeKiwi Orbbec camera servos.")
    parser.add_argument("--port", required=True, help="USB port of the LeKiwi MotorsBus.")
    parser.add_argument("--id", default=None, help="Robot id used to locate the calibration file.")
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="Sweep amplitude in degrees (default: 80%% of each servo's calibrated half-range).",
    )
    parser.add_argument("--cycles", type=int, default=2, help="Number of back-and-forth cycles per servo.")
    parser.add_argument("--period", type=float, default=2.0, help="Seconds per full back-and-forth cycle.")
    args = parser.parse_args()

    from lerobot.motors.feetech import OperatingMode
    from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig

    config = LeKiwiConfig(port=args.port, use_camera_head=True)
    if args.id is not None:
        config.id = args.id
    robot = LeKiwi(config)

    if not robot.calibration:
        raise SystemExit(
            "No calibration found for this id. Run calibrate_camera_motors.py first so goal positions "
            "are clamped to a safe range."
        )

    # handshake=False so only the two camera servos need to be connected.
    robot.bus.connect(handshake=False)
    try:
        # Position mode + torque on, for the camera servos only.
        robot.bus.disable_torque(robot.camera_motors)
        for name in robot.camera_motors:
            robot.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        robot.bus.enable_torque(robot.camera_motors)

        # Center both before moving.
        for name in robot.camera_motors:
            robot.bus.write("Goal_Position", name, 0.0)
        time.sleep(0.8)

        pan_amp = (
            args.amplitude if args.amplitude is not None else calibrated_amplitude_deg(robot.bus, "camera_pan")
        )
        tilt_amp = (
            args.amplitude
            if args.amplitude is not None
            else calibrated_amplitude_deg(robot.bus, "camera_tilt")
        )

        print("Panning...")
        sweep(robot.bus, "camera_pan", pan_amp, args.cycles, args.period, hz=50)

        print("Tilting...")
        sweep(robot.bus, "camera_tilt", tilt_amp, args.cycles, args.period, hz=50)

        print("Done, returning to center.")
    finally:
        # disable_torque=False: only the camera motors are connected, so disabling torque on the
        # absent arm/wheel motors would raise a ConnectionError.
        robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read SO-100/SO-101 temperatures without configuring motors or changing torque."""

import argparse
import logging
import math
import platform
import shutil
import subprocess
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)

logger = logging.getLogger(__name__)


def send_notification(message: str) -> None:
    """Warn in the terminal and attempt a desktop notification without interrupting polling."""
    logger.warning(message)
    title = "LeRobot: let the motors cool down"
    command: list[str] = []
    if platform.system() == "Darwin":
        # Pass text as arguments, rather than interpolating it into AppleScript code.
        command = [
            "osascript",
            "-e",
            "on run argv\ndisplay notification (item 1 of argv) with title (item 2 of argv)\nend run",
            message,
            title,
        ]
    elif platform.system() == "Linux" and shutil.which("notify-send"):
        command = ["notify-send", title, message]
    if command:
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=5)  # nosec B603
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Desktop notification unavailable: %s", exc)


def monitor_temperature(
    port: str, robot_id: str, threshold: int, interval: float, display_mode: str | None = None
) -> None:
    """Poll all six sensors, warn on each threshold crossing, and optionally plot telemetry.

    Args:
        port: Serial port, exclusively owned by this process.
        robot_id: Existing robot identifier; calibration is not required for temperature reads.
        threshold: Warning temperature in degrees Celsius.
        interval: Seconds between polls.
        display_mode: Optional existing visualization backend: ``rerun`` or ``foxglove``.
    """
    bus = SO101Follower(SO101FollowerConfig(port=port, id=robot_id)).bus
    hot: set[str] = set()
    try:
        bus.connect()
        if display_mode:
            init_visualization(display_mode, session_name="lerobot_temperature")
        while True:
            temperatures = bus.sync_read("Present_Temperature", normalize=False)
            if set(temperatures) != set(bus.motors) or any(
                not math.isfinite(value) or not 0 <= value <= 150 for value in temperatures.values()
            ):
                raise ValueError("Missing or invalid temperature telemetry")
            print(", ".join(f"{name}={value:g} C" for name, value in temperatures.items()), flush=True)
            if display_mode:
                log_visualization_data(
                    observation={f"{name}.temperature": float(value) for name, value in temperatures.items()},
                    display_mode=display_mode,
                )
            current_hot = {name for name, value in temperatures.items() if value >= threshold}
            for name in sorted(current_hot - hot):
                send_notification(
                    f"{name}: {temperatures[name]:g} C. Pause safely and let the motor cool down."
                )
            hot = current_hot
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    except Exception:
        send_notification("Temperature monitoring failed; current motor temperatures are unknown.")
        raise
    finally:
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        finally:
            if display_mode:
                shutdown_visualization(display_mode)


def main() -> None:
    """Run the standalone temperature monitor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="temperature_monitor")
    parser.add_argument(
        "--threshold", type=int, default=52, help="Warning temperature in Celsius (default: 52)."
    )
    parser.add_argument("--interval", type=float, default=1, help="Seconds between polls (default: 1).")
    parser.add_argument(
        "--display-mode", choices=("rerun", "foxglove"), help="Optional live temperature plots."
    )
    args = parser.parse_args()
    if not 1 <= args.threshold <= 150 or not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("require threshold in 1..150 and a finite positive interval")
    logging.basicConfig(level=logging.INFO)
    monitor_temperature(args.port, args.id, args.threshold, args.interval, args.display_mode)


if __name__ == "__main__":
    main()

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

"""One-second motor temperature sampling and staged operator warnings."""

import logging
import math
import platform
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from lerobot.motors.motors_bus import SerialMotorsBus
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower import SOFollower

logger = logging.getLogger(__name__)
STAGES = (
    (50, "Heating"),
    (55, "Overheating"),
    (65, "Excessive heating: pause safely and cool down before continuing"),
)


def _desktop_notification(message: str) -> None:
    """Deliver a best-effort desktop notification off the control thread."""
    title = "LeRobot motor temperature"
    command: list[str] = []
    if platform.system() == "Darwin":
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


class MotorTemperatureMonitor:
    """Read an already-open bus once a second; never send motion or torque commands.

    Only desktop notifications run in a worker. Call ``poll`` from the thread that
    owns the bus, between control transactions. Call ``close`` when the session ends.
    """

    def __init__(self, bus: SerialMotorsBus) -> None:
        self.bus = bus
        self.next_poll = 0.0
        self.warned: dict[str, int] = {}
        self.unavailable = False
        self.notifications = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motor-temperature")

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self.notifications.submit(_desktop_notification, message)

    def poll(self) -> dict[str, float]:
        """Return fresh visualization scalars, or an empty dict between polls/on failure."""
        now = time.monotonic()
        if now < self.next_poll:
            return {}
        self.next_poll = now + 1.0
        try:
            readings = self.bus.sync_read("Present_Temperature", normalize=False)
            if set(readings) != set(self.bus.motors) or any(
                not math.isfinite(value) or not 0 <= value <= 150 for value in readings.values()
            ):
                raise ValueError("Missing or invalid temperature telemetry")
        except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
            if not self.unavailable:
                self._warn(f"Motor temperatures UNAVAILABLE: {exc}. Monitoring will retry in one second.")
            self.unavailable = True
            return {}
        self.unavailable = False
        alerts = []
        for name, value in readings.items():
            stage = sum(value >= threshold for threshold, _ in STAGES)
            if stage == 0:
                self.warned.pop(name, None)
            elif stage > self.warned.get(name, 0):
                alerts.append(f"{name}={value:g} C — {STAGES[stage - 1][1]}")
                self.warned[name] = stage
        if alerts:
            self._warn("; ".join(alerts))
        logger.info(
            "Motor temperatures: %s", ", ".join(f"{name}={value:g} C" for name, value in readings.items())
        )
        return {f"{name}.temperature": float(value) for name, value in readings.items()}

    def close(self) -> None:
        """Stop accepting desktop notifications; leave the bus with its owner."""
        self.notifications.shutdown(wait=False)


def make_temperature_monitor(robot: Robot) -> MotorTemperatureMonitor:
    """Use the SO follower's existing bus without opening another serial connection."""
    if not isinstance(robot, SOFollower):
        raise ValueError("Motor temperature monitoring currently supports SO-100/SO-101 followers.")
    return MotorTemperatureMonitor(robot.bus)

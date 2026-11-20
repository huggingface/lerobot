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

"""Plot six SO-follower temperatures every second and warn at 50, 55 and 65 Celsius."""

import argparse
import logging
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.motor_temperature import MotorTemperatureMonitor
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


def monitor_temperature(port: str, robot_id: str, display_mode: str) -> None:
    """Open the bus for standalone monitoring, without configuring motors or changing torque."""
    bus = SO101Follower(SO101FollowerConfig(port=port, id=robot_id)).bus
    monitor = MotorTemperatureMonitor(bus)
    initialized = False
    try:
        bus.connect()
        init_visualization(display_mode, session_name="lerobot_temperature")
        initialized = True
        while True:
            temperatures = monitor.poll()
            if temperatures:
                log_visualization_data(display_mode, observation=temperatures)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        finally:
            if initialized:
                shutdown_visualization(display_mode)


def main() -> None:
    """Run the standalone monitor, using Rerun by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="temperature_monitor")
    parser.add_argument("--display-mode", choices=("rerun", "foxglove"), default="rerun")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    monitor_temperature(args.port, args.id, args.display_mode)


if __name__ == "__main__":
    main()

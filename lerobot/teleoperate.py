# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np
import rerun as rr

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    so101_mujoco,
)
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun

from .common.teleoperators import gamepad, koch_leader, so100_leader, so101_leader, keyboard  # noqa: F401
from lerobot.common.transport.udp_transport import UDPTransportReceiver, UDPTransportSender


@dataclass
class NetworkTransportConfig:
    """Optional networking layer configuration.

    If ``server`` is populated we act as *client* and stream actions to that endpoint.
    Otherwise we act as *server* (follower) and listen on ``port`` for incoming actions.
    """

    # Remote endpoint ("<ip>:<port>") we send actions to when *leader*.
    server: str | None = None
    # UDP port we listen on when *follower*.
    port: int = 5555

    # Maximum frames per second transmitted over the wire (defaults to TeleoperateConfig.fps)
    net_fps: int | None = None


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Optional UDP networking layer
    transport: NetworkTransportConfig | None = None


def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        if display_data:
            observation = robot.get_observation()
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation_{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation_{obs}", rr.Image(val), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action_{act}", rr.Scalar(val))

        robot.send_action(action)
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 5)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _client_loop(teleop: Teleoperator, sender: UDPTransportSender, fps: int, duration: float | None = None):
    """Collect actions from *teleop* and forward them to *sender*."""
    start_t = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        sender.send(action)

        # Throttle FPS
        dt_s = time.perf_counter() - loop_start
        busy_wait(max(0.0, 1 / fps - dt_s))

        if duration is not None and time.perf_counter() - start_t >= duration:
            return


def _server_loop(receiver: UDPTransportReceiver, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None):
    """Receive actions from *receiver* and send them to *robot*."""
    display_len = max(len(key) for key in robot.action_features)
    start_t = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = receiver.recv()
        if display_data:
            observation = robot.get_observation()
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation_{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation_{obs}", rr.Image(val), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action_{act}", rr.Scalar(val))

        robot.send_action(action)

        dt_s = time.perf_counter() - loop_start
        busy_wait(max(0.0, 1 / fps - dt_s))

        loop_s = time.perf_counter() - loop_start
        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start_t >= duration:
            return

        move_cursor_up(len(action) + 5)


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # ------------------------------------------------------------------
    # Determine operating mode
    # ------------------------------------------------------------------
    transport_cfg = cfg.transport
    is_network_client = transport_cfg is not None and transport_cfg.server is not None
    is_network_server = transport_cfg is not None and transport_cfg.server is None

    if is_network_client and is_network_server:
        raise ValueError("Transport configuration ambiguous – cannot be both client and server.")

    if is_network_client:
        if cfg.teleop is None:
            raise ValueError("Client mode requires a local teleoperator (leader arm)")

        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()

        # transport_cfg is guaranteed not None in client mode – assert to appease linters.
        assert transport_cfg is not None and transport_cfg.server is not None
        fps = transport_cfg.net_fps or cfg.fps
        sender = UDPTransportSender(transport_cfg.server)

        try:
            _client_loop(teleop, sender, fps, duration=cfg.teleop_time_s)
        except KeyboardInterrupt:
            pass
        finally:
            teleop.disconnect()
        return

    # ------------------------------------------------------------------
    # Network server path (follower)
    # ------------------------------------------------------------------
    if is_network_server:
        if cfg.robot is None:
            raise ValueError("Server mode requires a robot to control.")

        # transport_cfg is guaranteed not None in server mode by construction.
        assert transport_cfg is not None
        receiver = UDPTransportReceiver(transport_cfg.port)
        robot = make_robot_from_config(cfg.robot)
        robot.connect()

        try:
            _server_loop(receiver, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
        except KeyboardInterrupt:
            pass
        finally:
            robot.disconnect()
        return

    # ------------------------------------------------------------------
    # Original local mode – teleop & robot on the same machine.
    # ------------------------------------------------------------------
    if cfg.teleop is None or cfg.robot is None:
        raise ValueError("Local mode requires both a teleoperator and a robot configuration.")

    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # draccus wraps the function and handles CLI parsing, so we intentionally call it
    # without arguments.
    teleoperate()  # type: ignore[call-arg]

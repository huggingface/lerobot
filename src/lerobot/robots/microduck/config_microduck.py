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

from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("microduck")
@dataclass
class MicroDuckConfig(RobotConfig):
    """Pollen Robotics Microduck, driven through its onboard locomotion policy.

    The duck walks with an onboard RL policy, so LeRobot commands *intents* (a body twist and a head
    pose), never joint targets. Those intents go to ``robotd`` over its JSON-RPC socket, forwarded from
    the robot with ``ssh -N -L /tmp/robotd.sock:/run/robotd.sock microduck@<remote_ip>``.

    The robot stores its calibration. LeRobot does not calibrate the motors.
    """

    # IP of the duck on the LAN (``duckctl ip`` prints it). Used for the camera snapshot endpoint.
    remote_ip: str = "192.168.1.42"

    # robotd's socket as seen from this machine: a Unix socket path, or ``host:port`` for a TCP forward.
    robotd_address: str = "/tmp/robotd.sock"  # nosec B108: a forwarded socket path, not a temp file
    # updaterd's socket, only needed to fetch policies from the Hub (``MicroDuck.load_policy``).
    updaterd_address: str | None = None
    rpc_timeout_s: float = 5.0

    # Rate of the robot.state stream; robotd's control loop runs at 50 Hz.
    state_hz: int = 50

    # Ask the onboard policy to drive (Limp -> Homing -> Ready) when connecting.
    enable_on_connect: bool = True
    # Cut torque when disconnecting. Off by default: a relaxed duck collapses.
    relax_on_disconnect: bool = False

    # Twist limits, matching the onboard pad defaults. robotd does not clamp robot.move itself.
    max_vx: float = 0.3  # m/s
    max_vy: float = 0.3  # m/s
    max_vyaw: float = 1.5  # rad/s
    # Head pose limit applied to every head joint, rad.
    max_head: float = 0.8

    # Camera, read from mediad's ``GET http://<remote_ip>:<camera_port>/frame`` (upright PNG).
    # Set ``use_camera=False`` to record state only. Frames are resized to (camera_height, camera_width).
    use_camera: bool = True
    camera_port: int = 8080
    camera_width: int = 360
    camera_height: int = 640
    camera_timeout_s: float = 2.0

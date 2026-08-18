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

"""Config for the cloud-side WebRTCProxyRobot.

The proxy declares its observation/action schema from config alone (no hardware
present cloud-side), mirroring ``so_follower``: action/state are ``"<motor>.pos"``
floats and each camera is an ``HxWx3`` frame keyed by camera name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RobotConfig

# SO-100/101 follower motors in bus order (matches so_follower).
SO100_MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class WebRTCCameraSpec:
    """One camera streamed over the media track. Drives the obs feature shape."""

    height: int
    width: int
    fps: int = 30


@RobotConfig.register_subclass("webrtc_proxy")
@dataclass
class WebRTCProxyRobotConfig(RobotConfig):
    """Cloud-side proxy: real hardware lives on the user's robot, reached over WebRTC.

    M1 (loopback) ignores ``signaling_url`` and pairs the two peers in-process.
    Real deployments (M3+) point ``signaling_url`` at the K8s WebSocket signaler.
    """

    # Schema (must match the robot-side real robot exactly so dataset/policy shapes line up).
    motors: list[str] = field(default_factory=lambda: list(SO100_MOTORS))
    # camera_name -> spec. Default: one wrist+front pair is common, but keep it minimal here.
    cameras: dict[str, WebRTCCameraSpec] = field(
        default_factory=lambda: {"front": WebRTCCameraSpec(height=480, width=640, fps=30)}
    )

    # Capture / transport tuning.
    capture_fps: int = 30
    # P0 safety: robot watchdog safes the arm if no action arrives within this window.
    action_timeout_s: float = 0.5
    # Max capture-time skew tolerated when pairing state<->frame (telemetry/QA threshold).
    pair_tolerance_s: float = 0.1
    # How long connect() waits for the WebRTC link + first observation.
    connect_timeout_s: float = 10.0

    # Signaling. None / "loopback" => in-process loopback (demo + unit tests).
    # "ws://host:port/ws" => connect to a WebSocket signaling relay as the controller
    # and reach a real robot daemon (no in-process capture agent). See signaling_server.py.
    signaling_url: str | None = None
    # Session id pairing this controller with its robot daemon on the relay.
    session_id: str = "default"
    # Shared token presented to the (public) signaling relay. See signaling_server.py.
    signaling_token: str | None = None
    # ICE servers for the controller's peer connection. Empty => host candidates only
    # (loopback / same-host two-process). M4 injects STUN/TURN for real public-net peers.
    ice_servers: list[str] = field(default_factory=list)
    # Transport backend: "aiortc" (default, self-contained P2P) | "livekit" (SFU; both
    # the controller and the robot daemon must use the same backend).
    transport_backend: str = "aiortc"
    # Required when transport_backend == "livekit": the LiveKit server URL + a JWT.
    livekit_url: str | None = None
    livekit_token: str | None = None

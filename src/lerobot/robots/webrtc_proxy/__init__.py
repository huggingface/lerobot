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

"""WebRTCProxyRobot: cloud-side proxy for a robot-tethered real robot (see README.md).

The config imports with no extra deps so draccus registration always works. The
robot class (and its ``aiortc`` dependency) loads lazily on first access so the
``robots`` package still imports without the ``lerobot[webrtc]`` extra.
"""

from .configuration_webrtc_proxy import WebRTCCameraSpec, WebRTCProxyRobotConfig

__all__ = [
    "CameraLayoutMismatchError",
    "WebRTCCameraSpec",
    "WebRTCProxyRobot",
    "WebRTCProxyRobotConfig",
]


def __getattr__(name: str):
    if name in ("WebRTCProxyRobot", "CameraLayoutMismatchError"):
        from . import proxy_robot

        return getattr(proxy_robot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#!/usr/bin/env python

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

from ..configs import CameraConfig, ColorMode

__all__ = ["ZMQCameraConfig", "ColorMode"]


@CameraConfig.register_subclass("zmq")
@dataclass
class ZMQCameraConfig(CameraConfig):
    """Configuration for a camera served over a ZeroMQ socket by `ImageServer`.

    Use this to read frames from a camera attached to a different machine (e.g. a Raspberry Pi on a
    robot), which streams JPEG-encoded frames to this config's `server_address`/`port` over ZMQ.

    Args:
        server_address (`str`):
            Address of the machine running the image server, e.g. `"192.168.1.50"`.
        port (`int`, *optional*, defaults to 5555):
            TCP port the image server is publishing on. Must be between 1 and 65535.
        camera_name (`str`, *optional*, defaults to `"zmq_camera"`):
            Name used to identify this camera in logs and observation keys.
        color_mode (`ColorMode`, *optional*, defaults to `ColorMode.RGB`):
            Color mode for the decoded frames.
        timeout_ms (`int`, *optional*, defaults to 5000):
            How long to wait for a frame before raising a timeout, in milliseconds. Must be positive.
        warmup_s (`int`, *optional*, defaults to 1):
            Time spent reading frames before returning from connect, in seconds.
        fps (`int`, *optional*):
            Requested frames per second. `None` leaves it at the server's own rate.
        width (`int`, *optional*):
            Requested frame width in pixels. `None` leaves it at the server's own resolution.
        height (`int`, *optional*):
            Requested frame height in pixels. `None` leaves it at the server's own resolution.
    """

    server_address: str
    port: int = 5555
    camera_name: str = "zmq_camera"
    color_mode: ColorMode = ColorMode.RGB
    timeout_ms: int = 5000
    warmup_s: int = 1

    def __post_init__(self) -> None:
        """Normalize `color_mode` and validate the socket settings.

        Raises:
            ValueError: If `timeout_ms` is not positive, `server_address` is empty, or `port` is outside
                the 1-65535 range.
        """
        self.color_mode = ColorMode(self.color_mode)

        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be positive, but {self.timeout_ms} is provided.")

        if not self.server_address:
            raise ValueError("`server_address` cannot be empty.")

        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"`port` must be between 1 and 65535, but {self.port} is provided.")

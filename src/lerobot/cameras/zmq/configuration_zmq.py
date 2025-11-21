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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode

__all__ = ["ZMQCameraConfig", "ColorMode"]


@CameraConfig.register_subclass("zmq")
@dataclass
class ZMQCameraConfig(CameraConfig):
    """Configuration class for ZMQ-based remote camera streams.

    This class provides configuration options for cameras accessed through ZeroMQ (ZMQ),
    supporting remote camera streams over the network. The server must be running and
    streaming JPEG-encoded images over a ZMQ PUB socket.

    Example configurations:
    ```python
    # Basic configuration
    ZMQCameraConfig(
        server_address="192.168.123.164",
        port=5554,
        camera_name="remote_cam_1"
    )

    # With custom resolution
    ZMQCameraConfig(
        server_address="10.0.0.100",
        port=5555,
        camera_name="lab_cam",
        width=1280,
        height=480,
        fps=30
    )
    ```

    Attributes:
        server_address: IP address or hostname of the ZMQ image server.
        port: Port number where the ZMQ server is publishing images.
        camera_name: Identifier name for this camera (for logging/debugging).
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        timeout_ms: Timeout in milliseconds for receiving frames. Defaults to 1000ms.

    Note:
        - The server must be streaming JPEG-encoded images over ZMQ PUB socket.
        - Width and height should match the expected output dimensions from the server.
        - FPS is informational and doesn't control the server's frame rate.
    """

    server_address: str
    port: int = 5554
    camera_name: str = "zmq_camera"
    color_mode: ColorMode = ColorMode.RGB
    timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be positive, but {self.timeout_ms} is provided.")

        if not self.server_address:
            raise ValueError("`server_address` cannot be empty.")

        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"`port` must be between 1 and 65535, but {self.port} is provided.")


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
from enum import Enum

from ..configs import CameraConfig, ColorMode, Cv2Rotation


class KinectPipeline(str, Enum):
    """Available processing pipelines for Kinect v2."""

    AUTO = "auto"
    CUDA = "cuda"
    OPENCL = "opencl"
    OPENGL = "opengl"
    CPU = "cpu"


@CameraConfig.register_subclass("kinect")
@dataclass
class KinectCameraConfig(CameraConfig):
    """Configuration class for Microsoft Kinect v2 cameras.

    This class provides specialized configuration options for Kinect v2 cameras,
    including support for depth, IR sensing, and GPU-accelerated processing pipelines.

    Example configurations:
    ```python
    # Basic configuration (auto-detect first Kinect)
    KinectCameraConfig(device_index=0, fps=30)

    # High-performance with CUDA pipeline
    KinectCameraConfig(device_index=0, fps=30, pipeline=KinectPipeline.CUDA)

    # With depth and IR streams
    KinectCameraConfig(device_index=0, fps=30, use_depth=True, use_ir=True)

    # With colorized depth stream
    KinectCameraConfig(device_index=0, fps=30, use_depth=True, depth_colormap="jet")

    # Custom resolution (Note: Kinect v2 has fixed resolutions)
    # Color: 1920x1080, Depth/IR: 512x424
    KinectCameraConfig(device_index=0, fps=30, width=1920, height=1080)
    ```

    Attributes:
        device_index: Index of the Kinect device (0 for first device). If None, uses first available.
        serial_number: Optional serial number for specific device identification.
        fps: Requested frames per second (Kinect v2 supports up to 30 FPS).
        width: Frame width (1920 for color, 512 for depth/IR). If None, uses default.
        height: Frame height (1080 for color, 424 for depth/IR). If None, uses default.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        use_depth: Whether to enable depth stream. Defaults to False.
        use_ir: Whether to enable infrared stream. Defaults to False.
        pipeline: Processing pipeline to use. AUTO selects best available. Defaults to AUTO.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds). Defaults to 1.
        enable_bilateral_filter: Apply bilateral filter to depth for smoothing. Defaults to True.
        enable_edge_filter: Apply edge-aware filter to depth. Defaults to True.
        min_depth: Minimum depth in meters for depth processing. Defaults to 0.5.
        max_depth: Maximum depth in meters for depth processing. Defaults to 4.5.
        depth_colormap: Colormap for depth visualization. Options: jet, hot, cool, viridis, turbo, rainbow, bone. Defaults to jet.
        depth_min_meters: Minimum depth in meters for colorization. Defaults to 0.5.
        depth_max_meters: Maximum depth in meters for colorization. Defaults to 4.5.
        depth_clipping: Whether to clip depth values outside min/max range. Defaults to True.

    Note:
        - Kinect v2 has fixed resolutions: Color (1920x1080), Depth/IR (512x424)
        - Pipeline selection order: CUDA > OpenCL > OpenGL > CPU
        - Only one Kinect v2 can be connected per USB 3.0 controller
        - Requires USB 3.0 for full framerate operation
        - Depth colorization converts depth data to RGB for visualization
    """

    device_index: int | None = 0
    serial_number: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    use_ir: bool = False
    pipeline: KinectPipeline = KinectPipeline.AUTO
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: float = 1.0
    enable_bilateral_filter: bool = True
    enable_edge_filter: bool = True
    min_depth: float = 0.5
    max_depth: float = 4.5
    # Depth colorization settings
    depth_colormap: str = "jet"
    depth_min_meters: float = 0.5
    depth_max_meters: float = 4.5
    depth_clipping: bool = True

    def __post_init__(self):
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, "
                f"but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            valid_rotations = (
                Cv2Rotation.NO_ROTATION,
                Cv2Rotation.ROTATE_90,
                Cv2Rotation.ROTATE_180,
                Cv2Rotation.ROTATE_270,
            )
            raise ValueError(
                f"`rotation` is expected to be in {valid_rotations}, but {self.rotation} is provided."
            )

        if self.pipeline not in KinectPipeline:
            raise ValueError(
                f"`pipeline` must be one of {list(KinectPipeline)}, but {self.pipeline} is provided."
            )

        # Validate Kinect v2 specific constraints
        if self.width is not None and self.height is not None:
            # Color stream resolution
            if (self.width, self.height) not in [(1920, 1080), (512, 424)]:
                raise ValueError(
                    f"Kinect v2 supports fixed resolutions: Color (1920x1080), Depth/IR (512x424). "
                    f"Got ({self.width}x{self.height})"
                )

        if self.fps is not None and self.fps > 30:
            raise ValueError(f"Kinect v2 maximum FPS is 30. Got {self.fps}")

        if self.min_depth >= self.max_depth:
            raise ValueError(
                f"`min_depth` ({self.min_depth}) must be less than `max_depth` ({self.max_depth})"
            )

        if self.device_index is not None and self.device_index < 0:
            raise ValueError(f"`device_index` must be >= 0. Got {self.device_index}")

        # Validate depth colorization settings
        valid_colormaps = ["jet", "hot", "cool", "viridis", "turbo", "rainbow", "bone"]
        if self.depth_colormap not in valid_colormaps:
            raise ValueError(
                f"`depth_colormap` must be one of {valid_colormaps}, but {self.depth_colormap} is provided."
            )

        if self.depth_min_meters >= self.depth_max_meters:
            raise ValueError(
                f"`depth_min_meters` ({self.depth_min_meters}) must be less than `depth_max_meters` ({self.depth_max_meters})"
            )

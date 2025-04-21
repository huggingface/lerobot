from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    OpenCVCameraConfig(0, 30, 640, 480)
    OpenCVCameraConfig(0, 60, 640, 480)
    OpenCVCameraConfig(0, 90, 640, 480)
    OpenCVCameraConfig(0, 30, 1280, 720)
    ```
    """

    index_or_path: int | Path
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: ColorMode = ColorMode.RGB
    channels: int = 3
    rotation: int | None = None

    def __post_init__(self):
        if self.color_mode not in [ColorMode.RGB, ColorMode.BGR]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        if self.channels != 3:
            raise NotImplementedError(f"Unsupported number of channels: {self.channels}")

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")

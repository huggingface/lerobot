from dataclasses import dataclass
from ..configs import CameraConfig, ColorMode

__all__ = ["OpenMVCameraConfig"]

@CameraConfig.register_subclass("openmv")
@dataclass
class OpenMVCameraConfig(CameraConfig):
    port: str = "/dev/cu.usbmodem3071377B34302"
    color_mode: ColorMode = ColorMode.RGB

    def __post_init__(self):
        self.color_mode = ColorMode(self.color_mode)

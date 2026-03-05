"""Configuration for Luxonis OAK cameras (OAK-D, OAK-D Lite, OAK-D Pro, etc.)."""

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("oak")
@dataclass
class OAKCameraConfig(CameraConfig):
    """Configuration for Luxonis OAK cameras using DepthAI.

    Supports OAK-D, OAK-D Lite, OAK-D Pro, and other DepthAI-compatible devices.
    Identifies cameras by MxID (persistent hardware serial) or device name.

    Example:
        ```python
        OAKCameraConfig("14442C1091E0D1D700", fps=30, width=640, height=400)
        OAKCameraConfig("14442C1091E0D1D700", fps=30, width=640, height=400, use_depth=True)
        ```

    Attributes:
        mxid_or_ip: MxID serial string, IP address, or empty string for auto-detect.
        color_mode: RGB or BGR output. Defaults to RGB.
        use_depth: Enable aligned stereo depth stream. Defaults to False.
        rotation: Image rotation. Defaults to no rotation.
        warmup_s: Seconds to warm up before returning from connect().
    """

    mxid_or_ip: str = ""
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 2

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if self.fps is None:
            self.fps = 30
        if self.width is None:
            self.width = 640
        if self.height is None:
            self.height = 400

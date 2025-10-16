from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(CameraConfig):
    """Configuration class for ZED cameras.

    This class provides specialized configuration options for ZED cameras,
    including support for depth sensing and device identification via serial number or name.

    Example configurations for ZED 2i:
    ```python
    # Basic configurations
    ZedCameraConfig("0123456789", 30, 1280, 720)  # 1280x720 @ 30FPS
    ZedCameraConfig("0123456789", 15, 2208, 1242)  # 2208x1242 @ 15FPS

    # Advanced configurations
    ZedCameraConfig("0123456789", 30, 1280, 720, use_depth=True)  # With depth sensing
    ZedCameraConfig("0123456789", 30, 1280, 720, rotation=Cv2Rotation.ROTATE_90)  # With 90° rotation
    ```

    Attributes:
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        serial_number_or_name: Unique serial number or human-readable name to identify the camera.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        use_depth: Whether to enable depth stream. Defaults to False.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        depth_mode: Depth sensing mode for ZED camera. Options: 'QUALITY', 'ULTRA', 'NEURAL'

    Note:
        - Either name or serial_number must be specified.
        - Depth stream configuration (if enabled) will use the same FPS as the color stream.
        - The actual resolution and FPS may be adjusted by the camera to the nearest supported mode.
        - For `fps`, `width` and `height`, either all of them need to be set, or none of them.
    """

    serial_number_or_name: str = "" # Default to the unique ZED camera
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.ROTATE_180
    warmup_s: int = 3  # ZED cameras need longer warmup time
    depth_mode: str = "QUALITY"

    def __post_init__(self):
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )

        if self.depth_mode not in ("QUALITY", "ULTRA", "NEURAL"):
            raise ValueError(
                f"`depth_mode` is expected to be 'QUALITY', 'ULTRA', or 'NEURAL', but {self.depth_mode} is provided."
            )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )

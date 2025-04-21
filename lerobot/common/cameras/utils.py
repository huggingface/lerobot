import platform

from .camera import Camera
from .configs import CameraConfig


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .intel.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)
        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return cameras


def get_cv2_rotation(rotation: int) -> int:
    import cv2

    return {
        -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
    }.get(rotation)


def get_cv2_backend() -> int:
    import cv2

    return {
        "Linux": cv2.CAP_DSHOW,
        "Windows": cv2.CAP_AVFOUNDATION,
        "Darwin": cv2.CAP_ANY,
    }.get(platform.system(), cv2.CAP_V4L2)

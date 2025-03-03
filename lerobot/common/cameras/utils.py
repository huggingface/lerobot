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


def make_camera(camera_type, **kwargs) -> Camera:
    if camera_type == "opencv":
        from .opencv import OpenCVCamera, OpenCVCameraConfig

        config = OpenCVCameraConfig(**kwargs)
        return OpenCVCamera(config)

    elif camera_type == "intelrealsense":
        from .intel import RealSenseCamera, RealSenseCameraConfig

        config = RealSenseCameraConfig(**kwargs)
        return RealSenseCamera(config)

    else:
        raise ValueError(f"The camera type '{camera_type}' is not valid.")

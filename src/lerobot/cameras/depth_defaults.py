from dataclasses import dataclass
from typing import Optional

# Canonical depth scales (meters per unit) for sensors we support
# Kinect v2 (libfreenect2) depth is float32 in millimeters
KINECT_DEPTH_METERS_PER_UNIT: float = 1e-3
# RealSense D405 default (most D4xx expose this via device)
REALSENSE_DEFAULT_METERS_PER_UNIT: float = 1e-4


@dataclass
class DepthParams:
    min_m: float
    max_m: float
    colormap: str
    meters_per_unit: Optional[float] = None


SENSOR_DEFAULTS = {
    "kinect": DepthParams(min_m=0.5, max_m=4.5, colormap="TURBO", meters_per_unit=KINECT_DEPTH_METERS_PER_UNIT),
    # Assume D405-like unit scale by default to ensure scale is always provided
    "realsense": DepthParams(min_m=0.3, max_m=4.0, colormap="TURBO", meters_per_unit=REALSENSE_DEFAULT_METERS_PER_UNIT),
    "realsense_d405": DepthParams(min_m=0.07, max_m=0.5, colormap="TURBO", meters_per_unit=REALSENSE_DEFAULT_METERS_PER_UNIT),
}


def resolve_depth_params(
    *,
    sensor: str,
    cam_cfg: object | None,
    cli_colormap: str | None,
    cli_min: float | None,
    cli_max: float | None,
    device_scale: float | None,
) -> DepthParams:
    # Choose sensor defaults
    defaults = SENSOR_DEFAULTS.get(sensor, SENSOR_DEFAULTS["kinect"])  # fallback to Kinect-like ranges

    # Camera-config values if present
    cfg_min = getattr(cam_cfg, "depth_min_meters", None) if cam_cfg else None
    cfg_max = getattr(cam_cfg, "depth_max_meters", None) if cam_cfg else None
    cfg_cmap = getattr(cam_cfg, "depth_colormap", None) if cam_cfg else None

    min_m = cli_min if cli_min is not None else (cfg_min if cfg_min is not None else defaults.min_m)
    max_m = cli_max if cli_max is not None else (cfg_max if cfg_max is not None else defaults.max_m)
    cmap = cli_colormap if cli_colormap is not None else (cfg_cmap if cfg_cmap is not None else defaults.colormap)

    # meters_per_unit: CLI > device_scale > defaults
    meters_per_unit = device_scale if device_scale is not None else defaults.meters_per_unit

    return DepthParams(min_m=float(min_m), max_m=float(max_m), colormap=str(cmap), meters_per_unit=(float(meters_per_unit) if meters_per_unit is not None else None))



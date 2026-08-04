from __future__ import annotations

from copy import deepcopy

import numpy as np

ALIGNED_FRONT_CAMERA_PROFILE_ID = "icspring_front_crop_1280x960_to_640x480_v1"
SYNTHETIC_FRONT_CAMERA_PROFILE_ID = "synthetic_front_640x480_v1"

_CAMERA_PROFILES = {
    ALIGNED_FRONT_CAMERA_PROFILE_ID: {
        "profile_id": ALIGNED_FRONT_CAMERA_PROFILE_ID,
        "source": {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "fourcc": "MJPG",
            "color": "RGB",
        },
        "crop": {
            "x": 320,
            "y": 60,
            "width": 1280,
            "height": 960,
        },
        "output": {
            "width": 640,
            "height": 480,
            "color": "RGB",
            "resize": "opencv_inter_area",
        },
        "lens_processing": "raw_distortion_preserved",
        "alignment_reference": {
            "mujoco_vertical_fov_degrees": 47.0,
            "physical_crop_fov_estimate_degrees": {
                "horizontal": 63.0,
                "vertical": 49.0,
                "status": "approximate_grid_measurement_not_calibration",
            },
        },
    },
    SYNTHETIC_FRONT_CAMERA_PROFILE_ID: {
        "profile_id": SYNTHETIC_FRONT_CAMERA_PROFILE_ID,
        "source": {
            "width": 640,
            "height": 480,
            "fps": 30,
            "fourcc": None,
            "color": "RGB",
        },
        "crop": {
            "x": 0,
            "y": 0,
            "width": 640,
            "height": 480,
        },
        "output": {
            "width": 640,
            "height": 480,
            "color": "RGB",
            "resize": "none",
        },
        "lens_processing": "not_applicable_synthetic",
        "alignment_reference": None,
    },
}


def camera_profile(profile_id: str) -> dict:
    try:
        return deepcopy(_CAMERA_PROFILES[profile_id])
    except KeyError as exc:
        raise ValueError(f"unsupported camera_profile_id: {profile_id}") from exc


def validate_camera_profile_config(cfg: dict) -> dict:
    profile = camera_profile(str(cfg.get("camera_profile_id", "")))
    if int(cfg.get("camera_acquisition_fps", 0)) != profile["source"]["fps"]:
        raise ValueError(
            f"camera_acquisition_fps must match immutable camera profile ({profile['source']['fps']})"
        )
    if cfg.get("mode") == "real" and profile["profile_id"] != ALIGNED_FRONT_CAMERA_PROFILE_ID:
        raise ValueError(f"real mode requires aligned camera profile {ALIGNED_FRONT_CAMERA_PROFILE_ID}")
    if cfg.get("mode") == "synthetic" and profile["profile_id"] != SYNTHETIC_FRONT_CAMERA_PROFILE_ID:
        raise ValueError(f"synthetic mode requires camera profile {SYNTHETIC_FRONT_CAMERA_PROFILE_ID}")
    return profile


def canonicalize_front(frame_rgb: np.ndarray, profile_id: str) -> np.ndarray:
    profile = camera_profile(profile_id)
    source = profile["source"]
    expected = (source["height"], source["width"], 3)
    frame_rgb = np.asarray(frame_rgb)
    if frame_rgb.shape != expected or frame_rgb.dtype != np.uint8:
        raise RuntimeError(
            f"front source violated camera profile: expected uint8 {expected}, "
            f"got {frame_rgb.dtype} {frame_rgb.shape}"
        )

    crop = profile["crop"]
    y0, x0 = crop["y"], crop["x"]
    cropped = frame_rgb[y0 : y0 + crop["height"], x0 : x0 + crop["width"]]
    output = profile["output"]
    if output["resize"] == "none":
        canonical = cropped
    elif output["resize"] == "opencv_inter_area":
        import cv2

        canonical = cv2.resize(
            cropped,
            (output["width"], output["height"]),
            interpolation=cv2.INTER_AREA,
        )
    else:
        raise RuntimeError(f"unsupported camera profile resize: {output['resize']}")

    canonical = np.ascontiguousarray(canonical, dtype=np.uint8)
    expected_output = (output["height"], output["width"], 3)
    if canonical.shape != expected_output:
        raise RuntimeError(
            f"front output violated camera profile: expected {expected_output}, got {canonical.shape}"
        )
    return canonical

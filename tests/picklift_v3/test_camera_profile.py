import numpy as np
import pytest

from examples.picklift_v3.camera_profile import (
    ALIGNED_FRONT_CAMERA_PROFILE_ID,
    SYNTHETIC_FRONT_CAMERA_PROFILE_ID,
    camera_profile,
    canonicalize_front,
    validate_camera_profile_config,
)


def test_aligned_profile_freezes_source_crop_output_and_reference_fov():
    profile = camera_profile(ALIGNED_FRONT_CAMERA_PROFILE_ID)
    assert profile["source"] == {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "fourcc": "MJPG",
        "color": "RGB",
    }
    assert profile["crop"] == {"x": 320, "y": 60, "width": 1280, "height": 960}
    assert profile["output"] == {
        "width": 640,
        "height": 480,
        "color": "RGB",
        "resize": "opencv_inter_area",
    }
    assert profile["alignment_reference"]["mujoco_vertical_fov_degrees"] == 47.0


def test_aligned_profile_applies_exact_center_crop_then_half_scale():
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)
    source[60:1020, 320:1600] = (40, 80, 120)
    source[:60] = 255
    source[:, :320] = 255
    output = canonicalize_front(source, ALIGNED_FRONT_CAMERA_PROFILE_ID)
    assert output.shape == (480, 640, 3)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output, np.full_like(output, (40, 80, 120)))


def test_camera_profile_rejects_wrong_source_shape():
    with pytest.raises(RuntimeError, match="front source violated"):
        canonicalize_front(
            np.zeros((480, 640, 3), dtype=np.uint8),
            ALIGNED_FRONT_CAMERA_PROFILE_ID,
        )


def test_profile_mode_and_fps_are_fail_closed():
    with pytest.raises(ValueError, match="immutable camera profile"):
        validate_camera_profile_config(
            {
                "mode": "synthetic",
                "camera_profile_id": SYNTHETIC_FRONT_CAMERA_PROFILE_ID,
                "camera_acquisition_fps": 20,
            }
        )
    with pytest.raises(ValueError, match="real mode requires aligned"):
        validate_camera_profile_config(
            {
                "mode": "real",
                "camera_profile_id": SYNTHETIC_FRONT_CAMERA_PROFILE_ID,
                "camera_acquisition_fps": 30,
            }
        )

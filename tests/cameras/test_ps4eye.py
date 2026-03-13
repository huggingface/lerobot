#!/usr/bin/env python

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

"""Unit tests for PS4EyeCamera.

All tests run without a physical PS4 Eye; cv2.VideoCapture is patched with a
MockPS4EyeCapture that returns synthetic 3448×808 panoramic BGR frames.

Example:
    pytest tests/cameras/test_ps4eye.py -v
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.ps4eye import PS4EyeCamera, PS4EyeCameraConfig
from lerobot.cameras.ps4eye.camera_ps4eye import _SHARED_CAPTURES, _STEREO_CROPS
from lerobot.cameras.ps4eye.configuration_ps4eye import ColorMode, Cv2Rotation, EyeSelection
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# ---------------------------------------------------------------------------
# Panoramic frame dimensions used in the mock
# ---------------------------------------------------------------------------
MOCK_WIDTH, MOCK_HEIGHT = 3448, 808  # full-res PS4 Eye panoramic size
_CROP = _STEREO_CROPS[(MOCK_WIDTH, MOCK_HEIGHT)]  # (r0, r1, cl0, cl1, cr0, cr1)

EYE_HEIGHT = _CROP[1] - _CROP[0]    # 800
EYE_WIDTH  = _CROP[3] - _CROP[2]    # 1264


# ---------------------------------------------------------------------------
# Mock VideoCapture
# ---------------------------------------------------------------------------

class MockPS4EyeCapture:
    """Mimics cv2.VideoCapture returning a fixed synthetic panoramic frame."""

    _instance_count = 0  # class-level counter so tests can inspect open calls

    def __init__(self, source=0, *args, **kwargs):
        MockPS4EyeCapture._instance_count += 1
        self._source = source
        # Return True for any source that looks valid (not obviously broken)
        self._opened = source != "nonexistent_device_xyz"
        # Synthetic BGR panoramic frame
        self._frame = np.zeros((MOCK_HEIGHT, MOCK_WIDTH, 3), dtype=np.uint8)
        self._frame[:, :MOCK_WIDTH // 2] = (64, 128, 192)   # left half: blueish
        self._frame[:, MOCK_WIDTH // 2:] = (192, 128, 64)   # right half: orangeish

        self._props = {
            "CAP_PROP_FRAME_WIDTH": float(MOCK_WIDTH),
            "CAP_PROP_FRAME_HEIGHT": float(MOCK_HEIGHT),
            "CAP_PROP_FPS": 60.0,
        }

    # cv2 constants are integers; MockPS4EyeCapture.get() accepts either
    def get(self, prop_id):
        import cv2  # noqa: PLC0415
        prop_map = {
            cv2.CAP_PROP_FRAME_WIDTH:  self._props["CAP_PROP_FRAME_WIDTH"],
            cv2.CAP_PROP_FRAME_HEIGHT: self._props["CAP_PROP_FRAME_HEIGHT"],
            cv2.CAP_PROP_FPS:          self._props["CAP_PROP_FPS"],
        }
        return prop_map.get(prop_id, 0.0)

    def set(self, prop_id, value):
        import cv2  # noqa: PLC0415
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self._props["CAP_PROP_FRAME_WIDTH"] = value
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self._props["CAP_PROP_FRAME_HEIGHT"] = value
        elif prop_id == cv2.CAP_PROP_FPS:
            self._props["CAP_PROP_FPS"] = value
        return True

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        return True, self._frame.copy()

    def release(self):
        self._opened = False

    def getBackendName(self):
        return "MOCK"


# ---------------------------------------------------------------------------
# Autouse fixture: patch cv2.VideoCapture and clean the shared registry
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_ps4eye_videocapture():
    """Patches cv2.VideoCapture inside camera_ps4eye and resets the registry."""
    module_path = PS4EyeCamera.__module__
    target = f"{module_path}.cv2.VideoCapture"

    MockPS4EyeCapture._instance_count = 0
    _SHARED_CAPTURES.clear()  # ensure a clean slate between tests

    with patch(target, new=MockPS4EyeCapture):
        yield

    # Always clean up the shared registry so tests don't bleed into each other
    _SHARED_CAPTURES.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    index=0,
    eye: str = "left",
    width: int = MOCK_WIDTH,
    height: int = MOCK_HEIGHT,
    fps: int = 60,
    warmup_s: int = 0,
    color_mode: ColorMode = ColorMode.RGB,
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION,
) -> PS4EyeCameraConfig:
    return PS4EyeCameraConfig(
        index_or_path=index,
        eye=eye,
        width=width,
        height=height,
        fps=fps,
        warmup_s=warmup_s,
        color_mode=color_mode,
        rotation=rotation,
    )


# ---------------------------------------------------------------------------
# Configuration validation tests (no hardware needed at all)
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_valid_eye_values(self):
        for eye in ("left", "right", "both"):
            cfg = _make_config(eye=eye)
            assert cfg.eye == EyeSelection(eye)

    def test_wrong_eye_value(self):
        with pytest.raises(ValueError, match="eye"):
            PS4EyeCameraConfig(index_or_path=0, eye="center")  # type: ignore[arg-type]

    def test_wrong_color_mode(self):
        with pytest.raises(ValueError, match="color_mode"):
            PS4EyeCameraConfig(index_or_path=0, color_mode="hsv")  # type: ignore[arg-type]

    def test_wrong_rotation(self):
        with pytest.raises(ValueError, match="rotation"):
            PS4EyeCameraConfig(index_or_path=0, rotation=45)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_abc_implementation():
    """PS4EyeCamera should be instantiable without error."""
    config = _make_config()
    cam = PS4EyeCamera(config)
    assert cam is not None


# ---------------------------------------------------------------------------
# Connect / disconnect lifecycle
# ---------------------------------------------------------------------------

class TestConnectDisconnect:
    def test_connect(self):
        cam = PS4EyeCamera(_make_config(warmup_s=0))
        cam.connect(warmup=False)
        assert cam.is_connected
        cam.disconnect()

    def test_connect_already_connected(self):
        cam = PS4EyeCamera(_make_config(warmup_s=0))
        cam.connect(warmup=False)
        with pytest.raises(DeviceAlreadyConnectedError):
            cam.connect(warmup=False)
        cam.disconnect()

    def test_connect_invalid_camera_path(self):
        cfg = PS4EyeCameraConfig(index_or_path="nonexistent_device_xyz", eye="left")
        cam = PS4EyeCamera(cfg)
        with pytest.raises(ConnectionError):
            cam.connect(warmup=False)

    def test_context_manager(self):
        """PS4EyeCamera should support the `with` statement.

        __enter__ already calls connect(), so we must NOT call it again inside the block.
        """
        cam = PS4EyeCamera(_make_config(warmup_s=0))
        with cam:
            assert cam.is_connected
        assert not cam.is_connected

    def test_disconnect(self):
        cam = PS4EyeCamera(_make_config(warmup_s=0))
        cam.connect(warmup=False)
        cam.disconnect()
        assert not cam.is_connected

    def test_disconnect_before_connect(self):
        cam = PS4EyeCamera(_make_config(warmup_s=0))
        with pytest.raises(DeviceNotConnectedError):
            cam.disconnect()


# ---------------------------------------------------------------------------
# Shared capture registry
# ---------------------------------------------------------------------------

class TestSharedCaptureRegistry:
    def test_single_instance_creates_entry(self):
        cam = PS4EyeCamera(_make_config(index=5, warmup_s=0))
        cam.connect(warmup=False)

        assert "5" in _SHARED_CAPTURES
        assert _SHARED_CAPTURES["5"].ref_count == 1

        cam.disconnect()
        assert "5" not in _SHARED_CAPTURES

    def test_two_instances_share_one_capture(self):
        """Two eyes on the same device should share one VideoCapture."""
        MockPS4EyeCapture._instance_count = 0

        left  = PS4EyeCamera(_make_config(index=7, eye="left",  warmup_s=0))
        right = PS4EyeCamera(_make_config(index=7, eye="right", warmup_s=0))

        left.connect(warmup=False)
        right.connect(warmup=False)

        # Only one VideoCapture should have been constructed
        assert MockPS4EyeCapture._instance_count == 1
        assert _SHARED_CAPTURES["7"].ref_count == 2

        left.disconnect()
        assert _SHARED_CAPTURES["7"].ref_count == 1

        right.disconnect()
        assert "7" not in _SHARED_CAPTURES

    def test_resolution_conflict_raises(self):
        """Second instance requesting a different resolution must raise RuntimeError."""
        left = PS4EyeCamera(_make_config(index=9, eye="left",  width=3448, height=808, warmup_s=0))
        left.connect(warmup=False)

        right = PS4EyeCamera(
            PS4EyeCameraConfig(index_or_path=9, eye="right", width=1748, height=408, fps=30)
        )
        with pytest.raises(RuntimeError, match="resolution"):
            right.connect(warmup=False)

        left.disconnect()


# ---------------------------------------------------------------------------
# Frame reading — eye slicing
# ---------------------------------------------------------------------------

class TestRead:
    def test_read_left_eye_shape(self):
        cam = PS4EyeCamera(_make_config(eye="left", warmup_s=0))
        cam.connect(warmup=False)
        frame = cam.read()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (EYE_HEIGHT, EYE_WIDTH, 3), frame.shape
        cam.disconnect()

    def test_read_right_eye_shape(self):
        cam = PS4EyeCamera(_make_config(eye="right", warmup_s=0))
        cam.connect(warmup=False)
        frame = cam.read()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (EYE_HEIGHT, EYE_WIDTH, 3), frame.shape
        cam.disconnect()

    def test_read_both_eyes_shape(self):
        """eye='both' returns the full panoramic frame (no crop)."""
        cam = PS4EyeCamera(_make_config(eye="both", warmup_s=0))
        cam.connect(warmup=False)
        frame = cam.read()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (MOCK_HEIGHT, MOCK_WIDTH, 3), frame.shape
        cam.disconnect()

    def test_read_before_connect(self):
        cam = PS4EyeCamera(_make_config(warmup_s=0))
        with pytest.raises(DeviceNotConnectedError):
            cam.read()

    def test_read_color_mode_bgr(self):
        cam = PS4EyeCamera(_make_config(eye="left", color_mode=ColorMode.BGR, warmup_s=0))
        cam.connect(warmup=False)
        frame = cam.read()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        cam.disconnect()

    def test_read_color_mode_rgb(self):
        cam = PS4EyeCamera(_make_config(eye="left", color_mode=ColorMode.RGB, warmup_s=0))
        cam.connect(warmup=False)
        frame = cam.read()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        cam.disconnect()


# ---------------------------------------------------------------------------
# Stereo crop geometry
# ---------------------------------------------------------------------------

class TestStereoCropGeometry:
    """Verify that left and right crops are non-overlapping and correct size."""

    def test_left_crop_shape(self):
        cam = PS4EyeCamera(_make_config(eye="left", warmup_s=0))
        cam.connect(warmup=False)
        frame = cam.read()
        row_start, row_end, cl0, cl1, _cr0, _cr1 = _CROP
        expected_h = row_end - row_start
        expected_w = cl1 - cl0
        assert frame.shape[:2] == (expected_h, expected_w)
        cam.disconnect()

    def test_right_crop_shape(self):
        cam = PS4EyeCamera(_make_config(eye="right", warmup_s=0))
        cam.connect(warmup=False)
        frame = cam.read()
        row_start, row_end, _cl0, _cl1, cr0, cr1 = _CROP
        expected_h = row_end - row_start
        expected_w = cr1 - cr0
        assert frame.shape[:2] == (expected_h, expected_w)
        cam.disconnect()

    def test_left_right_crops_are_independent(self):
        """Left and right frames should differ because the mock colors the halves differently."""
        left  = PS4EyeCamera(_make_config(index=11, eye="left",  warmup_s=0))
        right = PS4EyeCamera(_make_config(index=11, eye="right", warmup_s=0))
        left.connect(warmup=False)
        right.connect(warmup=False)

        left_frame  = left.read()
        right_frame = right.read()

        # The mean pixel values must differ between left and right crops
        assert not np.allclose(left_frame.mean(axis=(0, 1)), right_frame.mean(axis=(0, 1)))

        left.disconnect()
        right.disconnect()


# ---------------------------------------------------------------------------
# async_read
# ---------------------------------------------------------------------------

class TestAsyncRead:
    def test_async_read_returns_frame(self):
        cam = PS4EyeCamera(_make_config(eye="left", warmup_s=0))
        cam.connect(warmup=False)
        frame = cam.async_read(timeout_ms=2000)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (EYE_HEIGHT, EYE_WIDTH, 3)
        cam.disconnect()

    def test_async_read_before_connect(self):
        cam = PS4EyeCamera(_make_config(warmup_s=0))
        with pytest.raises(DeviceNotConnectedError):
            cam.async_read()


# ---------------------------------------------------------------------------
# read_latest
# ---------------------------------------------------------------------------

class TestReadLatest:
    def test_read_latest_returns_frame(self):
        cam = PS4EyeCamera(_make_config(eye="left", warmup_s=0))
        cam.connect(warmup=False)

        # Prime the buffer
        cam.read()
        # Give the background thread a moment to populate the per-instance buffer
        time.sleep(0.1)

        latest = cam.read_latest(max_age_ms=5000)
        assert isinstance(latest, np.ndarray)
        assert latest.shape == (EYE_HEIGHT, EYE_WIDTH, 3)
        cam.disconnect()

    def test_read_latest_before_connect(self):
        cam = PS4EyeCamera(_make_config(warmup_s=0))
        with pytest.raises(DeviceNotConnectedError):
            cam.read_latest()

    def test_read_latest_too_old(self):
        cam = PS4EyeCamera(_make_config(eye="left", warmup_s=0))
        cam.connect(warmup=False)

        # Prime so there IS a frame
        cam.read()
        time.sleep(0.05)

        with pytest.raises(TimeoutError):
            cam.read_latest(max_age_ms=0)  # immediately stale

        cam.disconnect()

    def test_read_latest_high_frequency(self):
        cam = PS4EyeCamera(_make_config(eye="left", warmup_s=0))
        cam.connect(warmup=False)
        ref = cam.read()
        time.sleep(0.1)

        for _ in range(10):
            latest = cam.read_latest(max_age_ms=5000)
            assert isinstance(latest, np.ndarray)
            assert latest.shape == ref.shape

        cam.disconnect()

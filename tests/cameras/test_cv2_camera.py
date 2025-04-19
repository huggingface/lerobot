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
import os
import platform
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.common.cameras.interface_camera_sdk import FakeOpenCVSDKAdapter, IVideoCapture, OpenCVSDKAdapter
from lerobot.common.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

MODULE_PATH = "lerobot.common.cameras.opencv.camera_opencv"

# Define constants that might be used by mocks
MOCK_CV2_CAP_PROP_FPS = FakeOpenCVSDKAdapter.CAP_PROP_FPS
MOCK_CV2_CAP_PROP_FRAME_WIDTH = FakeOpenCVSDKAdapter.CAP_PROP_FRAME_WIDTH
MOCK_CV2_CAP_PROP_FRAME_HEIGHT = FakeOpenCVSDKAdapter.CAP_PROP_FRAME_HEIGHT

# TODO(Steven): Consider a CLI argument to set this
# Emulates the cheap USB camera in index 0
LEROBOT_USE_REAL_OPENCV_CAMERA_TESTS = (
    os.getenv("LEROBOT_USE_REAL_OPENCV_CAMERA_TESTS", "False").lower() == "true"
)


# Helper function to create a realistic-looking dummy image
def create_dummy_image(height, width, channels=3):
    return np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)


# --- Test Class ---


class TestOpenCVCamera(unittest.TestCase):
    def setUp(self):
        # Emulates the cheap USB camera in index 0
        self.default_config = OpenCVCameraConfig(
            camera_index=0,
            width=MOCK_CV2_CAP_PROP_FRAME_WIDTH,
            height=MOCK_CV2_CAP_PROP_FRAME_HEIGHT,
            fps=MOCK_CV2_CAP_PROP_FPS,
            color_mode="rgb",
            rotation=None,
        )
        # TODO(Steven): Consider calling this only once, instead of in every test
        if LEROBOT_USE_REAL_OPENCV_CAMERA_TESTS:
            # Use the real OpenCV SDK
            self.test_sdk = OpenCVSDKAdapter()
            # self.addCleanup()
        else:
            # Create a default dummy image based on config
            self.dummy_bgr_image = create_dummy_image(self.default_config.height, self.default_config.width)
            self.dummy_rgb_image = self.dummy_bgr_image[..., ::-1]  # Simple BGR -> RGB simulation
            FakeOpenCVSDKAdapter.init_configure_fake(simulated_image=self.dummy_bgr_image)
            self.test_sdk = FakeOpenCVSDKAdapter()
            # self.addCleanup()

        patcher_platform = patch(f"{MODULE_PATH}.platform.system")
        self.mock_platform_system = patcher_platform.start()
        self.addCleanup(patcher_platform.stop)
        self.mock_platform_system.return_value = "Darwin"  # Default to macOS

        patcher_is_valid_unix_path = patch(f"{MODULE_PATH}.is_valid_unix_path")
        self.mock_is_valid_unix_path = patcher_is_valid_unix_path.start()
        self.addCleanup(patcher_is_valid_unix_path.stop)
        self.mock_is_valid_unix_path.return_value = True

        patcher_time_sleep = patch(f"{MODULE_PATH}.time.sleep", return_value=None)
        self.mock_sleep = patcher_time_sleep.start()
        self.addCleanup(patcher_time_sleep.stop)

        # Mock threading
        patcher_thread = patch(f"{MODULE_PATH}.threading.Thread", autospec=True)
        self.mock_thread_class = patcher_thread.start()
        self.addCleanup(patcher_thread.stop)
        self.mock_thread_instance = MagicMock()
        self.mock_thread_class.return_value = self.mock_thread_instance

        patcher_event = patch(f"{MODULE_PATH}.threading.Event", autospec=True)
        self.mock_event_class = patcher_event.start()
        self.addCleanup(patcher_event.stop)
        self.mock_event_instance = MagicMock()
        self.mock_event_class.return_value = self.mock_event_instance
        self.mock_event_instance.is_set.return_value = False  # Default to not set

    # --- Tests __init__ ---
    def test_init_defaults(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)

        self.assertEqual(cam.camera_index, 0)
        self.assertEqual(cam.capture_width, MOCK_CV2_CAP_PROP_FRAME_WIDTH)
        self.assertEqual(cam.capture_height, MOCK_CV2_CAP_PROP_FRAME_HEIGHT)
        self.assertEqual(cam.width, MOCK_CV2_CAP_PROP_FRAME_WIDTH)  # No rotation
        self.assertEqual(cam.height, MOCK_CV2_CAP_PROP_FRAME_HEIGHT)  # No rotation
        self.assertEqual(cam.fps, MOCK_CV2_CAP_PROP_FPS)
        self.assertEqual(cam.color_mode, "rgb")
        self.assertIsNone(cam.rotation)  # Rotation 0
        self.assertIsNone(cam.port)
        self.assertFalse(cam.is_connected)

    def test_init_with_rotation_90(self):
        config = self.default_config
        config.rotation = 90
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)
        self.assertEqual(cam.width, MOCK_CV2_CAP_PROP_FRAME_HEIGHT)  # Swapped
        self.assertEqual(cam.height, MOCK_CV2_CAP_PROP_FRAME_WIDTH)  # Swapped
        self.assertEqual(cam.rotation, self.test_sdk.ROTATE_90_CLOCKWISE)

    def test_init_with_rotation_minus_90(self):
        config = self.default_config
        config.rotation = -90
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)
        self.assertEqual(cam.width, MOCK_CV2_CAP_PROP_FRAME_HEIGHT)  # Swapped
        self.assertEqual(cam.height, MOCK_CV2_CAP_PROP_FRAME_WIDTH)  # Swapped
        self.assertEqual(cam.rotation, self.test_sdk.ROTATE_90_COUNTERCLOCKWISE)

    def test_init_with_rotation_180(self):
        config = self.default_config
        config.rotation = 180
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)
        self.assertEqual(cam.width, MOCK_CV2_CAP_PROP_FRAME_WIDTH)  # Swapped
        self.assertEqual(cam.height, MOCK_CV2_CAP_PROP_FRAME_HEIGHT)  # Swapped
        self.assertEqual(cam.rotation, self.test_sdk.ROTATE_180)

    @pytest.mark.skipif(
        LEROBOT_USE_REAL_OPENCV_CAMERA_TESTS and platform.system() != "Linux",
        reason="Not valid for real camera for other platform than Linux",
    )
    def test_init_linux_with_index(self):
        self.mock_platform_system.return_value = "Linux"
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        self.assertEqual(cam.camera_index, 0)
        self.assertIsInstance(cam.port, Path)
        self.assertEqual(str(cam.port), "/dev/video0")

    @pytest.mark.skipif(
        LEROBOT_USE_REAL_OPENCV_CAMERA_TESTS and platform.system() != "Linux",
        reason="Not valid for real camera for other platform than Linux",
    )
    def test_init_linux_with_valid_path(self):
        self.mock_platform_system.return_value = "Linux"
        config = self.default_config
        config.camera_index = "/dev/video2"
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)

        self.assertIsInstance(cam.port, Path)
        self.assertEqual(str(cam.port), "/dev/video2")
        self.assertEqual(cam.camera_index, 2)

    @pytest.mark.skipif(
        LEROBOT_USE_REAL_OPENCV_CAMERA_TESTS and platform.system() != "Linux",
        reason="Not valid for real camera for other platform than Linux",
    )
    def test_init_linux_with_invalid_path(self):
        self.mock_platform_system.return_value = "Linux"
        self.mock_is_valid_unix_path.return_value = False
        config = self.default_config
        config.camera_index = "[*?:[/invalid/path"
        with self.assertRaisesRegex(ValueError, "Please check the provided camera_index"):
            OpenCVCamera(config, cv2_sdk=self.test_sdk)

    # --- Test connect ---
    def test_connect_success(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        cam.connect()

        self.assertTrue(cam.is_connected)
        self.assertIsNotNone(cam.camera)
        self.assertIsInstance(cam.camera, IVideoCapture)
        self.assertEqual(cam.fps, self.default_config.fps)
        self.assertEqual(cam.capture_width, self.default_config.width)
        self.assertEqual(cam.capture_height, self.default_config.height)

    @pytest.mark.skipif(
        LEROBOT_USE_REAL_OPENCV_CAMERA_TESTS and platform.system() != "Linux",
        reason="Not valid for real camera for other platform than Linux",
    )
    def test_connect_success_linux(self):
        self.mock_platform_system.return_value = "Linux"
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)

        cam.connect()

        self.assertTrue(cam.is_connected)
        self.assertIsNotNone(cam.camera)
        self.assertIsInstance(cam.camera, IVideoCapture)
        # self.assertIn('/dev/video0', self.test_sdk._cameras_opened)
        # self.assertEqual(self.test_sdk.FakeVideoCapture.backend, self.test_sdk.CAP_V4L2)

    def test_connect_already_connected(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        # Simulate already connected state
        cam.is_connected = True
        with self.assertRaisesRegex(DeviceAlreadyConnectedError, "already connected"):
            cam.connect()
        cam.is_connected = False  # To avoid warning when running the destructor

    def test_connect_camera_not_opened_invalid_index(self):
        config = self.default_config
        config.camera_index = 99  # Invalid index in fakeSDK != (0,1,2)
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)

        with self.assertRaisesRegex(
            ValueError, "expected to be one of these available cameras \\[0, 1, 2\\], but 99 is provided"
        ):
            cam.connect()
        self.assertFalse(cam.is_connected)

    def test_connect_fps_mismatch(self):
        config = self.default_config
        config.fps = 999  # Different FPS
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)

        with self.assertRaisesRegex(
            OSError, f"Can't set self.fps=999 .* Actual value is {MOCK_CV2_CAP_PROP_FPS}"
        ):
            cam.connect()
        self.assertFalse(cam.is_connected)

    def test_connect_width_mismatch(self):
        config = self.default_config
        config.width = 9999
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)

        with self.assertRaisesRegex(
            OSError, f"Can't set self.capture_width=9999.* Actual value is {MOCK_CV2_CAP_PROP_FRAME_WIDTH}"
        ):
            cam.connect()
        self.assertFalse(cam.is_connected)

    def test_connect_height_mismatch(self):
        config = self.default_config
        config.height = 9999
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)

        with self.assertRaisesRegex(
            OSError, f"Can't set self.capture_height=9999.* Actual value is {MOCK_CV2_CAP_PROP_FRAME_HEIGHT}"
        ):
            cam.connect()
        self.assertFalse(cam.is_connected)

    # --- Test read ---
    def test_read_not_connected(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        with self.assertRaisesRegex(DeviceNotConnectedError, "not connected"):
            cam.read()

    def test_read_success_rgb(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        cam.connect()

        img = cam.read()
        if isinstance(self.test_sdk, FakeOpenCVSDKAdapter):
            # When using fake SDK, verify exact match with dummy image
            np.testing.assert_array_equal(img, self.dummy_rgb_image)
        else:
            # When using real camera, verify basic properties of the captured image
            self.assertIsInstance(img, np.ndarray)

    def test_read_success_bgr(self):
        config = self.default_config
        config.color_mode = "bgr"
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)
        cam.connect()

        img = cam.read()
        if isinstance(self.test_sdk, FakeOpenCVSDKAdapter):
            # When using fake SDK, verify exact match with dummy image
            np.testing.assert_array_equal(img, self.dummy_bgr_image)
        else:
            # When using real camera, verify basic properties of the captured image
            self.assertIsInstance(img, np.ndarray)

    def test_read_success_with_rotation(self):
        config = self.default_config
        config.rotation = 90
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)  # Sets cam.rotation internally
        cam.connect()

        img = cam.read()
        if isinstance(self.test_sdk, FakeOpenCVSDKAdapter):
            # When using fake SDK, verify exact match with dummy image
            rotated_image = np.rot90(self.dummy_rgb_image)  # Simulate rotation
            np.testing.assert_array_equal(img, rotated_image)
        else:
            # When using real camera, verify basic properties of the captured image
            self.assertEqual(img.shape, (self.default_config.width, self.default_config.height, 3))

    @pytest.mark.skipif(LEROBOT_USE_REAL_OPENCV_CAMERA_TESTS, reason="Not valid for real camera")
    def test_read_capture_fails(self):
        FakeOpenCVSDKAdapter.configure_fail_read_after(fail_read_after=0)  # Simulate read failure
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        cam.connect()

        with self.assertRaisesRegex(OSError, f"Can't capture color image from camera {cam.camera_index}"):
            cam.read()

    def test_read_invalid_temporary_color_mode(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        cam.connect()

        with self.assertRaisesRegex(
            ValueError, "Expected color values are 'rgb' or 'bgr', but xyz is provided"
        ):
            cam.read(temporary_color_mode="xyz")

    @pytest.mark.skipif(LEROBOT_USE_REAL_OPENCV_CAMERA_TESTS, reason="Not valid for real camera")
    def test_read_dimension_mismatch(self):
        wrong_dim_image = create_dummy_image(240, 320)
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        cam.connect()
        FakeOpenCVSDKAdapter.configure_fake_simulated_image(
            simulated_image=wrong_dim_image
        )  # Different dimensions

        with self.assertRaisesRegex(
            OSError,
            f"Can't capture color image with expected height and width \\({MOCK_CV2_CAP_PROP_FRAME_HEIGHT} x {MOCK_CV2_CAP_PROP_FRAME_WIDTH}\\). \\(240 x 320\\) returned instead.",
        ):
            cam.read()

    # --- Async Read Tests ---
    def test_async_read_not_connected(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        with self.assertRaisesRegex(DeviceNotConnectedError, "not connected"):
            cam.async_read()

    # TODO(Steven): This is dirty, but at least we don't have to manually mock the read
    def test_async_read_starts_thread_and_returns_image(
        self,
    ):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        cam.connect()

        def mock_start():
            cam.read_loop()

        self.mock_thread_instance.start.side_effect = mock_start

        # Make Event.is_set return False initially, then True after disconnect/stop
        self.mock_event_instance.is_set.side_effect = [False, True]

        img = cam.async_read()

        # Assert thread was created and started
        self.mock_event_class.assert_called_once()
        self.mock_thread_class.assert_called_once_with(target=cam.read_loop, args=())
        self.mock_thread_instance.start.assert_called_once()
        self.assertTrue(self.mock_thread_instance.daemon)
        self.assertIsNotNone(cam.thread)
        self.assertIsNotNone(cam.stop_event)

        if isinstance(self.test_sdk, FakeOpenCVSDKAdapter):
            # When using fake SDK, verify exact match with dummy image
            np.testing.assert_array_equal(img, self.dummy_rgb_image)
        else:
            # When using real camera, verify basic properties of the captured image
            self.assertIsInstance(img, np.ndarray)

    def test_async_read_timeout(self):
        config = self.default_config
        cam = OpenCVCamera(config, cv2_sdk=self.test_sdk)
        cam.connect()

        with self.assertRaisesRegex(TimeoutError, "Timed out waiting for async_read"):
            cam.async_read()

        # Assert thread was created and started
        self.mock_event_class.assert_called_once()
        self.mock_thread_class.assert_called_once_with(target=cam.read_loop, args=())
        self.mock_thread_instance.start.assert_called_once()
        self.assertTrue(self.mock_thread_instance.daemon)
        self.assertIsNotNone(cam.thread)
        self.assertIsNotNone(cam.stop_event)

    # --- Disconnection Tests ---
    def test_disconnect_not_connected(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        with self.assertRaisesRegex(DeviceNotConnectedError, "not connected"):
            cam.disconnect()

    def test_disconnect_no_thread(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        cam.connect()

        cam.disconnect()

        self.assertIsNone(cam.camera)
        self.assertFalse(cam.is_connected)
        self.assertIsNone(cam.thread)
        self.assertIsNone(cam.stop_event)

    def test_disconnect_with_thread(self):
        cam = OpenCVCamera(self.default_config, cv2_sdk=self.test_sdk)
        cam.connect()

        def mock_start():
            cam.read_loop()

        self.mock_thread_instance.start.side_effect = mock_start

        # Make Event.is_set return False initially, then True after disconnect/stop
        self.mock_event_instance.is_set.side_effect = [False, True]

        _img = cam.async_read()

        self.assertIsNotNone(cam.thread)
        self.assertIsNotNone(cam.stop_event)

        cam.disconnect()

        # Check thread management
        self.mock_event_instance.set.assert_called_once()
        self.mock_thread_instance.join.assert_called_once()

        # Check state reset
        self.assertIsNone(cam.camera)
        self.assertFalse(cam.is_connected)
        self.assertIsNone(cam.thread)
        self.assertIsNone(cam.stop_event)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

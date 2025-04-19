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
# Note: The current test approach uses mock/patch per management decision.
# Future maintainers may consider Dependency Injection for improved testability.
# As it currently stands, it is brittle, requires complex mocks and discourages refactoring.
# Everytime that we change the implementation code, we might need to change the tests.
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np

from lerobot.common.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# We might need to mock these
from lerobot.common.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)

MODULE_PATH = "lerobot.common.cameras.opencv.camera_opencv"

# Define constants that might be used by mocks
MOCK_CV2_CAP_PROP_FPS = 30
MOCK_CV2_CAP_PROP_FRAME_WIDTH = 1280
MOCK_CV2_CAP_PROP_FRAME_HEIGHT = 720
MOCK_CV2_ROTATE_90_CLOCKWISE = 90
MOCK_CV2_ROTATE_180 = 180
MOCK_CV2_ROTATE_90_COUNTERCLOCKWISE = -90
MOCK_CV2_COLOR_BGR2RGB = 99
MOCK_CV2_CAP_V4L2 = 91
MOCK_CV2_CAP_DSHOW = 92
MOCK_CV2_CAP_AVFOUNDATION = 93
MOCK_CV2_CAP_ANY = 90


# Helper function to create a realistic-looking dummy image
def create_dummy_image(height, width, channels=3):
    return np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)


# --- Test Class ---


class TestOpenCVCamera(unittest.TestCase):
    def setUp(self):
        # Default config used in many tests
        self.default_config = OpenCVCameraConfig(
            camera_index=0,
            width=MOCK_CV2_CAP_PROP_FRAME_WIDTH,
            height=MOCK_CV2_CAP_PROP_FRAME_HEIGHT,
            fps=MOCK_CV2_CAP_PROP_FPS,
            color_mode="rgb",
            rotation=None,
        )
        # Create a default dummy image based on config
        self.dummy_bgr_image = create_dummy_image(self.default_config.height, self.default_config.width)
        self.dummy_rgb_image = self.dummy_bgr_image[..., ::-1]  # Simple BGR -> RGB simulation

        self.patch_dependencies()

    # --- Mock Setup ---
    def patch_dependencies(self):
        # Mock the cv2 module itself
        patcher_cv2 = patch(f"{MODULE_PATH}.cv2", autospec=True)
        self.mock_cv2 = patcher_cv2.start()
        self.addCleanup(patcher_cv2.stop)

        # Mock the VideoCapture class returned by cv2.VideoCapture
        self.mock_capture_instance = MagicMock()
        self.mock_cv2.VideoCapture.return_value = self.mock_capture_instance

        # Assign mock constants (important for comparisons and calls)
        self.mock_cv2.CAP_PROP_FPS = MOCK_CV2_CAP_PROP_FPS
        self.mock_cv2.CAP_PROP_FRAME_WIDTH = MOCK_CV2_CAP_PROP_FRAME_WIDTH
        self.mock_cv2.CAP_PROP_FRAME_HEIGHT = MOCK_CV2_CAP_PROP_FRAME_HEIGHT
        self.mock_cv2.ROTATE_90_CLOCKWISE = MOCK_CV2_ROTATE_90_CLOCKWISE
        self.mock_cv2.ROTATE_180 = MOCK_CV2_ROTATE_180
        self.mock_cv2.ROTATE_90_COUNTERCLOCKWISE = MOCK_CV2_ROTATE_90_COUNTERCLOCKWISE
        self.mock_cv2.COLOR_BGR2RGB = MOCK_CV2_COLOR_BGR2RGB
        self.mock_cv2.CAP_V4L2 = MOCK_CV2_CAP_V4L2
        self.mock_cv2.CAP_DSHOW = MOCK_CV2_CAP_DSHOW
        self.mock_cv2.CAP_AVFOUNDATION = MOCK_CV2_CAP_AVFOUNDATION
        self.mock_cv2.CAP_ANY = MOCK_CV2_CAP_ANY

        patcher_platform = patch(f"{MODULE_PATH}.platform.system")
        self.mock_platform_system = patcher_platform.start()
        self.addCleanup(patcher_platform.stop)
        self.mock_platform_system.return_value = "Darwin"  # Default to macOS

        mock_is_valid_unix_path = patch(f"{MODULE_PATH}.is_valid_unix_path")
        self.mock_is_valid_unix_path = mock_is_valid_unix_path.start()
        self.addCleanup(mock_is_valid_unix_path.stop)
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

        # Mock internal utility functions if they interact with system/cv2
        patcher_find_cameras = patch(f"{MODULE_PATH}.find_cameras")
        self.mock_find_cameras = patcher_find_cameras.start()
        self.addCleanup(patcher_find_cameras.stop)

        patcher_get_index = patch(f"{MODULE_PATH}.get_camera_index_from_unix_port")
        self.mock_get_camera_index_from_unix_port = patcher_get_index.start()
        self.addCleanup(patcher_get_index.stop)

    # --- Test __init__ ---
    def test_init_defaults(self):
        cam = OpenCVCamera(self.default_config)

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
        cam = OpenCVCamera(config)
        self.assertEqual(cam.width, MOCK_CV2_CAP_PROP_FRAME_HEIGHT)  # Swapped
        self.assertEqual(cam.height, MOCK_CV2_CAP_PROP_FRAME_WIDTH)  # Swapped
        self.assertEqual(cam.rotation, MOCK_CV2_ROTATE_90_CLOCKWISE)

    def test_init_with_rotation_minus_90(self):
        config = self.default_config
        config.rotation = -90
        cam = OpenCVCamera(config)
        self.assertEqual(cam.width, MOCK_CV2_CAP_PROP_FRAME_HEIGHT)  # Swapped
        self.assertEqual(cam.height, MOCK_CV2_CAP_PROP_FRAME_WIDTH)  # Swapped
        self.assertEqual(cam.rotation, MOCK_CV2_ROTATE_90_COUNTERCLOCKWISE)

    def test_init_with_rotation_180(self):
        config = self.default_config
        config.rotation = 180
        cam = OpenCVCamera(config)
        self.assertEqual(cam.width, MOCK_CV2_CAP_PROP_FRAME_WIDTH)  # Not swapped
        self.assertEqual(cam.height, MOCK_CV2_CAP_PROP_FRAME_HEIGHT)  # Not swapped
        self.assertEqual(cam.rotation, MOCK_CV2_ROTATE_180)

    def test_init_linux_with_index(self):
        self.mock_platform_system.return_value = "Linux"
        cam = OpenCVCamera(self.default_config)
        self.assertEqual(cam.camera_index, 0)
        self.assertIsInstance(cam.port, Path)
        self.assertEqual(str(cam.port), "/dev/video0")

    def test_init_linux_with_valid_path(self):
        self.mock_platform_system.return_value = "Linux"
        self.mock_is_valid_unix_path.return_value = True
        self.mock_get_camera_index_from_unix_port.return_value = 2
        config = self.default_config
        config.camera_index = "/dev/video2"
        cam = OpenCVCamera(config)

        self.assertIsInstance(cam.port, Path)
        self.assertEqual(str(cam.port), "/dev/video2")
        self.assertEqual(cam.camera_index, 2)
        self.mock_is_valid_unix_path.assert_called_once_with("/dev/video2")

    def test_init_linux_with_invalid_path(self):
        self.mock_platform_system.return_value = "Linux"
        self.mock_is_valid_unix_path.return_value = False
        config = self.default_config
        config.camera_index = "[*?:[/invalid/path"
        with self.assertRaisesRegex(ValueError, "Please check the provided camera_index"):
            OpenCVCamera(config)

    # --- Test connect ---
    def test_connect_success(self):
        cam = OpenCVCamera(self.default_config)

        # Mock the temporary camera check
        mock_tmp_capture = MagicMock()
        mock_tmp_capture.isOpened.return_value = True
        # First call to VideoCapture is the temporary one
        # Second call is the real one
        self.mock_cv2.VideoCapture.side_effect = [mock_tmp_capture, self.mock_capture_instance]

        # Mock the actual camera setup
        self.mock_capture_instance.get.side_effect = [
            self.default_config.fps,
            self.default_config.width,
            self.default_config.height,
        ]

        cam.connect()

        # Check temporary camera interactions
        self.assertEqual(self.mock_cv2.VideoCapture.call_count, 2)
        self.mock_cv2.VideoCapture.assert_any_call(
            self.default_config.camera_index, MOCK_CV2_CAP_AVFOUNDATION
        )
        mock_tmp_capture.isOpened.assert_called_once()
        mock_tmp_capture.release.assert_called_once()

        # Check actual camera interactions
        self.mock_capture_instance.set.assert_has_calls(
            [
                call(MOCK_CV2_CAP_PROP_FPS, self.default_config.fps),
                call(MOCK_CV2_CAP_PROP_FRAME_WIDTH, self.default_config.width),
                call(MOCK_CV2_CAP_PROP_FRAME_HEIGHT, self.default_config.height),
            ]
        )
        self.mock_capture_instance.get.assert_has_calls(
            [
                call(MOCK_CV2_CAP_PROP_FPS),
                call(MOCK_CV2_CAP_PROP_FRAME_WIDTH),
                call(MOCK_CV2_CAP_PROP_FRAME_HEIGHT),
            ]
        )
        self.assertTrue(cam.is_connected)
        self.assertIsNotNone(cam.camera)
        self.mock_cv2.setNumThreads.assert_called_once_with(1)
        # Check stored actual values (after rounding)
        self.assertEqual(cam.fps, self.default_config.fps)
        self.assertEqual(cam.capture_width, self.default_config.width)
        self.assertEqual(cam.capture_height, self.default_config.height)

    def test_connect_success_linux(self):
        self.mock_platform_system.return_value = "Linux"
        cam = OpenCVCamera(self.default_config)

        mock_tmp_capture = MagicMock()
        mock_tmp_capture.isOpened.return_value = True
        self.mock_cv2.VideoCapture.side_effect = [mock_tmp_capture, self.mock_capture_instance]
        self.mock_capture_instance.get.side_effect = [
            MOCK_CV2_CAP_PROP_FPS,
            MOCK_CV2_CAP_PROP_FRAME_WIDTH,
            MOCK_CV2_CAP_PROP_FRAME_HEIGHT,
        ]  # Floats are ok

        cam.connect()

        self.assertTrue(cam.is_connected)
        self.assertIsNotNone(cam.camera)
        self.assertEqual(self.mock_cv2.VideoCapture.call_count, 2)
        # Check it uses the port path and correct backend
        self.mock_cv2.VideoCapture.assert_any_call("/dev/video0", MOCK_CV2_CAP_V4L2)

    def test_connect_already_connected(self):
        cam = OpenCVCamera(self.default_config)
        # Simulate already connected state
        cam.is_connected = True
        with self.assertRaisesRegex(DeviceAlreadyConnectedError, "already connected"):
            cam.connect()
        cam.is_connected = False  # To avoid warning when running the destructor

    def test_connect_camera_not_opened_invalid_index(self):
        config = self.default_config
        config.camera_index = 99
        cam = OpenCVCamera(config)

        # Mock temporary check failing
        mock_tmp_capture = MagicMock()
        mock_tmp_capture.isOpened.return_value = False
        self.mock_cv2.VideoCapture.return_value = mock_tmp_capture  # Only tmp is created

        # Mock find_cameras returning *other* indices
        self.mock_find_cameras.return_value = [
            {"index": 0, "port": None},
            {"index": 1, "port": None},
            {"index": 2, "port": None},
        ]

        with self.assertRaisesRegex(
            ValueError, "expected to be one of these available cameras \\[0, 1, 2\\], but 99 is provided"
        ):
            cam.connect()

        self.assertFalse(cam.is_connected)
        self.mock_find_cameras.assert_called_once()
        mock_tmp_capture.release.assert_called_once()  # Should still release tmp capture

    def test_connect_fps_mismatch(self):
        config = self.default_config
        config.fps = 999
        cam = OpenCVCamera(config)

        mock_tmp_capture = MagicMock()
        mock_tmp_capture.isOpened.return_value = True
        self.mock_cv2.VideoCapture.side_effect = [mock_tmp_capture, self.mock_capture_instance]

        # Mock get to return a different FPS
        self.mock_capture_instance.get.side_effect = [
            MOCK_CV2_CAP_PROP_FPS,  # Different FPS
            self.default_config.width,
            self.default_config.height,
        ]

        with self.assertRaisesRegex(
            OSError, f"Can't set self.fps=999 .* Actual value is {MOCK_CV2_CAP_PROP_FPS}"
        ):
            cam.connect()
        self.assertFalse(cam.is_connected)

    def test_connect_width_mismatch(self):
        config = self.default_config
        config.width = 9999
        cam = OpenCVCamera(config)

        mock_tmp_capture = MagicMock()
        mock_tmp_capture.isOpened.return_value = True
        self.mock_cv2.VideoCapture.side_effect = [mock_tmp_capture, self.mock_capture_instance]

        self.mock_capture_instance.get.side_effect = [
            self.default_config.fps,
            MOCK_CV2_CAP_PROP_FRAME_WIDTH,  # Different width
            self.default_config.height,
        ]

        with self.assertRaisesRegex(
            OSError, f"Can't set self.capture_width=9999 .* Actual value is {MOCK_CV2_CAP_PROP_FRAME_WIDTH}"
        ):
            cam.connect()
        self.assertFalse(cam.is_connected)

    def test_connect_height_mismatch(self):
        config = self.default_config
        config.height = 9999
        cam = OpenCVCamera(config)

        mock_tmp_capture = MagicMock()
        mock_tmp_capture.isOpened.return_value = True
        self.mock_cv2.VideoCapture.side_effect = [mock_tmp_capture, self.mock_capture_instance]

        self.mock_capture_instance.get.side_effect = [
            self.default_config.fps,
            self.default_config.width,
            MOCK_CV2_CAP_PROP_FRAME_HEIGHT,  # Different height
        ]

        with self.assertRaisesRegex(
            OSError, f"Can't set self.capture_height=9999 .* Actual value is {MOCK_CV2_CAP_PROP_FRAME_HEIGHT}"
        ):
            cam.connect()
        self.assertFalse(cam.is_connected)

    # --- Test read ---
    def test_read_not_connected(self):
        cam = OpenCVCamera(self.default_config)
        with self.assertRaisesRegex(DeviceNotConnectedError, "not connected"):
            cam.read()

    def test_read_success_rgb(self):
        cam = OpenCVCamera(self.default_config)
        cam.camera = self.mock_capture_instance  # Simulate connection
        cam.is_connected = True
        cam.capture_width = MOCK_CV2_CAP_PROP_FRAME_WIDTH  # Set dimensions as if connect succeeded
        cam.capture_height = MOCK_CV2_CAP_PROP_FRAME_HEIGHT

        self.mock_capture_instance.read.return_value = (True, self.dummy_bgr_image)
        self.mock_cv2.cvtColor.return_value = self.dummy_rgb_image  # Mock conversion

        img = cam.read()

        self.mock_capture_instance.read.assert_called_once()
        self.mock_cv2.cvtColor.assert_called_once_with(self.dummy_bgr_image, MOCK_CV2_COLOR_BGR2RGB)
        self.mock_cv2.rotate.assert_not_called()

        np.testing.assert_array_equal(img, self.dummy_rgb_image)

    def test_read_success_bgr(self):
        config = self.default_config
        config.color_mode = "bgr"  # Request BGR
        cam = OpenCVCamera(config)
        cam.camera = self.mock_capture_instance
        cam.is_connected = True
        cam.capture_width = MOCK_CV2_CAP_PROP_FRAME_WIDTH
        cam.capture_height = MOCK_CV2_CAP_PROP_FRAME_HEIGHT

        self.mock_capture_instance.read.return_value = (True, self.dummy_bgr_image)

        img = cam.read()

        self.mock_capture_instance.read.assert_called_once()
        self.mock_cv2.cvtColor.assert_not_called()  # Should not convert
        np.testing.assert_array_equal(img, self.dummy_bgr_image)  # Expect BGR

    def test_read_success_with_rotation(self):
        config = self.default_config
        config.rotation = 90
        cam = OpenCVCamera(config)
        cam.camera = self.mock_capture_instance
        cam.is_connected = True
        cam.capture_width = MOCK_CV2_CAP_PROP_FRAME_WIDTH  # Original capture dimensions
        cam.capture_height = MOCK_CV2_CAP_PROP_FRAME_HEIGHT

        self.mock_capture_instance.read.return_value = (True, self.dummy_bgr_image)
        self.mock_cv2.cvtColor.return_value = self.dummy_rgb_image
        rotated_image = np.rot90(self.dummy_rgb_image)  # Simulate rotation
        self.mock_cv2.rotate.return_value = rotated_image  # Mock rotation result

        img = cam.read()

        self.mock_capture_instance.read.assert_called_once()
        self.mock_cv2.cvtColor.assert_called_once_with(self.dummy_bgr_image, MOCK_CV2_COLOR_BGR2RGB)
        self.mock_cv2.rotate.assert_called_once_with(self.dummy_rgb_image, MOCK_CV2_ROTATE_90_CLOCKWISE)
        np.testing.assert_array_equal(img, rotated_image)

    def test_read_capture_fails(self):
        cam = OpenCVCamera(self.default_config)
        cam.camera = self.mock_capture_instance
        cam.is_connected = True

        self.mock_capture_instance.read.return_value = (False, None)  # Simulate read failure

        with self.assertRaisesRegex(OSError, f"Can't capture color image from camera {cam.camera_index}"):
            cam.read()

    def test_read_invalid_temporary_color_mode(self):
        cam = OpenCVCamera(self.default_config)
        cam.camera = self.mock_capture_instance
        cam.is_connected = True
        cam.capture_width = MOCK_CV2_CAP_PROP_FRAME_WIDTH
        cam.capture_height = MOCK_CV2_CAP_PROP_FRAME_HEIGHT
        self.mock_capture_instance.read.return_value = (
            True,
            self.dummy_bgr_image,
        )  # Need read to succeed initially

        with self.assertRaisesRegex(
            ValueError, "Expected color values are 'rgb' or 'bgr', but xyz is provided"
        ):
            cam.read(temporary_color_mode="xyz")

    def test_read_dimension_mismatch(self):
        wrong_dim_image = create_dummy_image(240, 320)  # Different dimensions
        cam = OpenCVCamera(self.default_config)

        cam.camera = self.mock_capture_instance
        cam.is_connected = True
        cam.capture_width = MOCK_CV2_CAP_PROP_FRAME_WIDTH  # Expected dims
        cam.capture_height = MOCK_CV2_CAP_PROP_FRAME_HEIGHT

        self.mock_capture_instance.read.return_value = (True, wrong_dim_image)
        self.mock_cv2.cvtColor.return_value = wrong_dim_image

        with self.assertRaisesRegex(
            OSError,
            f"Can't capture color image with expected height and width \\({MOCK_CV2_CAP_PROP_FRAME_HEIGHT} x {MOCK_CV2_CAP_PROP_FRAME_WIDTH}\\). \\(240 x 320\\) returned instead.",
        ):
            cam.read()

    # --- Test async_read ---
    def test_async_read_not_connected(self):
        cam = OpenCVCamera(self.default_config)
        with self.assertRaisesRegex(DeviceNotConnectedError, "not connected"):
            cam.async_read()

    # TODO(Steven): Not sure about this one
    @patch.object(OpenCVCamera, "read", autospec=True)  # Mock the instance's read method
    def test_async_read_starts_thread_and_returns_image(self, mock_instance_read):
        cam = OpenCVCamera(self.default_config)
        cam.is_connected = True  # Simulate connection

        # TODO(Steven): This is dirty
        def mock_read_effect(self_cam_instance):
            self_cam_instance.color_image = self.dummy_rgb_image
            return self.dummy_rgb_image

        mock_instance_read.side_effect = mock_read_effect

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
        mock_instance_read.assert_called_once_with(cam)

        cam.is_connected = False  # To avoid warning when running the destructor

        # Assert the image returned by async_read is the one set by the mocked read
        np.testing.assert_array_equal(img, self.dummy_rgb_image)

    @patch.object(OpenCVCamera, "read", autospec=True)
    def test_async_read_timeout(self, mock_instance_read):
        config = self.default_config
        cam = OpenCVCamera(config)
        cam.is_connected = True

        # Mock 'read' so it *never* sets color_image
        def mock_read_effect(self_cam_instance):
            self_cam_instance.color_image = None
            return None

        mock_instance_read.side_effect = mock_read_effect

        with self.assertRaisesRegex(TimeoutError, "Timed out waiting for async_read"):
            cam.async_read()

        # Assert thread was created and started
        self.mock_event_class.assert_called_once()
        self.mock_thread_class.assert_called_once_with(target=cam.read_loop, args=())
        self.mock_thread_instance.start.assert_called_once()
        self.assertTrue(self.mock_thread_instance.daemon)
        self.assertIsNotNone(cam.thread)
        self.assertIsNotNone(cam.stop_event)

        cam.is_connected = False  # To avoid warning when running the destructor

    # --- Test disconnect ---
    def test_disconnect_not_connected(self):
        cam = OpenCVCamera(self.default_config)
        with self.assertRaisesRegex(DeviceNotConnectedError, "not connected"):
            cam.disconnect()

    def test_disconnect_no_thread(self):
        cam = OpenCVCamera(self.default_config)
        cam.camera = self.mock_capture_instance
        cam.is_connected = True

        cam.disconnect()

        self.assertIsNone(cam.camera)
        self.assertFalse(cam.is_connected)
        self.assertIsNone(cam.thread)
        self.assertIsNone(cam.stop_event)

        self.mock_capture_instance.release.assert_called_once()
        self.mock_event_instance.set.assert_not_called()
        self.mock_thread_instance.join.assert_not_called()

    @patch.object(OpenCVCamera, "read", autospec=True)
    def test_disconnect_with_thread(self, mock_instance_read):
        cam = OpenCVCamera(self.default_config)
        cam.camera = self.mock_capture_instance
        cam.is_connected = True

        # TODO(Steven): This is dirty
        def mock_read_effect(self_cam_instance):
            self_cam_instance.color_image = self.dummy_rgb_image
            return self.dummy_rgb_image

        mock_instance_read.side_effect = mock_read_effect

        def mock_start():
            cam.read_loop()

        self.mock_thread_instance.start.side_effect = mock_start

        # Make Event.is_set return False initially, then True after disconnect/stop
        self.mock_event_instance.is_set.side_effect = [False, True]

        # Start the thread
        _img = cam.async_read()

        self.assertIsNotNone(cam.thread)
        self.assertIsNotNone(cam.stop_event)

        cam.disconnect()

        # Check thread management
        self.mock_event_instance.set.assert_called_once()
        self.mock_thread_instance.join.assert_called_once()

        # Check camera release
        self.mock_capture_instance.release.assert_called_once()

        # Check state reset
        self.assertIsNone(cam.camera)
        self.assertFalse(cam.is_connected)
        self.assertIsNone(cam.thread)
        self.assertIsNone(cam.stop_event)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

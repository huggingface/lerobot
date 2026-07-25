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

from lerobot.cameras.opencv import camera_opencv
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.linux_v4l2 import find_linux_video_devices


def test_find_cameras_linux_does_not_open_with_opencv(monkeypatch):
    fake_cameras = [
        {
            "name": "OpenCV Camera @ /dev/video0",
            "type": "OpenCV",
            "id": "/dev/video0",
            "backend_api": "V4L2",
            "default_stream_profile": {
                "format": None,
                "fourcc": None,
                "width": None,
                "height": None,
                "fps": None,
            },
        }
    ]

    def fail_if_opened(*args, **kwargs):
        raise AssertionError("Linux discovery must not open cameras through OpenCV")

    monkeypatch.setattr(camera_opencv.platform, "system", lambda: "Linux")
    monkeypatch.setattr(camera_opencv, "find_linux_video_devices", lambda: fake_cameras)
    monkeypatch.setattr(camera_opencv.cv2, "VideoCapture", fail_if_opened)

    assert OpenCVCamera.find_cameras() == fake_cameras


def test_safe_linux_enumerator_returns_devices_without_profiles(tmp_path):
    sysfs_root = tmp_path / "sys" / "class" / "video4linux"
    dev_root = tmp_path / "dev"
    video0 = sysfs_root / "video0"
    video2 = sysfs_root / "video2"
    video0.mkdir(parents=True)
    video2.mkdir()
    dev_root.mkdir()
    (video0 / "name").write_text("Front Camera\n", encoding="utf-8")
    (dev_root / "video0").touch()
    (dev_root / "video2").touch()

    cameras = find_linux_video_devices(sysfs_video4linux_path=sysfs_root, dev_path=dev_root)

    assert cameras == [
        {
            "name": f"Front Camera @ {dev_root / 'video0'}",
            "type": "OpenCV",
            "id": str(dev_root / "video0"),
            "backend_api": "V4L2",
            "default_stream_profile": {
                "format": None,
                "fourcc": None,
                "width": None,
                "height": None,
                "fps": None,
            },
        },
        {
            "name": f"OpenCV Camera @ {dev_root / 'video2'}",
            "type": "OpenCV",
            "id": str(dev_root / "video2"),
            "backend_api": "V4L2",
            "default_stream_profile": {
                "format": None,
                "fourcc": None,
                "width": None,
                "height": None,
                "fps": None,
            },
        },
    ]


def test_safe_linux_enumerator_falls_back_to_dev_paths(tmp_path):
    sysfs_root = tmp_path / "missing-sysfs"
    dev_root = tmp_path / "dev"
    dev_root.mkdir()
    (dev_root / "video1").touch()
    (dev_root / "video0").touch()

    cameras = find_linux_video_devices(sysfs_video4linux_path=sysfs_root, dev_path=dev_root)

    assert [camera["id"] for camera in cameras] == [str(dev_root / "video0"), str(dev_root / "video1")]
    assert all(camera["default_stream_profile"]["width"] is None for camera in cameras)


def test_non_linux_discovery_uses_opencv_probe(monkeypatch):
    captures = []

    class FakeVideoCapture:
        def __init__(self, target):
            self.target = target
            self.released = False
            captures.append(self)

        def isOpened(self):  # noqa: N802
            return self.target == 0

        def get(self, prop):
            values = {
                camera_opencv.cv2.CAP_PROP_FRAME_WIDTH: 1280,
                camera_opencv.cv2.CAP_PROP_FRAME_HEIGHT: 720,
                camera_opencv.cv2.CAP_PROP_FPS: 30,
                camera_opencv.cv2.CAP_PROP_FORMAT: 16,
                camera_opencv.cv2.CAP_PROP_FOURCC: 1196444237,
            }
            return values[prop]

        def getBackendName(self):  # noqa: N802
            return "MOCK"

        def release(self):
            self.released = True

    monkeypatch.setattr(camera_opencv.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(camera_opencv, "MAX_OPENCV_INDEX", 2)
    monkeypatch.setattr(camera_opencv.cv2, "VideoCapture", FakeVideoCapture)

    assert OpenCVCamera.find_cameras() == [
        {
            "name": "OpenCV Camera @ 0",
            "type": "OpenCV",
            "id": 0,
            "backend_api": "MOCK",
            "default_stream_profile": {
                "format": 16,
                "fourcc": "MJPG",
                "width": 1280,
                "height": 720,
                "fps": 30,
            },
        }
    ]
    assert [capture.released for capture in captures] == [True, True]
